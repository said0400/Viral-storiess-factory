from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import sys
import unicodedata
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
import trafilatura
from bs4 import BeautifulSoup, UnicodeDammit

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_TEXT_OK = True
except ImportError:  # يُتخطى النص على الصورة مع تحذير
    ARABIC_TEXT_OK = False
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    ImageOps,
    ImageStat,
    UnidentifiedImageError,
)

import gemini
import thumbnail
import wide_collage
import writer
from grok import (
    MAX_VISION_IMAGES,
    default_gallery,
    GrokError,
    analyze_images,
    fallback_plan,
)


OUTPUT_ROOT = Path("output")
IMAGE_SIZE = 1080
SS = 2                      # دقة مضاعفة لنعومة الحواف
GUTTER = 10
GUTTER_COLOR = "#FFFFFF"

FONT_FILENAME = "IBMPlexSansArabic-Bold.ttf"
FONT_FALLBACKS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)
TEXT_ACCENT = "#FFD400"
MAX_PAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_ATTEMPTS = 14
MIN_IMAGE_WIDTH = 400
MIN_IMAGE_HEIGHT = 300
REQUEST_TIMEOUT = 30

JUNK_IMAGE_HINTS = (
    "logo", "icon", "avatar", "sprite", "pixel", "tracking",
    "spacer", "placeholder", "banner-ad", "/ads/", "advert",
    "emoji", "gravatar",
)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "ar,en;q=0.8",
})


class ProjectError(Exception):
    """خطأ متوقع أثناء استخراج المقال أو إنشاء النتائج."""


def validate_public_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ProjectError("رابط المقال فارغ.")

    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme.lower() not in ("http", "https"):
        raise ProjectError("يجب أن يبدأ الرابط بـ http:// أو https://.")

    if not parsed.hostname or parsed.username or parsed.password:
        raise ProjectError("الرابط غير صالح أو يحتوي بيانات دخول.")

    hostname = parsed.hostname.rstrip(".").lower()

    if hostname in {"localhost", "localhost.localdomain"}:
        raise ProjectError("الروابط المحلية غير مسموحة.")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(
            hostname, port, type=socket.SOCK_STREAM
        )
    except (ValueError, socket.gaierror) as exc:
        raise ProjectError(
            "تعذر التحقق من عنوان خادم الرابط."
        ) from exc

    if not addresses:
        raise ProjectError("لم يتم العثور على عنوان IP للموقع.")

    for address in addresses:
        raw_ip = address[4][0].split("%")[0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise ProjectError("عنوان IP غير صالح.") from exc

        if not ip.is_global:
            raise ProjectError(
                "تم رفض الرابط لأنه يشير إلى عنوان IP غير عام."
            )

    return url


def safe_get(
    url: str,
    max_bytes: int,
    expected_image: bool = False,
):
    current_url = url

    for _ in range(6):
        validate_public_url(current_url)

        try:
            response = SESSION.get(
                current_url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as exc:
            raise ProjectError(f"فشل الاتصال بالرابط: {exc}") from exc

        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            response.close()

            if not location:
                raise ProjectError("تحويل الرابط لا يحتوي على وجهة.")

            current_url = urljoin(current_url, location)
            continue

        if not 200 <= response.status_code < 300:
            status = response.status_code
            response.close()
            raise ProjectError(f"أعاد الموقع HTTP {status}.")

        content_type = response.headers.get("Content-Type", "").lower()

        if expected_image and not content_type.startswith("image/"):
            response.close()
            raise ProjectError("الرابط لا يعيد ملف صورة.")

        length = response.headers.get("Content-Length", "")
        if length:
            try:
                if int(length) > max_bytes:
                    response.close()
                    raise ProjectError("حجم الملف أكبر من الحد المسموح.")
            except ValueError:
                pass

        data = bytearray()

        try:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ProjectError("حجم الملف أكبر من الحد المسموح.")
        except requests.RequestException as exc:
            raise ProjectError(f"انقطع تنزيل الملف: {exc}") from exc
        finally:
            response.close()

        return response.url or current_url, content_type, bytes(data)

    raise ProjectError("تجاوز الرابط عدد التحويلات المسموح به.")


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def decode_html(data: bytes, content_type: str) -> str:
    match = re.search(r"charset=([\w\-]+)", content_type or "")
    declared = match.group(1) if match else None

    candidates = [declared] if declared else []
    decoded = UnicodeDammit(
        data,
        known_definite_encodings=candidates,
        is_html=True,
    )

    if decoded.unicode_markup:
        return decoded.unicode_markup

    return data.decode("utf-8", errors="replace")


def extract_article(article_url: str) -> dict:
    final_url, content_type, page_bytes = safe_get(
        article_url, MAX_PAGE_BYTES
    )

    if "html" not in content_type and "xhtml" not in content_type:
        raise ProjectError("الرابط لا يشير إلى صفحة مقال HTML.")

    html = decode_html(page_bytes, content_type)
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    for selector in (
        'meta[property="og:title"]',
        'meta[name="twitter:title"]',
    ):
        node = soup.select_one(selector)
        if node:
            title = clean_text(node.get("content"))
            if title:
                break

    if not title and soup.title:
        title = clean_text(soup.title.get_text(" ", strip=True))

    if not title:
        heading = soup.find("h1")
        title = clean_text(heading.get_text(" ", strip=True)) if heading else ""

    if not title:
        title = "مقال بدون عنوان واضح"

    extracted = trafilatura.extract(
        html,
        url=final_url,
        include_comments=False,
        include_tables=True,
        include_links=False,
        favor_precision=True,
    )
    article_text = clean_text(extracted or "")

    if len(article_text) < 200:
        fallback = BeautifulSoup(html, "html.parser")
        for tag in fallback(
            ["script", "style", "noscript", "nav", "footer", "aside"]
        ):
            tag.decompose()

        candidates = []
        for selector in ("article", "main", '[role="main"]'):
            candidates.extend(fallback.select(selector))

        if candidates:
            candidates.sort(
                key=lambda node: len(node.get_text(" ", strip=True)),
                reverse=True,
            )
            article_text = clean_text(
                candidates[0].get_text(" ", strip=True)
            )

    if len(article_text) < 200:
        raise ProjectError(
            "لم أتمكن من استخراج نص كافٍ. قد يكون الموقع محميًا "
            "أو يحتاج إلى JavaScript أو تسجيل الدخول."
        )

    image_urls = []
    seen = set()

    def add_image(candidate):
        if not isinstance(candidate, str) or not candidate.strip():
            return

        candidate = candidate.strip()
        if candidate.startswith("data:"):
            return

        absolute = urldefrag(urljoin(final_url, candidate))[0]
        if urlparse(absolute).scheme not in ("http", "https"):
            return

        if any(hint in absolute.lower() for hint in JUNK_IMAGE_HINTS):
            return

        if absolute not in seen:
            seen.add(absolute)
            image_urls.append(absolute)

    for selector in (
        'meta[property="og:image"]',
        'meta[name="twitter:image"]',
    ):
        node = soup.select_one(selector)
        if node:
            add_image(node.get("content"))

    nodes = soup.select("article img, main img, [role='main'] img")
    if not nodes:
        nodes = soup.select("img")

    for node in nodes:
        for attribute in (
            "src", "data-src", "data-lazy-src", "data-original"
        ):
            add_image(node.get(attribute))

        srcset = node.get("srcset") or node.get("data-srcset")
        if srcset:
            entries = [
                item.strip().split()[0]
                for item in srcset.split(",")
                if item.strip()
            ]
            if entries:
                add_image(entries[-1])

    return {
        "source_url": final_url,
        "title": title,
        "text": article_text,
        "image_urls": image_urls[:30],
    }


def average_hash(image: Image.Image) -> int:
    small = image.convert("L").resize(
        (8, 8), Image.Resampling.LANCZOS
    )
    pixels = list(small.getdata())
    mean = sum(pixels) / len(pixels)

    result = 0
    for pixel in pixels:
        result = (result << 1) | int(pixel >= mean)
    return result


def is_duplicate(value: int, known: list[int]) -> bool:
    return any(
        bin(value ^ other).count("1") <= 5 for other in known
    )


def download_article_images(image_urls: list[str]):
    collected = []
    hashes = []

    for image_url in image_urls[:MAX_IMAGE_ATTEMPTS]:
        try:
            _, _, data = safe_get(
                image_url,
                MAX_IMAGE_BYTES,
                expected_image=True,
            )

            with Image.open(BytesIO(data)) as source:
                source.verify()

            with Image.open(BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")

            width, height = image.size
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            if width / height > 3.0 or width / height < 0.33:
                continue

            digest = average_hash(image)
            if is_duplicate(digest, hashes):
                continue

            hashes.append(digest)
            collected.append((image_url, image.copy()))

            if len(collected) >= 4:
                break

        except (
            ProjectError,
            requests.RequestException,
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
        ) as exc:
            print(f"تحذير: تخطي صورة غير صالحة: {exc}")

    if not collected:
        return []

    # تُحفظ أولوية الصورة الرئيسية المكتشفة أولًا.
    main_item = collected[0]
    rest = sorted(
        collected[1:],
        key=lambda item: item[1].width * item[1].height,
        reverse=True,
    )
    return [main_item] + rest[:3]


# ---------------------------------------------------------------------------
# تصميم الصورة
# ---------------------------------------------------------------------------

def smart_window(image, box, aspect, mode="cover",
                 padding=0.12, min_frac=0.2):
    """
    يحسب نافذة قص بنسبة أبعاد الخانة تمامًا وتحتوي العنصر المهم.
    cover: أكبر نافذة ممكنة متمركزة على العنصر (للخلفيات والألواح).
    tight: نافذة محكمة حول العنصر مع هامش (للتفصيل المكبر).
    """
    W, H = image.size
    l, t, r, b = box
    bl, br, bt, bb = l * W, r * W, t * H, b * H
    cx, cy = (bl + br) / 2, (bt + bb) / 2

    if W / H > aspect:
        max_w, max_h = H * aspect, H
    else:
        max_w, max_h = W, W / aspect

    if mode == "cover":
        win_w, win_h = max_w, max_h
    else:
        bw = max((br - bl) * (1 + 2 * padding), 1.0)
        bh = max((bb - bt) * (1 + 2 * padding), 1.0)
        win_w = bw if bw / bh > aspect else bh * aspect
        win_w = min(max(win_w, max_w * min_frac), max_w)
        win_h = win_w / aspect

    x0 = min(max(cx - win_w / 2, 0), W - win_w)
    y0 = min(max(cy - win_h / 2, 0), H - win_h)

    # إن كان العنصر يتسع في النافذة فاجعله داخلها بالكامل.
    if br - bl <= win_w:
        x0 = min(max(x0, br - win_w), bl)
    if bb - bt <= win_h:
        y0 = min(max(y0, bb - win_h), bt)
    x0 = min(max(x0, 0), W - win_w)
    y0 = min(max(y0, 0), H - win_h)

    return (x0, y0, x0 + win_w, y0 + win_h)


def crop_window(image, win, size):
    W, H = image.size
    x0 = max(0, min(W - 1, round(win[0])))
    y0 = max(0, min(H - 1, round(win[1])))
    x1 = max(x0 + 1, min(W, round(win[2])))
    y1 = max(y0 + 1, min(H, round(win[3])))
    return ImageOps.fit(
        image.crop((x0, y0, x1, y1)),
        size,
        method=Image.Resampling.LANCZOS,
    )


def polish(tile):
    tile = ImageEnhance.Contrast(tile).enhance(1.06)
    tile = ImageEnhance.Color(tile).enhance(1.10)
    return ImageEnhance.Sharpness(tile).enhance(1.15)


def choose_anchor(image, item, aspect):
    """
    إن اتسع العنصر الرئيسي في نافذة القص فهو المرجع، وإلا (لوحة نحيفة
    ووجه عريض مثلًا) تُثبَّت النافذة على التفصيل المهم كالعينين لئلا تُقطع.
    """
    W, H = image.size
    if W / H > aspect:
        max_w, max_h = H * aspect, H
    else:
        max_w, max_h = W, W / aspect

    sb = item["subject_box"]
    fits = (
        (sb[2] - sb[0]) * W <= max_w * 1.02
        and (sb[3] - sb[1]) * H <= max_h * 1.02
    )
    if fits:
        return sb
    return item.get("detail_box") or sb


def paste_tile(canvas, image, item, box, keep=None):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    anchor = choose_anchor(image, item, w / h)
    win = smart_window(image, anchor, w / h, "cover")
    canvas.paste(polish(crop_window(image, win, (w, h))), (x0, y0))

    if keep is not None:
        W, H = image.size
        ww, wh = win[2] - win[0], win[3] - win[1]
        for b in item.get("avoid_boxes", []):
            keep.append((
                (
                    x0 + (b[0] * W - win[0]) / ww * w,
                    y0 + (b[1] * H - win[1]) / wh * h,
                    x0 + (b[2] * W - win[0]) / ww * w,
                    y0 + (b[3] * H - win[1]) / wh * h,
                ),
                6.0,
            ))


def paste_inset(canvas, tile, x, y, shape, border):
    d = tile.width
    mask = Image.new("L", (d, d), 0)
    mdraw = ImageDraw.Draw(mask)
    if shape == "circle":
        mdraw.ellipse((0, 0, d - 1, d - 1), fill=255)
    else:
        mdraw.rounded_rectangle(
            (0, 0, d - 1, d - 1), radius=int(d * 0.07), fill=255
        )

    # ظل ناعم
    shadow = Image.new("L", canvas.size, 0)
    shadow.paste(mask, (x + 6 * SS, y + 8 * SS))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12 * SS))
    shadow = shadow.point(lambda v: int(v * 0.55))
    canvas.paste(
        (0, 0, 0), (0, 0, canvas.width, canvas.height), shadow
    )

    canvas.paste(tile, (x, y), mask)
    draw = ImageDraw.Draw(canvas)
    box = (x, y, x + d - 1, y + d - 1)
    if shape == "circle":
        draw.ellipse(box, outline="white", width=border)
    else:
        draw.rounded_rectangle(
            box, radius=int(d * 0.07), outline="white", width=border
        )


def window_is_flat(image, win, threshold=14.0):
    """منطقة شبه موحدة (سواد، سماء، جدار) لا تصلح للتكبير."""
    crop = image.crop(tuple(int(v) for v in win)).convert("L")
    crop = crop.resize((64, 64))
    return ImageStat.Stat(crop).stddev[0] < threshold


def make_detail_window(image, box, mw):
    """
    نافذة تفصيل مربعة بتكبير حقيقي (نحو 1.5x إلى 3x) قياسًا بالخلفية.
    تعيد (النافذة، نسبة قطر الدائرة الصغيرة إلى الإطار).
    """
    W, H = image.size
    tight = smart_window(image, box, 1.0, "tight", padding=0.08,
                         min_frac=0.05)
    r = (tight[2] - tight[0]) / mw
    d_frac = min(0.38, max(0.27, 1.4 * r))
    r = min(max(r, d_frac / 3.2), d_frac / 1.5)
    dw = min(r * mw, W, H)

    cx = (box[0] + box[2]) / 2 * W
    cy = (box[1] + box[3]) / 2 * H
    x0 = min(max(cx - dw / 2, 0), W - dw)
    y0 = min(max(cy - dw / 2, 0), H - dw)
    return (x0, y0, x0 + dw, y0 + dw), d_frac


def pick_detail(image, item, mw):
    """يجرّب صندوق النموذج ثم مركز العنصر، ويرفض المناطق الفارغة."""
    sb = item["subject_box"]
    cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
    hw, hh = (sb[2] - sb[0]) * 0.2, (sb[3] - sb[1]) * 0.2
    centered = [
        max(0.0, cx - hw), max(0.0, cy - hh),
        min(1.0, cx + hw), min(1.0, cy + hh),
    ]
    for box in (item["detail_box"], centered):
        win, d_frac = make_detail_window(image, box, mw)
        if not window_is_flat(image, win):
            return win, d_frac
    return None, None


def build_single(canvas, images, plan, keep=None):
    size = canvas.width
    item = plan["images"][0]
    img = images[item["index"]]
    W, H = img.size

    # 1) الخلفية: مربع كامل متمركز على العنصر
    main_win = smart_window(
        img, choose_anchor(img, item, 1.0), 1.0, "cover"
    )
    mx0, my0, mx1, _ = main_win
    mw = mx1 - mx0
    canvas.paste(
        polish(crop_window(img, main_win, (size, size))), (0, 0)
    )

    if keep is not None:
        for b in item.get("avoid_boxes", []):
            keep.append((
                (
                    (b[0] * W - mx0) / mw * size,
                    (b[1] * H - my0) / mw * size,
                    (b[2] * W - mx0) / mw * size,
                    (b[3] * H - my0) / mw * size,
                ),
                6.0,
            ))

    # 2) نافذة التفصيل
    dwin, d_frac = pick_detail(img, item, mw)
    if dwin is None:
        print("تحذير: لم تُوجد منطقة تفصيل مفيدة؛ صورة بلا دائرة.")
        return

    dcx, dcy = (dwin[0] + dwin[2]) / 2, (dwin[1] + dwin[3]) / 2
    dw = dwin[2] - dwin[0]
    d = int(size * d_frac)
    detail = polish(crop_window(img, dwin, (d, d)))
    print(f"نسبة التكبير الفعلية: {d / (dw / mw * size):.2f}x")

    # 3) موضع الحلقة على الخلفية
    ring_cx = (dcx - mx0) / mw * size
    ring_cy = (dcy - my0) / mw * size
    ring_r = dw / mw * size / 2

    # 4) اختيار الزاوية: أقل تغطية للعنصر وأهدأ منطقة وأبعد عن الحلقة
    l, t, r, b = item["subject_box"]
    sl, sr = (l * W - mx0) / mw * size, (r * W - mx0) / mw * size
    st, sb = (t * H - my0) / mw * size, (b * H - my0) / mw * size
    scx, scy = (sl + sr) / 2, (st + sb) / 2

    def to_canvas(box):
        return (
            (box[0] * W - mx0) / mw * size,
            (box[1] * H - my0) / mw * size,
            (box[2] * W - mx0) / mw * size,
            (box[3] * H - my0) / mw * size,
        )

    avoid_canvas = [to_canvas(b) for b in item.get("avoid_boxes", [])]

    grid = 256
    edges = canvas.convert("L").resize((grid, grid)).filter(
        ImageFilter.FIND_EDGES
    )
    k = grid / size

    def edge_density(px, py):
        box = (
            max(0, int(px * k)), max(0, int(py * k)),
            min(grid, int((px + d) * k) + 1),
            min(grid, int((py + d) * k) + 1),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return 1.0
        return min(1.0, ImageStat.Stat(edges.crop(box)).mean[0] / 40.0)

    def overlap_frac(px, py, ax0, ay0, ax1, ay1):
        ow = max(0, min(px + d, ax1) - max(px, ax0))
        oh = max(0, min(py + d, ay1) - max(py, ay0))
        return ow * oh / (d * d)

    m = int(size * 0.035)
    corners = [
        (size - d - m, m),
        (m, m),
        (size - d - m, size - d - m),
        (m, size - d - m),
    ]

    def penalty(pos):
        px, py = pos
        dist = ((px + d / 2 - scx) ** 2 + (py + d / 2 - scy) ** 2) ** 0.5
        return (
            3.0 * overlap_frac(px, py, sl, st, sr, sb)
            + 2.5 * sum(
                overlap_frac(px, py, *box) for box in avoid_canvas
            )
            + 4.0 * overlap_frac(
                px, py,
                ring_cx - ring_r, ring_cy - ring_r,
                ring_cx + ring_r, ring_cy + ring_r,
            )
            + 1.0 * edge_density(px, py)
            + (0.6 if py > size / 2 else 0.0)
            - 0.2 * dist / size
        )

    x, y = min(corners, key=penalty)

    # 5) حلقة على مصدر التكبير + خط يصلها بالدائرة
    ring_r = max(ring_r, 40 * SS)
    ins_cx, ins_cy = x + d / 2, y + d / 2

    if plan["show_ring"] and 0 <= ring_cx <= size and 0 <= ring_cy <= size:
        draw = ImageDraw.Draw(canvas)
        vx, vy = ins_cx - ring_cx, ins_cy - ring_cy
        dist = (vx ** 2 + vy ** 2) ** 0.5
        if dist > ring_r + d / 2 + 20 * SS:
            ux, uy = vx / dist, vy / dist
            draw.line(
                (
                    ring_cx + ux * ring_r, ring_cy + uy * ring_r,
                    ins_cx - ux * d / 2, ins_cy - uy * d / 2,
                ),
                fill="white", width=4 * SS,
            )
        draw.ellipse(
            (ring_cx - ring_r, ring_cy - ring_r,
             ring_cx + ring_r, ring_cy + ring_r),
            outline="white", width=5 * SS,
        )

    paste_inset(canvas, detail, x, y, plan["inset_shape"], 8 * SS)

    if keep is not None:
        keep.append(((x, y, x + d, y + d), 5.0))
        keep.append((
            (ring_cx - ring_r, ring_cy - ring_r,
             ring_cx + ring_r, ring_cy + ring_r),
            2.0,
        ))


def build_asis(canvas, images, plan):
    """صورة مركّبة أو لقطة شاشة: تُعرض كاملة دون قص ولا دمج."""
    size = canvas.width
    item = plan["images"][0]
    img = images[item["index"]]
    W, H = img.size

    if 0.8 <= W / H <= 1.25:
        paste_tile(canvas, img, item, (0, 0, size, size))
        return

    small = size // 4
    bg = ImageOps.fit(img, (small, small), Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(8 * SS))
    bg = bg.resize((size, size), Image.Resampling.BICUBIC)
    bg = ImageEnhance.Brightness(bg).enhance(0.5)
    canvas.paste(bg, (0, 0))

    scale = min(size / W, size / H)
    fg = img.resize(
        (max(1, round(W * scale)), max(1, round(H * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas.paste(fg, ((size - fg.width) // 2, (size - fg.height) // 2))


def window_coverage(image, item, w, h):
    """نسبة ما تحفظه نافذة القص من العنصر الرئيسي (1.0 = كامل)."""
    W, H = image.size
    aspect = w / h
    anchor = choose_anchor(image, item, aspect)
    win = smart_window(image, anchor, aspect, "cover")

    sl, st, sr, sb = (
        item["subject_box"][0] * W, item["subject_box"][1] * H,
        item["subject_box"][2] * W, item["subject_box"][3] * H,
    )
    area = (sr - sl) * (sb - st)
    if area <= 0:
        return 1.0
    iw = max(0.0, min(sr, win[2]) - max(sl, win[0]))
    ih = max(0.0, min(sb, win[3]) - max(st, win[1]))
    return iw * ih / area


def two_panel_boxes(size):
    g = GUTTER * SS
    half = (size - g) // 2
    side = [(0, 0, half, size), (half + g, 0, size, size)]
    stacked = [(0, 0, size, half), (0, half + g, size, size)]
    return side, stacked


def panel_score(images, plan, boxes):
    return sum(
        window_coverage(
            images[item["index"]], item,
            box[2] - box[0], box[3] - box[1],
        )
        for item, box in zip(plan["images"], boxes)
    )


def build_panels(canvas, images, plan, keep=None):
    size = canvas.width
    g = GUTTER * SS
    layout = plan["layout"]

    if layout == "two_panel":
        side_boxes, stack_boxes = two_panel_boxes(size)

        wanted_stacked = plan["orientation"] == "stacked"
        chosen = stack_boxes if wanted_stacked else side_boxes
        other = side_boxes if wanted_stacked else stack_boxes

        # نغيّر اتجاه النموذج فقط إذا كان الآخر يحفظ العنصر أفضل بوضوح.
        if (
            panel_score(images, plan, other)
            > panel_score(images, plan, chosen) + 0.05
        ):
            chosen = other
            print("تم تغيير اتجاه اللوحتين لأنه يحفظ العنصر الرئيسي أفضل.")
        boxes = chosen

    elif layout == "three_panel":
        s = (size - g) // 2
        boxes = [
            (0, 0, s, size),                       # مستطيل طولي
            (s + g, 0, size, s),                   # مربع علوي
            (s + g, s + g, size, size),            # مربع سفلي
        ]
        if plan["main_side"] == "right":
            boxes = [(size - x1, y0, size - x0, y1)
                     for x0, y0, x1, y1 in boxes]

    else:  # four_grid
        half = (size - g) // 2
        boxes = [
            (0, 0, half, half),
            (half + g, 0, size, half),
            (0, half + g, half, size),
            (half + g, half + g, size, size),
        ]

    for item, box in zip(plan["images"], boxes):
        paste_tile(canvas, images[item["index"]], item, box, keep)


def find_font_path():
    candidates = [os.getenv("FONT_PATH", "").strip()]
    here = Path(__file__).resolve().parent
    candidates += [
        here / "fonts" / FONT_FILENAME,
        Path("fonts") / FONT_FILENAME,
        *FONT_FALLBACKS,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def clean_headline(text):
    """يزيل الرموز التعبيرية والهاشتاغات وعلامات الاقتباس الزائدة."""
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]", "", text or "")
    text = text.replace("#", " ").replace("«", "").replace("»", "")
    text = text.replace('"', "").replace("“", "").replace("”", "")
    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("C") or ch == " "
    )
    return clean_text(text)


def shape_word(word):
    return get_display(arabic_reshaper.reshape(word), base_dir="R")


def layout_headline(words, font_path, width):
    """يفضّل سطرين بخط كبير، وإن لم يتسع فثلاثة أسطر بخط أصغر."""
    max_w = width * 0.88
    shaped = [shape_word(w) for w in words]
    result = None

    for max_lines, min_frac in ((2, 0.052), (3, 0.044), (4, 0.038)):
        for size in range(int(width * 0.078), int(width * min_frac) - 1, -2):
            font = ImageFont.truetype(
                font_path, size, layout_engine=ImageFont.Layout.BASIC
            )
            space = font.getlength(" ") + size * 0.10
            widths = [font.getlength(s) for s in shaped]

            lines, line_widths = [], []
            cur, cur_w = [], 0.0
            for i, wd in enumerate(widths):
                add = wd if not cur else space + wd
                if cur and cur_w + add > max_w:
                    lines.append(cur)
                    line_widths.append(cur_w)
                    cur, cur_w = [i], wd
                else:
                    cur.append(i)
                    cur_w += add
            if cur:
                lines.append(cur)
                line_widths.append(cur_w)

            result = {
                "font": font, "size": size, "space": space,
                "shaped": shaped, "widths": widths,
                "lines": lines, "line_widths": line_widths,
            }
            if len(lines) <= max_lines and max(line_widths) <= max_w:
                return result

    return result


def zone_penalty(zone, keep):
    zx0, zy0, zx1, zy1 = zone
    total = 0.0
    for (bx0, by0, bx1, by1), weight in keep:
        area = max(1.0, (bx1 - bx0) * (by1 - by0))
        ow = max(0.0, min(zx1, bx1) - max(zx0, bx0))
        oh = max(0.0, min(zy1, by1) - max(zy0, by0))
        total += weight * ow * oh / area
    return total


def add_headline(canvas, headline, highlight, keep):
    """يكتب عبارة عربية على الصورة في الأعلى أو الأسفل حسب الأقل تغطية."""
    if not ARABIC_TEXT_OK:
        print("::warning title=النص العربي::مكتبتا arabic-reshaper و "
              "python-bidi غير مثبتتين؛ تُخطي الكتابة على الصورة.")
        return None

    font_path = find_font_path()
    if not font_path:
        print("::warning title=النص العربي::لم يُعثر على خط عربي؛ "
              "تُخطي الكتابة على الصورة.")
        return None

    words = clean_headline(headline).split()[:24]
    if not words:
        return None

    W, H = canvas.size
    lay = layout_headline(words, font_path, W)
    if lay is None:
        return None

    size = lay["size"]
    line_h = int(size * 1.28)
    pad = int(size * 0.55)
    n_lines = len(lay["lines"])
    block_h = line_h * n_lines + pad * 2

    zones = {
        "bottom": (0, H - block_h, W, H),
        "top": (0, 0, W, block_h),
    }
    zone_name = min(
        ("bottom", "top"), key=lambda z: zone_penalty(zones[z], keep)
    )
    zx0, zy0, zx1, zy1 = zones[zone_name]

    # تدرج داكن خلف النص لضمان الوضوح فوق أي صورة
    gh = int(block_h * 1.5)
    alpha = Image.linear_gradient("L").resize((W, gh))
    alpha = alpha.point(lambda v: int(v * 0.88))
    if zone_name == "bottom":
        canvas.paste((0, 0, 0), (0, H - gh, W, H), alpha)
    else:
        canvas.paste((0, 0, 0), (0, 0, W, gh), ImageOps.flip(alpha))

    draw = ImageDraw.Draw(canvas)

    # شريط لوني قصير على الحافة الداخلية للنص
    bar_w, bar_h = int(W * 0.14), max(6, int(size * 0.12))
    bar_y = (
        zy0 + pad - int(size * 0.30) - bar_h
        if zone_name == "bottom"
        else zy1 - pad + int(size * 0.30)
    )
    draw.rounded_rectangle(
        ((W - bar_w) // 2, bar_y, (W + bar_w) // 2, bar_y + bar_h),
        radius=bar_h // 2, fill=TEXT_ACCENT,
    )

    def norm(word):
        return re.sub(r"[^\w]", "", word, flags=re.UNICODE)

    marked = {norm(w) for w in clean_headline(highlight).split()} - {""}
    stroke = max(3, size // 16)
    y = zy0 + pad

    for line, line_w in zip(lay["lines"], lay["line_widths"]):
        x = (W + line_w) / 2           # الكتابة من اليمين إلى اليسار
        for idx in line:
            x -= lay["widths"][idx]
            color = TEXT_ACCENT if norm(words[idx]) in marked else "white"
            draw.text(
                (x, y), lay["shaped"][idx], font=lay["font"], fill=color,
                stroke_width=stroke, stroke_fill="black",
            )
            x -= lay["space"]
        y += line_h

    return {
        "text": " ".join(words),
        "zone": zone_name,
        "font": Path(font_path).name,
        "font_size": size,
        "lines": n_lines,
    }


def create_design(
    images, plan, destination: Path, headline=None, highlight=None
):
    if not images:
        raise ProjectError("لا توجد صور صالحة للتصميم.")

    source_plan = plan
    plan = dict(plan)
    plan["images"] = [
        i for i in plan.get("images", [])
        if 0 <= i.get("index", -1) < len(images)
    ]
    if not plan["images"]:
        raise ProjectError("خطة التصميم لا تشير إلى صور متاحة.")

    size = IMAGE_SIZE * SS

    # لوحتان تفقدان جزءًا كبيرًا من العنصر المهم: صورة واحدة مع تكبير أفضل.
    if plan["layout"] == "two_panel" and len(plan["images"]) == 2:
        side_boxes, stack_boxes = two_panel_boxes(size)
        best = max(
            panel_score(images, plan, side_boxes),
            panel_score(images, plan, stack_boxes),
        )
        if best < 1.5:
            print(
                f"تحذير: اللوحتان تحفظان {best:.2f} من 2.0 فقط؛ "
                "التحول إلى صورة واحدة مع تفصيل مكبر."
            )
            plan["layout"] = "single_inset"
            plan["images"] = plan["images"][:1]
            source_plan["layout_applied"] = "single_inset"

    canvas = Image.new("RGB", (size, size), GUTTER_COLOR)

    keep = []
    if plan["layout"] == "as_is":
        build_asis(canvas, images, plan)
    elif plan["layout"] == "single_inset" or len(plan["images"]) == 1:
        build_single(canvas, images, plan, keep)
    else:
        build_panels(canvas, images, plan, keep)

    canvas = canvas.resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )

    text_enabled = os.getenv("ADD_IMAGE_TEXT", "").strip().lower() not in {
        "0", "false", "no", "off",
    }
    if headline and text_enabled:
        scaled = [
            (tuple(v / SS for v in box), weight) for box, weight in keep
        ]
        info = add_headline(canvas, headline, highlight or "", scaled)
        if info:
            source_plan["headline"] = info
            print(f"النص على الصورة: {info['text']} ({info['zone']})")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, "JPEG", quality=93, optimize=True)
    return destination


# ---------------------------------------------------------------------------
# الملفات والتشغيل
# ---------------------------------------------------------------------------

def analyze_with_providers(images, title, text):
    """
    يحلل الصور بـ Gemini أولًا ثم Groq احتياطًا.
    VISION_PROVIDER: auto (الافتراضي) أو gemini أو groq.
    يعيد (الخطة، الصور التي حُللت، أخطاء المزودات الفاشلة).
    """
    provider = os.getenv("VISION_PROVIDER", "auto").strip().lower()
    if provider not in ("auto", "gemini", "groq"):
        provider = "auto"

    summary = text[:1200]
    errors = []

    if provider in ("auto", "gemini"):
        if os.getenv("GEMINI_API_KEY", "").strip():
            batch = images[:gemini.MAX_IMAGES]
            try:
                plan = gemini.analyze_images(batch, title, summary)
                return plan, batch, errors
            except (
                gemini.GeminiError, GrokError, ValueError, KeyError,
                TypeError,
            ) as exc:
                errors.append(f"Gemini: {exc}")
                print(f"::warning title=فشل Gemini::{exc}")
        else:
            errors.append("Gemini: المفتاح GEMINI_API_KEY غير موجود.")
            print("تنبيه: GEMINI_API_KEY غير موجود؛ سيُستخدم Groq.")

    if provider in ("auto", "groq"):
        batch = images[:MAX_VISION_IMAGES]
        try:
            plan = analyze_images(batch, title, summary)
            plan["provider"] = "groq"
            return plan, batch, errors
        except (GrokError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"Groq: {exc}")

    raise ProjectError(" | ".join(errors) or "لا يوجد مزود تحليل صور.")


def build_wide_images(images, plan, rewritten, out_dir):
    """
    الصورتان العريضتان 16:9:
    1) article_middle.jpg: دمج صور المقال بلا نص (توضع في منتصف المقال).
    2) thumbnail.jpg: صورة المقال عبر Cloudflare + العنوان، أو الاحتياط.
    """
    info = {"middle": None, "thumbnail": None, "errors": []}
    if not images or not plan:
        info["errors"].append("لا توجد صور لصنع الصورتين العريضتين.")
        return info

    collage = None
    try:
        gallery = plan.get("gallery") or default_gallery(len(images))
        collage = wide_collage.create_gallery_collage(
            images, gallery, out_dir / "article_middle.jpg"
        )
        info["middle"] = {
            "file": "article_middle.jpg",
            "layout": collage["layout"],
            "count": collage["count"],
            "images": collage["images"],
        }
        if collage["count"] == 1:
            info["errors"].append(
                "لم تتوفر إلا صورة صالحة واحدة لمعرض المقال."
            )
    except (wide_collage.CollageError, OSError, ValueError, KeyError) as exc:
        info["errors"].append(f"الصورة الأولى: {exc}")
        print(f"::warning title=فشل الصورة العريضة الأولى::{exc}")

    sensitive = bool((rewritten.get("review") or {}).get("sensitive"))
    try:
        info["thumbnail"] = thumbnail.create_thumbnail(
            images=images,
            plan=plan,
            headline=rewritten["image_title"],
            highlight=rewritten.get("image_highlight", ""),
            collage=collage,
            destination=out_dir / "thumbnail.jpg",
            prompt_file=out_dir / "thumbnail_prompt.txt",
            sensitive=sensitive,
        )
    except (thumbnail.ThumbnailError, OSError, ValueError, KeyError) as exc:
        info["errors"].append(f"الصورة الثانية: {exc}")
        print(f"::warning title=فشل صورة المقال::{exc}")

    return info


def write_text_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def publish_output_dir(out_dir: Path) -> None:
    """يكتب مسار مجلد هذا المقال فقط ليرفعه الـ workflow دون بقية المشاريع."""
    target = os.getenv("GITHUB_OUTPUT")
    if not target:
        return
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(f"out_dir={out_dir.as_posix()}\n")


def output_folder_for(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return OUTPUT_ROOT / digest


def main():
    article_url = os.getenv("ARTICLE_URL", "").strip()
    force = os.getenv("FORCE_REPROCESS", "").strip().lower() in {
        "1", "true", "yes",
    }

    if not article_url:
        raise ProjectError("لم يتم إدخال ARTICLE_URL.")

    validate_public_url(article_url)
    out_dir = output_folder_for(article_url)

    if (out_dir / "result.json").exists() and not force:
        print(
            f"هذا الرابط عولج سابقًا: {out_dir}. "
            "فعّل force_reprocess لإعادة المعالجة."
        )
        publish_output_dir(out_dir)
        return

    print("1/6: استخراج المقال والصور...")
    article = extract_article(article_url)

    print("العنوان:", article["title"])
    print("طول النص:", len(article["text"]))
    print("روابط الصور المكتشفة:", len(article["image_urls"]))

    print("2/6: كتابة المقال والمنشور وحزمة SEO...")
    try:
        rewritten = writer.write_article(
            article_title=article["title"],
            article_text=article["text"],
            source_url=article["source_url"],
        )
    except writer.WriterError as exc:
        raise ProjectError(f"فشلت كتابة المقال: {exc}") from exc
    print(
        f"الكاتب: {rewritten['writer']['provider']} "
        f"({rewritten['writer']['model']})، "
        f"الكلمات: {rewritten['stats']['words']}"
    )

    print("3/6: تنزيل الصور والتحقق منها...")
    selected = download_article_images(article["image_urls"])
    images = [image for _, image in selected]
    used_urls = [url for url, _ in selected]
    print("الصور الصالحة:", len(images))

    image_file = None
    image_note = ""
    plan = None
    wide_plan, wide_images = None, []

    if images:
        print("4/6: تحليل الصور بالذكاء الاصطناعي (Gemini ثم Groq)...")
        try:
            plan, design_images, provider_errors = analyze_with_providers(
                images, article["title"], article["text"]
            )
            wide_plan = {
                k: v for k, v in plan.items()
                if k not in ("layout_applied", "headline")
            }
            wide_images = design_images
            create_design(
                design_images, plan, out_dir / "facebook_image.jpg",
                headline=rewritten["image_title"],
                highlight=rewritten.get("image_highlight"),
            )
            image_file = "facebook_image.jpg"
            if provider_errors:
                image_note = (
                    "نجح التحليل عبر "
                    f"{plan.get('provider')} بعد فشل: "
                    + " | ".join(provider_errors)
                )

        except (GrokError, ProjectError, OSError, ValueError) as exc:
            image_note = (
                "تعذر إكمال التحليل البصري أو تطبيق التصميم: "
                f"{exc}"
            )
            print("تحذير:", image_note)
            print(f"::warning title=فشل تحليل الصور::{image_note}")

            # احتياط محلي واضح: صورة واحدة فقط دون ادعاء نجاح تحليل AI.
            wide_images = images[:4]
            wide_plan = fallback_plan(len(wide_images))
            try:
                plan = fallback_plan(1)
                create_design(
                    images[:1], plan, out_dir / "facebook_image.jpg",
                    headline=rewritten["image_title"],
                    highlight=rewritten.get("image_highlight"),
                )
                image_file = "facebook_image.jpg"
                image_note += (
                    " استُخدم تصميم احتياطي محلي دون تحليل بصري ناجح."
                )
            except (ProjectError, OSError, ValueError) as fallback_exc:
                image_note += f" وفشل التصميم الاحتياطي: {fallback_exc}"
    else:
        image_note = "لم يتم العثور على صور صالحة. لم تُنشأ صورة بديلة."
        print("تحذير:", image_note)

    print("4b/6: الصورتان العريضتان 16:9...")
    wide_info = build_wide_images(
        wide_images, wide_plan, rewritten, out_dir
    )

    print("5/6: حفظ المنشور والمقال...")
    write_text_file(
        out_dir / "facebook_post.txt", writer.build_post_text(rewritten)
    )
    write_text_file(
        out_dir / "rewritten_article.md",
        writer.build_markdown(
            rewritten,
            middle_image="article_middle.jpg" if wide_info["middle"] else None,
        ),
    )
    write_text_file(out_dir / "seo.txt", writer.build_seo_text(rewritten))
    write_text_file(
        out_dir / "editor_notes.txt",
        writer.build_notes(
            rewritten, article["source_url"], article["title"]
        ),
    )

    print("6/6: حفظ بيانات النتيجة...")
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": article["source_url"],
        "original_title": article["title"],
        "title": rewritten["title"],
        "facebook_post": rewritten["facebook_post"],
        "hashtags": rewritten["hashtags"],
        "rewritten_article": rewritten["rewritten_article"],
        "alt_titles": rewritten["alt_titles"],
        "seo": rewritten["seo"],
        "writer": rewritten["writer"],
        "article_stats": rewritten["stats"],
        "article_audit": rewritten["audit"],
        "article_review": rewritten["review"],
        "image_file": image_file,
        "wide_images": wide_info,
        "image_title": rewritten["image_title"],
        "image_text": rewritten["image_title"],
        "image_highlight": rewritten.get("image_highlight"),
        "image_count": len(images),
        "image_source_urls": used_urls,
        "image_analysis_plan": plan,
        "image_note": image_note,
        "rights_note": (
            "لا يعني ظهور الصورة في المقال أن إعادة نشرها مسموحة. "
            "تحقق من الترخيص أو الإذن قبل النشر."
        ),
        "verification_note": (
            "أعيدت صياغة المقال اعتمادًا على النص المستخرج فقط؛ "
            "لم يتم التحقق من الوقائع بشكل مستقل. المراجعة الآلية "
            "في article_review قد تخطئ، فراجع النص قبل النشر."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    publish_output_dir(out_dir)

    print("\nاكتمل إنشاء المحتوى:", out_dir)
    for name in (
        "facebook_image.jpg",
        "article_middle.jpg",
        "thumbnail.jpg",
        "thumbnail_prompt.txt",
        "facebook_post.txt",
        "rewritten_article.md",
        "seo.txt",
        "editor_notes.txt",
        "result.json",
    ):
        if (out_dir / name).exists():
            print("-", out_dir / name)


if __name__ == "__main__":
    try:
        main()
    except (ProjectError, GrokError) as exc:
        print(f"\nخطأ: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nتم إيقاف التشغيل.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(
            f"\nخطأ غير متوقع: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
