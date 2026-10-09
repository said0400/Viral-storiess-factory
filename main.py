from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
import trafilatura
from bs4 import BeautifulSoup, UnicodeDammit
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
    ImageStat,
    UnidentifiedImageError,
)

from grok import (
    MAX_VISION_IMAGES,
    GrokError,
    analyze_images,
    fallback_plan,
    rewrite_article,
)


OUTPUT_ROOT = Path("output")
IMAGE_SIZE = 1080
SS = 2                      # دقة مضاعفة لنعومة الحواف
GUTTER = 10
GUTTER_COLOR = "#FFFFFF"
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


def paste_tile(canvas, image, item, box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    win = smart_window(image, item["subject_box"], w / h, "cover")
    canvas.paste(polish(crop_window(image, win, (w, h))), (x0, y0))


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


def build_single(canvas, images, plan):
    size = canvas.width
    item = plan["images"][0]
    img = images[item["index"]]
    W, H = img.size

    # 1) الخلفية: مربع كامل متمركز على العنصر
    main_win = smart_window(img, item["subject_box"], 1.0, "cover")
    mx0, my0, mx1, _ = main_win
    mw = mx1 - mx0
    canvas.paste(
        polish(crop_window(img, main_win, (size, size))), (0, 0)
    )

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
            + 4.0 * overlap_frac(
                px, py,
                ring_cx - ring_r, ring_cy - ring_r,
                ring_cx + ring_r, ring_cy + ring_r,
            )
            + 1.0 * edge_density(px, py)
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


def build_panels(canvas, images, plan):
    size = canvas.width
    g = GUTTER * SS
    layout = plan["layout"]

    if layout == "two_panel":
        half = (size - g) // 2
        if plan["orientation"] == "stacked":
            boxes = [(0, 0, size, half), (0, half + g, size, size)]
        else:
            boxes = [(0, 0, half, size), (half + g, 0, size, size)]

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
        paste_tile(canvas, images[item["index"]], item, box)


def create_design(images, plan, destination: Path):
    if not images:
        raise ProjectError("لا توجد صور صالحة للتصميم.")

    plan = dict(plan)
    plan["images"] = [
        i for i in plan.get("images", [])
        if 0 <= i.get("index", -1) < len(images)
    ]
    if not plan["images"]:
        raise ProjectError("خطة التصميم لا تشير إلى صور متاحة.")

    size = IMAGE_SIZE * SS
    canvas = Image.new("RGB", (size, size), GUTTER_COLOR)

    if plan["layout"] == "as_is":
        build_asis(canvas, images, plan)
    elif plan["layout"] == "single_inset" or len(plan["images"]) == 1:
        build_single(canvas, images, plan)
    else:
        build_panels(canvas, images, plan)

    canvas = canvas.resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, "JPEG", quality=93, optimize=True)
    return destination


# ---------------------------------------------------------------------------
# الملفات والتشغيل
# ---------------------------------------------------------------------------

def write_text_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


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
        return

    print("1/6: استخراج المقال والصور...")
    article = extract_article(article_url)

    print("العنوان:", article["title"])
    print("طول النص:", len(article["text"]))
    print("روابط الصور المكتشفة:", len(article["image_urls"]))

    print("2/6: إعادة كتابة المقال والمنشور...")
    rewritten = rewrite_article(
        article_title=article["title"],
        article_text=article["text"],
        source_url=article["source_url"],
    )

    print("3/6: تنزيل الصور والتحقق منها...")
    selected = download_article_images(article["image_urls"])
    images = [image for _, image in selected]
    used_urls = [url for url, _ in selected]
    print("الصور الصالحة:", len(images))

    image_file = None
    image_note = ""
    plan = None

    if images:
        print("4/6: تحليل الصور باستخدام Groq Vision...")
        try:
            design_images = images[:MAX_VISION_IMAGES]
            plan = analyze_images(
                design_images,
                article["title"],
                article["text"][:1200],
            )
            create_design(design_images, plan, out_dir / "facebook_image.jpg")
            image_file = "facebook_image.jpg"

        except (GrokError, ProjectError, OSError, ValueError) as exc:
            image_note = (
                "تعذر إكمال التحليل البصري أو تطبيق التصميم: "
                f"{exc}"
            )
            print("تحذير:", image_note)

            # احتياط محلي واضح: صورة واحدة فقط دون ادعاء نجاح تحليل AI.
            try:
                plan = fallback_plan(1)
                create_design(
                    images[:1], plan, out_dir / "facebook_image.jpg"
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

    print("5/6: حفظ المنشور والمقال...")
    hashtags_text = " ".join(rewritten["hashtags"])
    post_parts = [
        rewritten["title"],
        rewritten["facebook_post"],
    ]
    if hashtags_text:
        post_parts.append(hashtags_text)
    post_parts.append(f"المصدر: {article['source_url']}")

    write_text_file(
        out_dir / "facebook_post.txt",
        "\n\n".join(post_parts),
    )

    markdown = (
        f"# {rewritten['title']}\n\n"
        f"**رابط المقال الأصلي:** {article['source_url']}\n\n"
        "---\n\n"
        f"{rewritten['rewritten_article']}\n"
    )
    write_text_file(out_dir / "rewritten_article.md", markdown)

    print("6/6: حفظ بيانات النتيجة...")
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": article["source_url"],
        "original_title": article["title"],
        "title": rewritten["title"],
        "facebook_post": rewritten["facebook_post"],
        "hashtags": rewritten["hashtags"],
        "rewritten_article": rewritten["rewritten_article"],
        "image_file": image_file,
        "image_count": len(images),
        "image_source_urls": used_urls,
        "image_analysis_plan": plan,
        "image_note": image_note,
        "rights_note": (
            "لا يعني ظهور الصورة في المقال أن إعادة نشرها مسموحة. "
            "تحقق من الترخيص أو الإذن قبل النشر."
        ),
        "verification_note": (
            "أعيدت صياغة المقال اعتمادًا على النص المستخرج؛ "
            "لم يتم التحقق من الوقائع بشكل مستقل."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nاكتمل إنشاء المحتوى:", out_dir)
    for name in (
        "facebook_image.jpg",
        "facebook_post.txt",
        "rewritten_article.md",
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
