
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
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from grok import GrokError, analyze_images, rewrite_article


OUTPUT_ROOT = Path("output")
IMAGE_SIZE = 1080
GUTTER = 12
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


def crop_region(
    image: Image.Image,
    crop: list[float],
) -> Image.Image:
    left, top, right, bottom = crop
    width, height = image.size

    box = (
        max(0, min(width - 1, round(left * width))),
        max(0, min(height - 1, round(top * height))),
        max(1, min(width, round(right * width))),
        max(1, min(height, round(bottom * height))),
    )

    if box[2] <= box[0] or box[3] <= box[1]:
        return image.copy()

    return image.crop(box)


def fit_tile(
    image: Image.Image,
    size: tuple[int, int],
    crop: list[float] | None = None,
) -> Image.Image:
    source = crop_region(image, crop or [0, 0, 1, 1])
    return ImageOps.fit(
        source,
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.4),
    )


def create_design(
    images: list[Image.Image],
    plan: dict,
    destination: Path,
):
    if not images:
        raise ProjectError("لا توجد صور صالحة للتصميم.")

    # التحليل يرسل ثلاث صور كحد أقصى، لذا نستخدم القائمة نفسها.
    images = images[:3]
    count = len(images)
    order = plan.get("image_order", list(range(count)))
    order = [
        i for i in order
        if isinstance(i, int) and 0 <= i < count
    ]
    order += [i for i in range(count) if i not in order]

    crops = plan.get("crops", [])
    layout = plan.get("layout", "single_inset")
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "#111111")

    if count == 1 or layout == "single_inset":
        main_index = order[0]
        main_crop = crops[main_index] if main_index < len(crops) else None
        canvas = fit_tile(
            images[main_index],
            (IMAGE_SIZE, IMAGE_SIZE),
            main_crop,
        )

        detail_index = plan.get("detail_image_index", main_index)
        if not isinstance(detail_index, int) or not 0 <= detail_index < count:
            detail_index = main_index

        detail_crop = plan.get("detail_crop", [0.2, 0.2, 0.8, 0.8])
        detail = fit_tile(
            images[detail_index],
            (300, 300),
            detail_crop,
        )

        position = plan.get("inset_position", "top_right")
        margin = 34
        positions = {
            "top_left": (margin, margin),
            "top_right": (IMAGE_SIZE - 300 - margin, margin),
            "bottom_left": (margin, IMAGE_SIZE - 300 - margin),
            "bottom_right": (
                IMAGE_SIZE - 300 - margin,
                IMAGE_SIZE - 300 - margin,
            ),
        }
        x, y = positions.get(position, positions["top_right"])

        if plan.get("inset_shape") == "circle":
            mask = Image.new("L", (300, 300), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, 299, 299), fill=255)
            canvas.paste(detail, (x, y), mask)
            draw = ImageDraw.Draw(canvas)
            draw.ellipse(
                (x, y, x + 299, y + 299),
                outline="white",
                width=8,
            )
        else:
            canvas.paste(detail, (x, y))
            ImageDraw.Draw(canvas).rectangle(
                (x, y, x + 299, y + 299),
                outline="white",
                width=8,
            )

    elif count == 2 or layout == "two_panel":
        half = (IMAGE_SIZE - GUTTER) // 2
        for slot, index in enumerate(order[:2]):
            box = (
                (0, 0, IMAGE_SIZE, half)
                if slot == 0
                else (0, half + GUTTER, IMAGE_SIZE, IMAGE_SIZE)
            )
            crop = crops[index] if index < len(crops) else None
            tile = fit_tile(
                images[index],
                (box[2] - box[0], box[3] - box[1]),
                crop,
            )
            canvas.paste(tile, (box[0], box[1]))

    elif count == 3 and layout == "three_panel":
        left_width = 660
        right_width = IMAGE_SIZE - left_width - GUTTER
        half_height = (IMAGE_SIZE - GUTTER) // 2

        boxes = [
            (0, 0, left_width, IMAGE_SIZE),
            (left_width + GUTTER, 0, IMAGE_SIZE, half_height),
            (
                left_width + GUTTER,
                half_height + GUTTER,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
        ]

        for slot, index in enumerate(order[:3]):
            box = boxes[slot]
            crop = crops[index] if index < len(crops) else None
            tile = fit_tile(
                images[index],
                (box[2] - box[0], box[3] - box[1]),
                crop,
            )
            canvas.paste(tile, (box[0], box[1]))

    else:
        # شبكة آمنة عند وجود عدد صور أو تخطيط غير متوقع.
        half = (IMAGE_SIZE - GUTTER) // 2
        boxes = [
            (0, 0, half, half),
            (half + GUTTER, 0, IMAGE_SIZE, half),
            (0, half + GUTTER, half, IMAGE_SIZE),
            (half + GUTTER, half + GUTTER, IMAGE_SIZE, IMAGE_SIZE),
        ]
        for slot, index in enumerate(order[:4]):
            box = boxes[slot]
            crop = crops[index] if index < len(crops) else None
            tile = fit_tile(
                images[index],
                (box[2] - box[0], box[3] - box[1]),
                crop,
            )
            canvas.paste(tile, (box[0], box[1]))

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(
        destination,
        "JPEG",
        quality=92,
        optimize=True,
    )
    return destination


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
            plan = analyze_images(images[:3])
            # يجب استخدام الصور الثلاث نفسها التي حللها النموذج.
            design_images = images[:3]
            create_design(
                design_images,
                plan,
                out_dir / "facebook_image.jpg",
            )
            image_file = "facebook_image.jpg"

        except (GrokError, ProjectError, OSError, ValueError) as exc:
            image_note = (
                "تعذر إكمال التحليل البصري أو تطبيق التصميم: "
                f"{exc}"
            )
            print("تحذير:", image_note)

            # احتياط محلي واضح: صورة واحدة فقط دون ادعاء نجاح تحليل AI.
            try:
                fallback_plan = {
                    "layout": "single_inset",
                    "image_order": [0],
                    "crops": [[0.0, 0.0, 1.0, 1.0]],
                    "detail_image_index": 0,
                    "detail_crop": [0.2, 0.2, 0.8, 0.8],
                    "inset_shape": "circle",
                    "inset_position": "top_right",
                }
                create_design(
                    images[:1],
                    fallback_plan,
                    out_dir / "facebook_image.jpg",
                )
                image_file = "facebook_image.jpg"
                image_note += (
                    " استُخدم تصميم احتياطي محلي دون تحليل بصري ناجح."
                )
            except (ProjectError, OSError, ValueError) as fallback_exc:
                image_note += f" وفشل التصميم الاحتياطي: {fallback_exc}"
    else:
        image_note = (
            "لم يتم العثور على صور صالحة. لم تُنشأ صورة بديلة."
        )
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
