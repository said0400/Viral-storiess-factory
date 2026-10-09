
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
from PIL import Image, ImageOps, UnidentifiedImageError

from grok import GrokError, rewrite_article


OUTPUT_ROOT = Path("output")

IMAGE_SIZE = 1080
IMAGE_GUTTER = 12

MAX_PAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 4
MAX_IMAGE_ATTEMPTS = 20
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
SESSION.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "ar,en;q=0.8",
    }
)


class ProjectError(RuntimeError):
    """خطأ متوقع برسالة واضحة للمستخدم."""


def validate_public_url(url: str) -> str:
    """قبول HTTP(S) فقط ورفض أسماء المضيفين وعناوين IP غير العامة."""

    if not isinstance(url, str) or not url.strip():
        raise ProjectError("رابط المقال فارغ.")

    url = url.strip()

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ProjectError("الرابط غير صالح.") from exc

    if parsed.scheme.lower() not in ("http", "https"):
        raise ProjectError("الرابط يجب أن يبدأ بـ http:// أو https://.")

    if not hostname:
        raise ProjectError("الرابط لا يحتوي على اسم موقع صالح.")

    if parsed.username or parsed.password:
        raise ProjectError("الروابط التي تحتوي على بيانات دخول مرفوضة.")

    hostname = hostname.rstrip(".").lower()

    if hostname in {
        "localhost",
        "localhost.localdomain",
    } or hostname.endswith(".localhost"):
        raise ProjectError("لا يمكن استخدام عنوان محلي.")

    port = port or (443 if parsed.scheme.lower() == "https" else 80)

    try:
        # عناوين IP المكتوبة مباشرة لا تحتاج إلى DNS.
        try:
            literal_ip = ipaddress.ip_address(hostname)
            addresses = [literal_ip]
        except ValueError:
            answers = socket.getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
            addresses = [
                ipaddress.ip_address(answer[4][0].split("%")[0])
                for answer in answers
            ]
    except (OSError, ValueError) as exc:
        raise ProjectError(
            f"تعذر التحقق من عنوان الموقع: {hostname}"
        ) from exc

    if not addresses:
        raise ProjectError("لم يتم العثور على عنوان IP للموقع.")

    if any(not address.is_global for address in addresses):
        raise ProjectError(
            "تم رفض الرابط لأنه يشير إلى عنوان IP غير عام."
        )

    return url


def safe_get(
    url: str,
    max_bytes: int,
    expected_image: bool = False,
) -> tuple[str, str, bytes]:
    """تنزيل محدود الحجم مع التحقق من وجهات التحويل."""

    current_url = url

    for _ in range(6):
        current_url = validate_public_url(current_url)

        try:
            response = SESSION.get(
                current_url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as exc:
            raise ProjectError(
                f"تعذر الاتصال بالموقع: {exc}"
            ) from exc

        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            response.close()

            if not location:
                raise ProjectError("تحويل الرابط بلا وجهة.")

            current_url = urljoin(current_url, location)
            continue

        if not 200 <= response.status_code < 300:
            status = response.status_code
            response.close()

            hint = (
                " قد يمنع الموقع الوصول الآلي."
                if status in (401, 403, 429)
                else ""
            )

            raise ProjectError(
                f"أعاد الموقع HTTP {status}.{hint}"
            )

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if expected_image and not content_type.startswith("image/"):
            response.close()
            raise ProjectError("الرابط لا يعيد نوع ملف صورة.")

        declared_size = response.headers.get("Content-Length")

        if declared_size:
            try:
                if int(declared_size) > max_bytes:
                    response.close()
                    raise ProjectError("حجم الملف يتجاوز الحد المسموح.")
            except ValueError:
                pass

        data = bytearray()

        try:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue

                data.extend(chunk)

                if len(data) > max_bytes:
                    raise ProjectError(
                        "حجم الملف تجاوز الحد المسموح أثناء التنزيل."
                    )
        except requests.RequestException as exc:
            raise ProjectError(
                f"انقطع تنزيل الملف: {exc}"
            ) from exc
        finally:
            response.close()

        return current_url, content_type, bytes(data)

    raise ProjectError("تجاوز الرابط عدد التحويلات المسموح به.")


def clean_text(value: str | None) -> str:
    if not value:
        return ""

    return re.sub(r"\s+", " ", value).strip()


def decode_html(html_bytes: bytes, content_type: str) -> str:
    match = re.search(r"charset=([\w\-]+)", content_type or "")
    declared = match.group(1) if match else None

    encodings = [declared] if declared else []

    dammit = UnicodeDammit(
        html_bytes,
        known_definite_encodings=encodings,
        is_html=True,
    )

    if dammit.unicode_markup:
        return dammit.unicode_markup

    return html_bytes.decode("utf-8", errors="replace")


def extract_article(article_url: str) -> dict:
    final_url, content_type, html_bytes = safe_get(
        article_url,
        max_bytes=MAX_PAGE_BYTES,
    )

    if "html" not in content_type and "xhtml" not in content_type:
        raise ProjectError(
            "الرابط لا يشير إلى صفحة HTML. أدخل رابط المقال نفسه."
        )

    html = decode_html(html_bytes, content_type)
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

    if not title and soup.find("h1"):
        title = clean_text(soup.find("h1").get_text(" ", strip=True))

    if not title and soup.title:
        title = clean_text(soup.title.get_text(" ", strip=True))

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
            "تعذر استخراج نص كافٍ. قد يكون الموقع محميًا أو يحتاج "
            "إلى JavaScript أو تسجيل الدخول."
        )

    image_urls: list[str] = []

    def add_image(candidate: str | None) -> None:
        if not isinstance(candidate, str):
            return

        candidate = candidate.strip()

        if not candidate or candidate.startswith("data:"):
            return

        absolute = urldefrag(urljoin(final_url, candidate))[0]

        parsed_image = urlparse(absolute)

        if parsed_image.scheme not in ("http", "https"):
            return

        lowered = absolute.lower()

        if any(hint in lowered for hint in JUNK_IMAGE_HINTS):
            return

        if absolute not in image_urls:
            image_urls.append(absolute)

    # الصور الرئيسية المعلنة، ثم الصور الموجودة داخل المقال.
    for selector in (
        'meta[property="og:image"]',
        'meta[name="twitter:image"]',
    ):
        node = soup.select_one(selector)

        if node:
            add_image(node.get("content"))

    nodes = []

    for selector in ("article img", "main img", '[role="main"] img'):
        nodes.extend(soup.select(selector))

    if not nodes:
        nodes = soup.select("img")

    seen_nodes = set()

    for node in nodes:
        identity = id(node)

        if identity in seen_nodes:
            continue

        seen_nodes.add(identity)

        for attribute in (
            "src",
            "data-src",
            "data-lazy-src",
            "data-original",
        ):
            add_image(node.get(attribute))

        srcset = node.get("srcset") or node.get("data-srcset")

        if srcset:
            entries = [
                item.strip().split()[0]
                for item in srcset.split(",")
                if item.strip()
            ]

            # الأفضلية لأكبر عرض معلن، مع دعم srcset القياسي.
            def srcset_width(entry: str) -> int:
                match = re.search(r"\s+(\d+)w$", entry)
                return int(match.group(1)) if match else 0

            entries.sort(key=srcset_width, reverse=True)

            for entry in entries:
                add_image(entry)

    return {
        "source_url": final_url,
        "title": title,
        "text": article_text,
        "image_urls": image_urls[:40],
    }


def average_hash(image: Image.Image) -> int:
    small = image.convert("L").resize(
        (8, 8),
        Image.Resampling.LANCZOS,
    )

    pixels = list(small.getdata())
    mean = sum(pixels) / len(pixels)
    result = 0

    for pixel in pixels:
        result = (result << 1) | int(pixel >= mean)

    return result


def is_duplicate(value: int, known: list[int]) -> bool:
    return any(
        (value ^ other).bit_count() <= 5
        for other in known
    )


def download_article_images(
    image_urls: list[str],
) -> list[tuple[str, Image.Image]]:
    collected: list[tuple[str, Image.Image]] = []
    hashes: list[int] = []

    for image_url in image_urls[:MAX_IMAGE_ATTEMPTS]:
        if len(collected) >= MAX_IMAGES:
            break

        try:
            _, _, data = safe_get(
                image_url,
                max_bytes=MAX_IMAGE_BYTES,
                expected_image=True,
            )

            with Image.open(BytesIO(data)) as source:
                if getattr(source, "is_animated", False):
                    source.seek(0)

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

        except (
            ProjectError,
            requests.RequestException,
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
        ) as exc:
            print(f"تحذير: تخطي صورة غير صالحة: {exc}")
            continue

    if not collected:
        return []

    # نحتفظ بترتيب الصور المصدرية حتى لا تتغير هوية الصورة الرئيسية.
    return collected[:MAX_IMAGES]


def crop_to_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> Image.Image:
    x1, y1, x2, y2 = box

    return ImageOps.fit(
        image,
        (x2 - x1, y2 - y1),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.4),
    )


def create_collage(
    images: list[Image.Image],
    destination: Path,
) -> Path:
    """إنشاء تصميم مربع من صورة إلى أربع صور، بلا كتابة أو شعارات."""

    if not images:
        raise ProjectError(
            "لم يتم العثور على صور صالحة في المقال."
        )

    images = images[:MAX_IMAGES]

    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (18, 18, 18),
    )

    gap = IMAGE_GUTTER
    half = (IMAGE_SIZE - gap) // 2

    if len(images) == 1:
        boxes = [(0, 0, IMAGE_SIZE, IMAGE_SIZE)]

    elif len(images) == 2:
        boxes = [
            (0, 0, half, IMAGE_SIZE),
            (half + gap, 0, IMAGE_SIZE, IMAGE_SIZE),
        ]

    elif len(images) == 3:
        boxes = [
            (0, 0, half, IMAGE_SIZE),
            (half + gap, 0, IMAGE_SIZE, half),
            (half + gap, half + gap, IMAGE_SIZE, IMAGE_SIZE),
        ]

    else:
        boxes = [
            (0, 0, half, half),
            (half + gap, 0, IMAGE_SIZE, half),
            (0, half + gap, half, IMAGE_SIZE),
            (half + gap, half + gap, IMAGE_SIZE, IMAGE_SIZE),
        ]

    for image, box in zip(images, boxes):
        tile = crop_to_box(image, box)
        canvas.paste(tile, (box[0], box[1]))

    destination.parent.mkdir(parents=True, exist_ok=True)

    canvas.save(
        destination,
        format="JPEG",
        quality=92,
        optimize=True,
        progressive=True,
    )

    return destination


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        content.rstrip() + "\n",
        encoding="utf-8",
    )


def output_folder_for(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return OUTPUT_ROOT / digest


def main() -> None:
    article_url = os.getenv("ARTICLE_URL", "").strip()

    force = os.getenv("FORCE_REPROCESS", "").strip().lower() in {
        "1", "true", "yes",
    }

    if not article_url:
        raise ProjectError("لم يتم إدخال رابط المقال.")

    # توحيد الرابط قبل احتساب بصمة المجلد.
    article_url = validate_public_url(article_url)
    out_dir = output_folder_for(article_url)

    if (out_dir / "result.json").exists() and not force:
        print(
            f"تمت معالجة هذا الرابط سابقًا: {out_dir}\n"
            "فعّل force_reprocess لإعادة المعالجة."
        )
        return

    print("1/5: استخراج نص المقال والصور...")
    article = extract_article(article_url)

    print(f"العنوان الأصلي: {article['title']}")
    print(f"طول النص: {len(article['text'])} حرف")
    print(f"روابط الصور المكتشفة: {len(article['image_urls'])}")

    print("2/5: إعادة صياغة المقال عبر GroqCloud...")
    rewritten = rewrite_article(
        article_title=article["title"],
        article_text=article["text"],
        source_url=article["source_url"],
    )

    print("3/5: تنزيل الصور والتحقق منها...")
    selected = download_article_images(article["image_urls"])
    images = [image for _, image in selected]
    used_urls = [url for url, _ in selected]

    print(f"عدد الصور الصالحة: {len(images)}")

    print("4/5: إنشاء صورة فيسبوك مربعة...")
    image_file = None
    image_note = ""

    try:
        create_collage(images, out_dir / "facebook_image.jpg")
        image_file = "facebook_image.jpg"

    except ProjectError as exc:
        image_note = str(exc)
        print(f"تحذير: {image_note}")

    print("5/5: حفظ المقال والمنشور والبيانات...")
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
        f"**المصدر الأصلي:** {article['source_url']}\n\n"
        f"---\n\n"
        f"{rewritten['rewritten_article']}\n"
    )

    write_text_file(
        out_dir / "rewritten_article.md",
        markdown,
    )

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
        "image_note": image_note,
        "rights_notice": (
            "تحقق من ترخيص كل صورة وحق إعادة نشرها قبل استخدامها. "
            "وجود الصورة على الموقع الأصلي لا يمنح تلقائيًا حق استخدامها."
        ),
        "verification_notice": (
            "تمت إعادة الصياغة من النص المستخرج، ولم يحدث تحقق مستقل "
            "من الوقائع."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nاكتمل التشغيل: {out_dir}")

    for filename in (
        "facebook_image.jpg",
        "facebook_post.txt",
        "rewritten_article.md",
        "result.json",
    ):
        path = out_dir / filename

        if path.exists():
            print(f"- {path}")


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
