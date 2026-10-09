
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
MAX_IMAGE_ATTEMPTS = 14

MIN_IMAGE_WIDTH = 400
MIN_IMAGE_HEIGHT = 300

REQUEST_TIMEOUT = 30
MAX_REDIRECTS = 5

JUNK_IMAGE_HINTS = (
    "logo",
    "icon",
    "avatar",
    "sprite",
    "pixel",
    "tracking",
    "spacer",
    "placeholder",
    "banner-ad",
    "/ads/",
    "advert",
    "emoji",
    "gravatar",
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


class ProjectError(Exception):
    """خطأ متوقع يمكن عرضه للمستخدم بصورة واضحة."""


def validate_public_url(url: str) -> str:
    """التحقق من أن الرابط HTTP(S) ويشير إلى عنوان IP عام."""
    if not isinstance(url, str) or not url.strip():
        raise ProjectError("رابط المقال فارغ.")

    url = url.strip()

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ProjectError("الرابط أو رقم المنفذ غير صالح.") from exc

    if parsed.scheme.lower() not in ("http", "https"):
        raise ProjectError("الرابط يجب أن يبدأ بـ http:// أو https://.")

    if not hostname:
        raise ProjectError("الرابط لا يحتوي على اسم موقع صالح.")

    if parsed.username or parsed.password:
        raise ProjectError("الروابط التي تحتوي على بيانات دخول غير مسموحة.")

    hostname = hostname.rstrip(".").lower()

    if hostname in {"localhost", "localhost.localdomain"}:
        raise ProjectError("لا يمكن استخدام عنوان محلي.")

    effective_port = port or (443 if parsed.scheme.lower() == "https" else 80)

    try:
        addresses = socket.getaddrinfo(
            hostname,
            effective_port,
            type=socket.SOCK_STREAM,
        )
    except (socket.gaierror, OSError) as exc:
        raise ProjectError(
            f"تعذر العثور على خادم الرابط: {hostname}"
        ) from exc

    if not addresses:
        raise ProjectError("لم يتم العثور على عنوان IP للموقع.")

    for address in addresses:
        raw_ip = address[4][0].split("%")[0]

        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise ProjectError("عنوان IP غير صالح للموقع.") from exc

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
    """تنزيل محتوى مع فحص الرابط وكل وجهة تحويل وحدود الحجم."""
    current_url = url

    for redirect_number in range(MAX_REDIRECTS + 1):
        current_url = validate_public_url(current_url)

        response = None

        try:
            response = SESSION.get(
                current_url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
                stream=True,
            )

            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")

                if not location:
                    raise ProjectError(
                        "تحويل الرابط لا يحتوي على وجهة."
                    )

                if redirect_number >= MAX_REDIRECTS:
                    raise ProjectError(
                        "تجاوز الرابط عدد التحويلات المسموح به."
                    )

                current_url = urljoin(current_url, location)
                continue

            if not 200 <= response.status_code < 300:
                status = response.status_code
                hint = ""

                if status in (401, 403, 429):
                    hint = " قد يمنع الموقع الوصول الآلي."

                raise ProjectError(
                    f"أعاد الموقع رمز HTTP {status}.{hint}"
                )

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if expected_image and not content_type.startswith("image/"):
                raise ProjectError("الرابط لا يعيد ملف صورة.")

            content_length = response.headers.get("Content-Length")

            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise ProjectError(
                            "حجم الملف أكبر من الحد المسموح."
                        )
                except ValueError:
                    pass

            data = bytearray()

            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue

                data.extend(chunk)

                if len(data) > max_bytes:
                    raise ProjectError(
                        "حجم الملف أكبر من الحد المسموح."
                    )

            return current_url, content_type, bytes(data)

        except requests.RequestException as exc:
            raise ProjectError(
                f"فشل الاتصال أو تنزيل الملف: {exc}"
            ) from exc

        finally:
            if response is not None:
                response.close()

    raise ProjectError("تعذر الوصول إلى الرابط بعد التحويلات.")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def decode_html(html_bytes: bytes, content_type: str) -> str:
    """دعم ترميزات HTML المختلفة، بما فيها بعض الترميزات العربية القديمة."""
    match = re.search(
        r"charset\s*=\s*[\"']?([\w.-]+)",
        content_type or "",
        flags=re.IGNORECASE,
    )

    declared = match.group(1) if match else None

    try:
        dammit = UnicodeDammit(
            html_bytes,
            known_definite_encodings=[declared] if declared else [],
            is_html=True,
        )

        if dammit.unicode_markup:
            return dammit.unicode_markup

    except (LookupError, UnicodeError):
        pass

    return html_bytes.decode("utf-8", errors="replace")


def extract_article(article_url: str) -> dict:
    """استخراج عنوان المقال ونصه وروابط الصور من صفحة المصدر."""
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

    if not title and soup.title:
        title = clean_text(soup.title.get_text(" ", strip=True))

    if not title:
        heading = soup.find("h1")

        if heading:
            title = clean_text(heading.get_text(" ", strip=True))

    if not title:
        raise ProjectError(
            "لم يتم العثور على عنوان واضح للمقال."
        )

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
        fallback_soup = BeautifulSoup(html, "html.parser")

        for tag in fallback_soup(
            ["script", "style", "noscript", "nav", "footer", "aside"]
        ):
            tag.decompose()

        candidates = []

        for selector in ("article", "main", '[role="main"]'):
            candidates.extend(fallback_soup.select(selector))

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
            "لم أتمكن من استخراج نص كافٍ من المقال. "
            "قد يكون الموقع محميًا أو يحتاج إلى JavaScript أو تسجيل دخول."
        )

    image_urls: list[str] = []

    def add_image(candidate: str | None):
        if not isinstance(candidate, str):
            return

        candidate = candidate.strip()

        if not candidate or candidate.startswith("data:"):
            return

        absolute = urldefrag(urljoin(final_url, candidate))[0]

        try:
            parsed_image = urlparse(absolute)

            if parsed_image.scheme.lower() not in ("http", "https"):
                return

            # فحص أمني مبكر؛ يُعاد الفحص قبل التنزيل أيضًا.
            validate_public_url(absolute)

        except (ProjectError, ValueError):
            return

        lowered = parsed_image.path.lower()

        if any(hint in lowered for hint in JUNK_IMAGE_HINTS):
            return

        if absolute not in image_urls:
            image_urls.append(absolute)

    # الصورة الرئيسية المعلنة في بيانات الصفحة.
    for selector in (
        'meta[property="og:image"]',
        'meta[name="twitter:image"]',
    ):
        node = soup.select_one(selector)

        if node:
            add_image(node.get("content"))

    def collect_from(selectors):
        found = []
        seen_ids = set()

        for selector in selectors:
            for node in soup.select(selector):
                if id(node) not in seen_ids:
                    seen_ids.add(id(node))
                    found.append(node)

        return found

    image_nodes = collect_from(
        ("article img", "main img", '[role="main"] img')
    )

    if not image_nodes:
        image_nodes = collect_from(("img",))

    for node in image_nodes:
        candidates = [
            node.get("src"),
            node.get("data-src"),
            node.get("data-lazy-src"),
            node.get("data-original"),
        ]

        srcset = node.get("srcset") or node.get("data-srcset")

        if srcset:
            entries = [
                item.strip().split()[0]
                for item in srcset.split(",")
                if item.strip()
            ]

            if entries:
                candidates.append(entries[-1])

        for candidate in candidates:
            add_image(candidate)

    return {
        "source_url": final_url,
        "title": title,
        "text": article_text,
        "image_urls": image_urls[:30],
    }


def average_hash(image: Image.Image) -> int:
    """بصمة بسيطة للمساعدة على اكتشاف الصور المتشابهة."""
    small = image.convert("L").resize(
        (8, 8),
        Image.Resampling.LANCZOS,
    )

    pixels = list(small.getdata())
    mean = sum(pixels) / len(pixels)

    value = 0

    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)

    return value


def is_duplicate(hash_value: int, known: list[int]) -> bool:
    return any(
        bin(hash_value ^ other).count("1") <= 5
        for other in known
    )


def download_article_images(image_urls: list[str]):
    """تنزيل الصور الصالحة واستبعاد المكررة والصغيرة."""
    collected = []
    hashes: list[int] = []
    attempts = 0

    for image_url in image_urls:
        if len(collected) >= MAX_IMAGES + 4:
            break

        if attempts >= MAX_IMAGE_ATTEMPTS:
            break

        attempts += 1

        try:
            _, _, data = safe_get(
                image_url,
                max_bytes=MAX_IMAGE_BYTES,
                expected_image=True,
            )

            with Image.open(BytesIO(data)) as source:
                source.verify()

            with Image.open(BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")

            width, height = image.size

            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            ratio = width / height

            if ratio > 3.0 or ratio < 0.33:
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
            print(f"تحذير: تم تخطي صورة: {exc}")
            continue

    if not collected:
        return []

    # نحافظ على أول صورة صالحة بوصفها الصورة الرئيسية.
    main_item = collected[0]
    rest = collected[1:]

    rest.sort(
        key=lambda item: item[1].width * item[1].height,
        reverse=True,
    )

    return [main_item] + rest[: MAX_IMAGES - 1]


def crop_to_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> Image.Image:
    """قص الصورة لتملأ المساحة المحددة دون تشويه أبعادها."""
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    return ImageOps.fit(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.35),
    )


def create_collage(
    images: list[Image.Image],
    destination: Path,
):
    """إنشاء تصميم مربع من صورة إلى أربع صور دون كتابة نصوص."""
    if not images:
        raise ProjectError(
            "لم يتم العثور على صور صالحة لإنشاء التصميم."
        )

    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (255, 255, 255),
    )

    gap = IMAGE_GUTTER
    half = (IMAGE_SIZE - gap) // 2

    if len(images) == 1:
        boxes = [
            (0, 0, IMAGE_SIZE, IMAGE_SIZE),
        ]

    elif len(images) == 2:
        boxes = [
            (0, 0, IMAGE_SIZE, half),
            (0, half + gap, IMAGE_SIZE, IMAGE_SIZE),
        ]

    elif len(images) == 3:
        boxes = [
            (half + gap, 0, IMAGE_SIZE, half),
            (0, 0, half, IMAGE_SIZE),
            (half + gap, half + gap, IMAGE_SIZE, IMAGE_SIZE),
        ]

    else:
        boxes = [
            (half + gap, 0, IMAGE_SIZE, half),
            (0, 0, half, half),
            (0, half + gap, half, IMAGE_SIZE),
            (half + gap, half + gap, IMAGE_SIZE, IMAGE_SIZE),
        ]

    for image, box in zip(images, boxes):
        tile = crop_to_box(image, box)
        canvas.paste(tile, (box[0], box[1]))

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        destination,
        "JPEG",
        quality=92,
        optimize=True,
    )

    return destination


def write_text_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        content.rstrip() + "\n",
        encoding="utf-8",
    )


def output_folder_for(url: str) -> Path:
    """إنشاء مجلد ثابت لكل رابط مقال."""
    digest = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:12]

    return OUTPUT_ROOT / digest


def main():
    article_url = os.getenv("ARTICLE_URL", "").strip()

    force = os.getenv(
        "FORCE_REPROCESS", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }

    if not article_url:
        raise ProjectError(
            "لم يتم إدخال رابط المقال. "
            "شغّل workflow يدويًا وأدخل ARTICLE_URL."
        )

    # التحقق من الرابط قبل استخدامه في تحديد مجلد الإخراج.
    article_url = validate_public_url(article_url)
    out_dir = output_folder_for(article_url)

    if (out_dir / "result.json").exists() and not force:
        print(
            f"تمت معالجة هذا الرابط سابقًا: {out_dir}\n"
            "فعّل force_reprocess لإعادة المعالجة."
        )
        return

    print("1/5: استخراج المقال...")
    article = extract_article(article_url)

    print(f"العنوان الأصلي: {article['title']}")
    print(f"عدد أحرف المقال: {len(article['text'])}")
    print(f"روابط الصور المكتشفة: {len(article['image_urls'])}")

    print("2/5: إعادة صياغة المقال باستخدام Grok...")
    rewritten = rewrite_article(
        article_title=article["title"],
        article_text=article["text"],
        source_url=article["source_url"],
    )

    print("3/5: تنزيل الصور...")
    selected = download_article_images(article["image_urls"])

    images = [image for _, image in selected]
    used_urls = [url for url, _ in selected]

    print(f"عدد الصور المختارة: {len(images)}")

    print("4/5: إنشاء التصميم المربع...")
    image_file = None
    image_note = ""

    try:
        create_collage(
            images,
            out_dir / "facebook_image.jpg",
        )

        image_file = "facebook_image.jpg"

    except ProjectError as exc:
        image_note = str(exc)
        print(f"تحذير: {image_note}")

    print("5/5: حفظ النتائج...")
    hashtags_text = " ".join(rewritten["hashtags"])

    post_parts = [
        rewritten["title"],
        rewritten["facebook_post"],
    ]

    if hashtags_text:
        post_parts.append(hashtags_text)

    post_parts.append(
        f"المصدر: {article['source_url']}"
    )

    write_text_file(
        out_dir / "facebook_post.txt",
        "\n\n".join(post_parts),
    )

    rewritten_markdown = (
        f"# {rewritten['title']}\n\n"
        f"**رابط المقال الأصلي:** {article['source_url']}\n\n"
        "---\n\n"
        f"{rewritten['rewritten_article']}\n"
    )

    write_text_file(
        out_dir / "rewritten_article.md",
        rewritten_markdown,
    )

    result = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
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
        "note": (
            "تحقق من حقوق استخدام الصور قبل نشرها. "
            "وجود الصورة في صفحة المصدر لا يمنح تلقائيًا حق إعادة نشرها. "
            "المحتوى المعاد صياغته لم يخضع لتحقق مستقل من الوقائع."
        ),
    }

    write_text_file(
        out_dir / "result.json",
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
    )

    print("\nاكتمل إنشاء المحتوى.")

    for name in (
        "facebook_image.jpg",
        "facebook_post.txt",
        "rewritten_article.md",
        "result.json",
    ):
        path = out_dir / name

        if path.exists():
            print(f"- {path}")


if __name__ == "__main__":
    try:
        main()

    except (ProjectError, GrokError) as exc:
        print(
            f"\nخطأ: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    except KeyboardInterrupt:
        print(
            "\nتم إيقاف التشغيل.",
            file=sys.stderr,
        )
        sys.exit(130)

    except Exception as exc:
        print(
            f"\nخطأ غير متوقع: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
