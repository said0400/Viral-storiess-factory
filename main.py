
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
    ImageFilter,
    ImageOps,
    UnidentifiedImageError,
)

from grok import GrokError, analyze_images, rewrite_article


OUTPUT_ROOT = Path("output")
SIZE = 1080
GUTTER = 10
MAX_PAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 4
MAX_IMAGE_ATTEMPTS = 20
MIN_IMAGE_WIDTH = 350
MIN_IMAGE_HEIGHT = 250
REQUEST_TIMEOUT = 30

JUNK_HINTS = (
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
    "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ar,en;q=0.8",
})


class ProjectError(Exception):
    """خطأ متوقع يمكن عرضه للمستخدم."""


def validate_public_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ProjectError("رابط المقال فارغ.")

    parsed = urlparse(url.strip())

    if parsed.scheme.lower() not in ("http", "https"):
        raise ProjectError("يجب أن يبدأ الرابط بـ http:// أو https://.")

    if not parsed.hostname or parsed.username or parsed.password:
        raise ProjectError("الرابط غير صالح أو يحتوي على بيانات دخول.")

    host = parsed.hostname.rstrip(".").lower()

    if host in {"localhost", "localhost.localdomain"}:
        raise ProjectError("لا يُسمح باستخدام رابط محلي.")

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ProjectError("منفذ الرابط غير صالح.") from exc

    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ProjectError(f"تعذر العثور على خادم {host}.") from exc

    if not addresses:
        raise ProjectError("لم يتم العثور على عنوان IP للموقع.")

    for address in addresses:
        raw_ip = address[4][0].split("%")[0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise ProjectError("عنوان IP غير صالح.") from exc

        if not ip.is_global:
            raise ProjectError("رُفض الرابط لأنه يشير إلى عنوان غير عام.")

    return url.strip()


def safe_get(url: str, max_bytes: int, expected_image: bool = False):
    """تنزيل محدود الحجم مع التحقق من كل وجهة تحويل."""
    current = url

    for _ in range(6):
        validate_public_url(current)

        try:
            response = SESSION.get(
                current,
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

            current = urljoin(current, location)
            continue

        if not 200 <= response.status_code < 300:
            status = response.status_code
            response.close()
            hint = " قد يمنع الموقع الوصول الآلي." if status in (401, 403, 429) else ""
            raise ProjectError(f"أعاد الموقع HTTP {status}.{hint}")

        content_type = response.headers.get("Content-Type", "").lower()

        if expected_image and not content_type.startswith("image/"):
            response.close()
            raise ProjectError("الرابط لا يعيد ملف صورة.")

        length = response.headers.get("Content-Length", "")
        try:
            if length and int(length) > max_bytes:
                response.close()
                raise ProjectError("الملف أكبر من الحد المسموح.")
        except ValueError:
            pass

        data = bytearray()

        try:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise ProjectError("الملف أكبر من الحد المسموح.")
        except requests.RequestException as exc:
            raise ProjectError(f"انقطع تنزيل الملف: {exc}") from exc
        finally:
            response.close()

        return current, content_type, bytes(data)

    raise ProjectError("تجاوز الرابط عدد التحويلات المسموح.")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def decode_html(data: bytes, content_type: str) -> str:
    match = re.search(r"charset=([\w\-]+)", content_type or "")
    encodings = [match.group(1)] if match else []

    decoded = UnicodeDammit(
        data,
        known_definite_encodings=encodings,
        is_html=True,
    )
    return decoded.unicode_markup or data.decode("utf-8", errors="replace")


def extract_article(url: str) -> dict:
    final_url, content_type, data = safe_get(url, MAX_PAGE_BYTES)

    if "html" not in content_type and "xhtml" not in content_type:
        raise ProjectError("الرابط ليس صفحة مقال HTML.")

    html = decode_html(data, content_type)
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
    title = title or "مقال بدون عنوان واضح"

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
            ["script", "style", "noscript", "nav", "footer", "aside", "form"]
        ):
            tag.decompose()

        candidates = fallback.select("article, main, [role='main']")
        candidates.sort(
            key=lambda node: len(node.get_text(" ", strip=True)),
            reverse=True,
        )
        if candidates:
            article_text = clean_text(
                candidates[0].get_text(" ", strip=True)
            )

    if len(article_text) < 200:
        raise ProjectError(
            "لم يُستخرج نص كافٍ. قد يكون الموقع محميًا أو يعتمد على JavaScript."
        )

    image_urls: list[str] = []

    def add_image(candidate):
        if not isinstance(candidate, str) or not candidate.strip():
            return

        candidate = candidate.strip()
        if candidate.startswith("data:"):
            return

        absolute = urldefrag(urljoin(final_url, candidate))[0]
        if urlparse(absolute).scheme not in ("http", "https"):
            return

        if any(hint in absolute.lower() for hint in JUNK_HINTS):
            return

        if absolute not in image_urls:
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
        for key in (
            "src", "data-src", "data-lazy-src", "data-original",
        ):
            add_image(node.get(key))

        srcset = node.get("srcset") or node.get("data-srcset")
        if srcset:
            entries = [
                part.strip().split()[0]
                for part in srcset.split(",")
                if part.strip()
            ]
            if entries:
                add_image(entries[-1])

    return {
        "source_url": final_url,
        "title": title,
        "text": article_text,
        "image_urls": image_urls[:40],
    }


def download_article_images(urls: list[str]) -> list[tuple[str, Image.Image]]:
    collected = []
    seen_hashes = set()

    for url in urls[:MAX_IMAGE_ATTEMPTS]:
        if len(collected) >= MAX_IMAGES:
            break

        try:
            _, _, data = safe_get(
                url, MAX_IMAGE_BYTES, expected_image=True
            )

            with Image.open(BytesIO(data)) as source:
                source.verify()

            with Image.open(BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")

            width, height = image.size
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue
            if width / height > 3.5 or width / height < 0.28:
                continue

            thumb = image.resize((16, 16), Image.Resampling.LANCZOS).convert("L")
            digest = hashlib.sha256(thumb.tobytes()).hexdigest()

            if digest in seen_hashes:
                continue

            seen_hashes.add(digest)
            collected.append((url, image.copy()))

        except (
            ProjectError,
            requests.RequestException,
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
        ) as exc:
            print(f"تحذير: تم تخطي صورة: {exc}")

    return collected


def crop_from_plan(
    image: Image.Image,
    normalized: list[float],
) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = normalized

    box = (
        max(0, min(width - 1, round(left * width))),
        max(0, min(height - 1, round(top * height))),
        max(1, min(width, round(right * width))),
        max(1, min(height, round(bottom * height))),
    )

    if box[2] <= box[0] or box[3] <= box[1]:
        return image.copy()

    return image.crop(box)


def fill_box(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    return ImageOps.fit(
        image,
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.45),
    )


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1),
        radius=radius,
        fill=255,
    )
    return mask


def add_inset(
    canvas: Image.Image,
    detail: Image.Image,
    shape: str,
    position: str,
):
    """تفصيل مكبّر بحد أبيض وظل خفيف، دون أي نص أو شعار."""
    inset_size = 350
    margin = 46
    border = 9

    detail = fill_box(detail, (inset_size, inset_size))
    outer_size = inset_size + border * 2

    layer = Image.new("RGBA", (outer_size + 30, outer_size + 30), (0, 0, 0, 0))
    shadow_mask = Image.new("L", (outer_size, outer_size), 0)
    shadow_draw = ImageDraw.Draw(shadow_mask)

    if shape == "circle":
        shadow_draw.ellipse((0, 0, outer_size - 1, outer_size - 1), fill=180)
    else:
        shadow_draw.rounded_rectangle(
            (0, 0, outer_size - 1, outer_size - 1),
            radius=22,
            fill=180,
        )

    shadow = Image.new("RGBA", shadow_mask.size, (0, 0, 0, 160))
    shadow.putalpha(shadow_mask.filter(ImageFilter.GaussianBlur(8)))
    layer.alpha_composite(shadow, (12, 12))

    frame = Image.new("RGBA", (outer_size, outer_size), (255, 255, 255, 255))
    inner_mask = Image.new("L", (inset_size, inset_size), 0)
    draw = ImageDraw.Draw(inner_mask)

    if shape == "circle":
        draw.ellipse((0, 0, inset_size - 1, inset_size - 1), fill=255)
    else:
        draw.rounded_rectangle(
            (0, 0, inset_size - 1, inset_size - 1),
            radius=15,
            fill=255,
        )

    frame.paste(detail, (border, border), inner_mask)
    layer.alpha_composite(frame, (0, 0))

    if "left" in position:
        x = margin
    else:
        x = SIZE - margin - outer_size

    if "top" in position:
        y = margin
    else:
        y = SIZE - margin - outer_size

    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.alpha_composite(layer, (x - 10, y - 10))
    canvas.paste(canvas_rgba.convert("RGB"))


def create_design(
    selected: list[tuple[str, Image.Image]],
    plan: dict,
    destination: Path,
) -> Path:
    if not selected:
        raise ProjectError("لا توجد صور صالحة لإنشاء التصميم.")

    images = [item[1] for item in selected]
    order = [i for i in plan["image_order"] if i < len(images)]
    images = [images[i] for i in order]
    if not images:
        raise ProjectError("خطة ترتيب الصور فارغة.")

    crops_by_original = plan["crops"]
    ordered_crops = [crops_by_original[i] for i in order]

    canvas = Image.new("RGB", (SIZE, SIZE), (18, 18, 18))
    layout = plan["layout"]

    if len(images) == 1:
        layout = "single_inset"
    elif len(images) == 2 and layout in ("three_panel", "four_grid"):
        layout = "two_panel"
    elif len(images) == 3 and layout == "four_grid":
        layout = "three_panel"
    elif len(images) >= 4 and layout not in ("four_grid", "three_panel"):
        layout = "four_grid"

    gap = GUTTER
    half = (SIZE - gap) // 2

    if layout == "single_inset":
        main_crop = crop_from_plan(images[0], ordered_crops[0])
        canvas = fill_box(main_crop, (SIZE, SIZE))

        detail_index = plan["detail_image_index"]
        if detail_index >= len(selected):
            detail_index = order[0]

        detail = crop_from_plan(
            selected[detail_index][1],
            plan["detail_crop"],
        )
        add_inset(
            canvas,
            detail,
            plan["inset_shape"],
            plan["inset_position"],
        )

    elif layout == "two_panel":
        boxes = [
            (0, 0, half, SIZE),
            (half + gap, 0, SIZE, SIZE),
        ]
        for image, crop, box in zip(images[:2], ordered_crops[:2], boxes):
            tile = fill_box(crop_from_plan(image, crop), (
                box[2] - box[0], box[3] - box[1]
            ))
            canvas.paste(tile, (box[0], box[1]))

    elif layout == "three_panel":
        boxes = [
            (0, 0, half, SIZE),
            (half + gap, 0, SIZE, half),
            (half + gap, half + gap, SIZE, SIZE),
        ]
        for image, crop, box in zip(images[:3], ordered_crops[:3], boxes):
            tile = fill_box(crop_from_plan(image, crop), (
                box[2] - box[0], box[3] - box[1]
            ))
            canvas.paste(tile, (box[0], box[1]))

    else:
        boxes = [
            (0, 0, half, half),
            (half + gap, 0, SIZE, half),
            (0, half + gap, half, SIZE),
            (half + gap, half + gap, SIZE, SIZE),
        ]
        for image, crop, box in zip(images[:4], ordered_crops[:4], boxes):
            tile = fill_box(crop_from_plan(image, crop), (
                box[2] - box[0], box[3] - box[1]
            ))
            canvas.paste(tile, (box[0], box[1]))

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, "JPEG", quality=94, optimize=True)
    return destination


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def output_folder_for(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return OUTPUT_ROOT / digest


def main():
    article_url = os.getenv("ARTICLE_URL", "").strip()
    force = os.getenv("FORCE_REPROCESS", "").lower() in {
        "1", "true", "yes",
    }

    if not article_url:
        raise ProjectError("أدخل رابط المقال من نموذج تشغيل GitHub Actions.")

    article_url = validate_public_url(article_url)
    out_dir = output_folder_for(article_url)

    if (out_dir / "result.json").exists() and not force:
        print(
            f"هذا الرابط عولج سابقًا: {out_dir}. "
            "فعّل force_reprocess لإعادة المعالجة."
        )
        return

    print("1/6: استخراج المقال والصور...")
    article = extract_article(article_url)

    print(f"العنوان: {article['title']}")
    print(f"طول النص: {len(article['text'])} حرفًا")
    print(f"روابط الصور المكتشفة: {len(article['image_urls'])}")

    print("2/6: إعادة كتابة المقال والمنشور...")
    rewritten = rewrite_article(
        article["title"],
        article["text"],
        article["source_url"],
    )

    print("3/6: تنزيل الصور والتحقق منها...")
    selected = download_article_images(article["image_urls"])
    print(f"الصور الصالحة: {len(selected)}")

    image_file = None
    image_note = ""
    image_plan = None

    if selected:
        print("4/6: تحليل الصور واختيار التصميم والقص باستخدام Groq Vision...")
        try:
            image_plan = analyze_images([image for _, image in selected])
            print("التكوين المختار:", image_plan["layout"])
            print("سبب التصميم:", image_plan.get("reason", ""))

            print("5/6: تنفيذ التصميم المربع...")
            create_design(
                selected,
                image_plan,
                out_dir / "facebook_image.jpg",
            )
            image_file = "facebook_image.jpg"

        except (GrokError, ProjectError, OSError, ValueError) as exc:
            image_note = (
                "تعذر إكمال تحليل الصور أو التصميم. "
                f"التفاصيل: {exc}"
            )
            print("تحذير:", image_note)
    else:
        image_note = (
            "لم يتم العثور على صور صالحة. "
            "لم تُنشأ صورة بديلة من مصدر غير متعلق بالمقال."
        )
        print("تحذير:", image_note)

    print("6/6: حفظ ملفات النتائج...")
    hashtags = " ".join(rewritten["hashtags"])

    post_parts = [
        rewritten["title"],
        rewritten["facebook_post"],
    ]
    if hashtags:
        post_parts.append(hashtags)
    post_parts.append(f"المصدر: {article['source_url']}")

    write_text(out_dir / "facebook_post.txt", "\n\n".join(post_parts))

    markdown = (
        f"# {rewritten['title']}\n\n"
        f"**رابط المقال الأصلي:** {article['source_url']}\n\n"
        "---\n\n"
        f"{rewritten['rewritten_article']}\n"
    )
    write_text(out_dir / "rewritten_article.md", markdown)

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": article["source_url"],
        "original_title": article["title"],
        "title": rewritten["title"],
        "facebook_post": rewritten["facebook_post"],
        "hashtags": rewritten["hashtags"],
        "rewritten_article": rewritten["rewritten_article"],
        "image_file": image_file,
        "image_count": len(selected),
        "image_source_urls": [url for url, _ in selected],
        "image_plan": image_plan,
        "image_note": image_note,
        "rights_note": (
            "تحقق من امتلاك حق إعادة نشر الصور والمقال. "
            "وجود الصور في صفحة المصدر لا يمنح تلقائيًا حق إعادة استخدامها."
        ),
        "fact_check_note": (
            "أعيدت صياغة النص اعتمادًا على المادة المستخرجة؛ "
            "لم يُجرَ تحقق مستقل من الوقائع."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        out_dir / "result.json",
        json.dumps(result, ensure_ascii=False, indent=2),
    )

    print("\nاكتمل التشغيل. الملفات في:", out_dir)
    for name in (
        "facebook_image.jpg",
        "facebook_post.txt",
        "rewritten_article.md",
        "result.json",
    ):
        path = out_dir / name
        if path.is_file():
            print("-", path)


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
