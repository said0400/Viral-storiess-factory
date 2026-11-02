from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
import trafilatura
from bs4 import BeautifulSoup, UnicodeDammit
from PIL import Image, ImageOps, UnidentifiedImageError

import gemini
import thumbnail
import wide_collage
import writer
import related
from fb_cover import create_design
from grok import (
    MAX_VISION_IMAGES,
    default_gallery,
    GrokError,
    analyze_images,
    fallback_plan,
)


OUTPUT_ROOT = Path("output")

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

# حزمة ترويسات كاملة لتبدو كمتصفح حقيقي يتصفح مقالًا طبيعيًا
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
    "Referer": "https://google.com",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

SESSION = requests.Session()


class ProjectError(Exception):
    """خطأ متوقع أثناء استخراج المقال أو إنشاء النتائج."""


class StoryRejected(ProjectError):
    """القصة مرفوضة (مخالفات واقعية في المقال): يُختار موضوع آخر بدل النشر."""


# ---------------------------------------------------------------------------
# التنزيل واستخراج المقال
# ---------------------------------------------------------------------------

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
    if hostname in {"localhost", "localhost.localdomain", "127.0.0.1"}:
        raise ProjectError("الروابط المحلية غير مسموحة.")

    return url


def safe_get(url: str, max_bytes: int, expected_image: bool = False):
    validate_public_url(url)

    try:
        response = SESSION.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            stream=True,
        )
    except requests.RequestException as exc:
        raise ProjectError(f"فشل الاتصال بالرابط: {exc}") from exc

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

    return response.url or url, content_type, bytes(data)


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


# ---------------------------------------------------------------------------
# تنزيل الصور
# ---------------------------------------------------------------------------

def average_hash(image: Image.Image) -> int:
    small = image.convert("L").resize(
        (8, 8), Image.Resampling.LANCZOS
    )
    pixels = list(small.tobytes())
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
# تحليل الصور والصور العريضة
# (تصميم صورة فيسبوك نفسه في fb_cover.py)
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


# ---------------------------------------------------------------------------
# الملفات والتشغيل
# ---------------------------------------------------------------------------

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
    content_category = os.getenv("CONTENT_CATEGORY", "").strip()
    related_articles = related.published_articles(
        category=content_category,
        exclude_source=article["source_url"],
    )
    print("مقالات سابقة متاحة للربط:", len(related_articles))
    try:
        rewritten = writer.write_article(
            article_title=article["title"],
            article_text=article["text"],
            source_url=article["source_url"],
            category=content_category,
            related=related_articles,
        )
    except writer.WriterError as exc:
        raise ProjectError(f"فشلت كتابة المقال: {exc}") from exc
    print(
        f"الكاتب: {rewritten['writer']['provider']} "
        f"({rewritten['writer']['model']})، "
        f"الكلمات: {rewritten['stats']['words']}"
    )

    # وضع الاختيار التلقائي: لا يُنشر مقال فيه مخالفات واقعية، بل تُرفض القصة
    if os.getenv("REJECT_ON_ISSUES", "").strip().lower() in {"1", "true", "yes"}:
        reject_reason = writer.rejection_reason(rewritten)
        if reject_reason:
            raise StoryRejected(reject_reason)

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
            # نسخة للصورتين العريضتين قبل أن يضيف التصميم بياناته إلى الخطة
            wide_plan = dict(plan)
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
        out_dir / "first_comment.txt",
        writer.build_first_comment_text(rewritten),
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
        "first_comment": rewritten["first_comment"],
        "category": rewritten["category"],
        "related_articles": rewritten.get("related_used", []),
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

    # نحافظ على بيانات النشر السابق (Make/Blogger) حتى يُحدَّث المقال لا يتكرر
    previous_path = out_dir / "result.json"
    if previous_path.is_file():
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if previous.get("blogger"):
                result["blogger"] = previous["blogger"]
        except (OSError, ValueError):
            pass

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
        "first_comment.txt",
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
