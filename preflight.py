"""
فحص ما قبل التشغيل: يكتشف في أول ثوانٍ لماذا قد يفشل المشروع، ويطبع سببًا واضحًا.

يتحقق من:
  1) وجود كل ملفات المشروع في جذر المستودع.
  2) أن كل ملف يُستورد بلا أخطاء (يعرض سبب الفشل الحقيقي).
  3) أن إصدارات الملفات متطابقة (ملف قديم مع ملف جديد يسبب أعطالًا غريبة).
  4) تثبيت المكتبات المطلوبة.
  5) الأسرار: الناقص الحرج خطأ، والاختياري تحذير (لا تُطبع قيمها أبدًا).

التشغيل: python preflight.py   (يخرج بالرمز 1 إن وُجد خلل حرج)
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

# الملف: ما يجب أن يوفّره (أسماء تكشف الإصدار القديم من الجديد)
EXPECTED: dict[str, list[str]] = {
    "categories": ["CATEGORIES", "DEFAULT_CATEGORY", "labels_for"],
    "key_pool": ["load_keys", "usable", "mark_dead", "split_keys"],
    "grok": ["analyze_images", "_images_for_attempt", "fallback_plan", "default_gallery"],
    "gemini": ["analyze_images", "_generate"],
    "cloudflare_ai": ["generate_image", "CloudflareQuotaError", "CloudflareFlaggedError",
                      "fallback_model_name"],
    "thumbnail_design": ["extract_subjects", "compose"],
    "wide_collage": ["create_gallery_collage", "CollageError"],
    "thumbnail": ["create_thumbnail", "_try_ai_full_ladder"],
    "fb_cover": ["create_design", "facebook_height"],
    "writer": ["write_article", "rejection_reason", "build_first_comment_text",
               "build_post_text", "CATEGORIES", "build_source_line", "OPINION_RULES"],
    "related": ["published_articles"],
    "main": ["main", "StoryRejected", "output_folder_for", "extract_article"],
    "picker": ["pick_story", "rank_items", "collect_items"],
    "run_factory": ["run"],
    "make_publish": ["publish", "fill_first_comment", "labels_for"],
    "story_categories": ["STORY_CATEGORIES", "next_category", "load_story_results",
                         "canonical_category", "labels_for"],
    "story_writer": ["write_story", "load_history", "append_related", "SEO_RULES",
                     "build_author_line"],
    "story_image": ["build_story_images"],
    "story_factory": ["main", "run"],
    "story_text": ["clean_story_text", "story_paragraphs", "split_chunks", "AUTHOR_MARK"],
    "story_scenes": ["make_storyboard", "generate_scene_images", "scene_segments"],
    "story_video": ["main", "run", "clean_story_text", "insert_video_links", "make_slideshow",
                    "build_chapters", "build_hashtags"],
    "whatsapp_notify": ["main", "notify"],
}

THIRD_PARTY = {          # الوحدة: (حرجة؟، اسم الحزمة في pip)
    "requests": (True, "requests"),
    "PIL": (True, "Pillow"),
    "bs4": (True, "beautifulsoup4"),
    "trafilatura": (True, "trafilatura"),
    "numpy": (True, "numpy"),
    "arabic_reshaper": (False, "arabic-reshaper (النص العربي على الصور)"),
    "bidi": (False, "python-bidi (النص العربي على الصور)"),
    "rembg": (False, "rembg (وضع hybrid فقط)"),
}

SECRETS = {              # الاسم: (حرج؟، الغرض)
    "GROQ_API_KEY": (True, "الكتابة الاحتياطية وتحليل الصور"),
    "GEMINI_WRITER_API_KEY": (False, "كتابة المقال (الأفضل)"),
    "GEMINI_API_KEY": (False, "تحليل الصور"),
    "CF_ACCOUNT_ID": (False, "الصورة المصغرة بالذكاء الاصطناعي"),
    "CF_API_TOKEN": (False, "الصورة المصغرة بالذكاء الاصطناعي"),
    "MAKE_WEBHOOK_URL": (False, "النشر على Blogger عبر Make"),
    "BLOGGER_BLOG_ID": (False, "جلب رابط المقال بعد النشر (رقم المدونة)"),
    "GREENAPI_ID_INSTANCE": (False, "واتساب"),
    "GREENAPI_API_TOKEN": (False, "واتساب"),
    "GREENAPI_CHAT_ID": (False, "واتساب"),
}


def main(root: Path | None = None) -> int:
    root = Path(root or Path.cwd())
    sys.path.insert(0, str(root))
    critical: list[str] = []
    warnings: list[str] = []

    print("== 1) الملفات ==")
    for name in EXPECTED:
        ok = (root / f"{name}.py").is_file()
        print(f"  {'OK     ' if ok else 'مفقود '} {name}.py")
        if not ok:
            critical.append(f"الملف {name}.py غير موجود في جذر المستودع")

    prompt = root / "prompts" / "story_style.txt"
    ok = prompt.is_file() and prompt.read_text(encoding="utf-8").strip() != ""
    print(f"  {'OK     ' if ok else 'مفقود '} prompts/story_style.txt")
    if not ok:
        critical.append("الملف prompts/story_style.txt غير موجود أو فارغ (برومبت القصص)")

    music_dir = root / "music"
    tracks = [f.name for f in music_dir.glob("*")
              if f.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}] \
        if music_dir.is_dir() else []
    print(f"  {'OK     ' if tracks else 'تنبيه '} music/ ({len(tracks)} ملف موسيقى)")
    if not tracks:
        warnings.append("لا ملفات موسيقى في المجلد music/ (فيديو القصة بلا موسيقى خلفية)")

    print("== 2) المكتبات ==")
    for module, (is_critical, pip_name) in THIRD_PARTY.items():
        try:
            importlib.import_module(module)
            print(f"  OK     {module}")
        except ImportError:
            print(f"  ناقصة  {module}")
            (critical if is_critical else warnings).append(f"المكتبة {pip_name} غير مثبتة")

    print("== 3) استيراد الملفات وتطابق الإصدارات ==")
    for name, attrs in EXPECTED.items():
        if not (root / f"{name}.py").is_file():
            continue
        try:
            module = importlib.import_module(name)
        except Exception as exc:                      # noqa: BLE001 - نريد كل الأسباب
            last = traceback.format_exception_only(type(exc), exc)[-1].strip()
            print(f"  فشل    {name}: {last}")
            critical.append(f"تعذر استيراد {name}.py: {last}")
            continue
        missing = [a for a in attrs if not hasattr(module, a)]
        if missing:
            print(f"  قديم؟  {name}: ينقصه {', '.join(missing)}")
            critical.append(
                f"{name}.py لا يطابق بقية الملفات (ينقصه: {', '.join(missing)}). "
                "على الأرجح نسخة قديمة؛ استبدله بأحدث نسخة")
        else:
            print(f"  OK     {name}")

    print("== 4) الأسرار (لا تُطبع قيمها) ==")
    for key, (is_critical, purpose) in SECRETS.items():
        present = bool(os.getenv(key, "").strip())
        print(f"  {'موجود ' if present else 'غير موجود'} {key}  ({purpose})")
        if not present:
            (critical if is_critical else warnings).append(f"السر {key} غير مضبوط ({purpose})")

    try:                                              # عدد الحسابات المضبوطة لكل خدمة
        import key_pool
        for label, names in (("Gemini للكتابة", ("GEMINI_WRITER_API_KEY",)),
                             ("Gemini للصوت والصور", ("GEMINI_TTS_API_KEY", "GEMINI_API_KEY"))):
            print(f"  عدد حسابات {label}: {len(key_pool.load_keys(*names))}")
    except Exception:                                 # noqa: BLE001 - معلومة إضافية فقط
        pass

    for message in warnings:
        print(f"::warning title=تحذير::{message}")
    for message in critical:
        print(f"::error title=خلل يمنع التشغيل::{message}")

    if critical:
        print(f"\nالنتيجة: {len(critical)} خلل حرج يجب إصلاحه قبل التشغيل.")
        return 1
    print(f"\nالنتيجة: جاهز للتشغيل ({len(warnings)} تحذير غير حرج).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
