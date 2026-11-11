"""
خط إنتاج القصص: يولّد قصة أصلية واحدة بالتصنيف التالي في الدوران، ويكتب حزمة
بالأسماء نفسها التي يستعملها خط الأخبار، فيعمل النشر عبر Make وواتساب دون أي تغيير:
  facebook_image.jpg, thumbnail.jpg, facebook_post.txt, first_comment.txt,
  rewritten_article.md, seo.txt, editor_notes.txt, result.json

لا احتياط من Groq: إن فشلت الكتابة يخرج بخطأ ولا يُنشر شيء، وتعيد دورة cron التالية.

التشغيل: python story_factory.py   (متغيرات اختيارية: STORY_CATEGORY, STORY_MIN_WORDS)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import related
import story_categories
import story_image
import story_scenes
import story_writer
from story_writer import StoryError

OUTPUT_ROOT = Path("output")
WORK_ROOT = Path(os.getenv("STORY_WORK_DIR", "").strip() or "story_work")


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def publish_output_dir(out_dir: Path, work_dir: Path | None = None) -> None:
    """يكتب مسار مجلد هذه القصة (ومجلد عمل الفيديو) ليستعملهما الـ workflow."""
    target = os.getenv("GITHUB_OUTPUT")
    if not target:
        return
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(f"out_dir={out_dir.as_posix()}\n")
        if work_dir is not None:
            handle.write(f"work_dir={work_dir.as_posix()}\n")


def build_seo_text(story: dict[str, Any]) -> str:
    seo = story["seo"]
    return "\n".join([
        f"العنوان الرئيسي: {story['title']}",
        f"عنوان الصورة: {story['image_title']}",
        f"التصنيف: {story['category']}",
        f"عنوان SEO: {seo['meta_title']}",
        f"وصف الميتا ({len(seo['meta_description'])} حرفًا): {seo['meta_description']}",
        f"الكلمة المفتاحية: {seo['focus_keyword']}",
        f"الرابط المختصر (slug): {seo['slug']}",
        "الكلمات المفتاحية: " + "، ".join(seo["keywords"]),
    ])


def build_notes(story: dict[str, Any]) -> str:
    writer, stats, log = story["writer"], story["stats"], story["story_log"]
    lines = [
        "ملاحظات المحرر (للاستخدام الداخلي، لا تُنشر)",
        "=" * 40,
        f"التصنيف: {story['category']}",
        f"الكاتب: {writer['provider']} ({writer['model']})، المحاولات: {writer['attempts']}، "
        f"توسيعات القصة القصيرة: {writer.get('expansions', 0)}",
        f"عدد الكلمات: {stats['words']} (الحد الأدنى {stats['min_words']}) "
        f"— مدة القراءة نحو {stats['reading_minutes']} دقيقة",
        "",
        "سجل القصة (يُستعمل لمنع التكرار):",
    ]
    labels = {"idea": "الفكرة", "conflict": "الصراع", "characters": "الشخصيات",
              "setting": "المكان", "opening": "الافتتاحية", "twist": "المفاجأة",
              "ending": "النهاية"}
    lines += [f"- {labels[k]}: {log.get(k, '')}" for k in story_writer.LOG_KEYS]
    lines += ["", "ملاحظات الفحص:"]
    lines += [f"- {p}" for p in story["problems"]] or ["- لا ملاحظات."]
    lines += ["", "الروابط الداخلية:"]
    lines += [f"- {i['title']} ← {i['url']}" for i in story["related_used"]] \
        or ["- لا روابط داخلية (لا قصص منشورة سابقة بعد)."]
    return "\n".join(lines)


def run() -> Path:
    category = story_categories.next_category(
        OUTPUT_ROOT, os.getenv("STORY_CATEGORY", ""))
    print(f"تصنيف هذه القصة: {category}")

    history = story_writer.load_history(OUTPUT_ROOT)
    print(f"قصص سابقة في السجل: {len(history)}")

    story = story_writer.write_story(category, history)
    print(f"العنوان: {story['title']}")
    print(f"الكلمات: {story['stats']['words']} | الكاتب: {story['writer']['model']}")

    # لوحة مشاهد من أحداث القصة نفسها: تعطي وصفًا خاصًا بالغلاف، وتُستعمل صور مشاهدها في الفيديو.
    # فشلها لا يمنع القصة: يُستعمل وصف الصورة العام الذي كتبه كاتب القصة.
    storyboard = None
    if story_scenes.scenes_enabled():
        try:
            storyboard = story_scenes.make_storyboard(
                story["title"], story["article"], category)
            story["image_prompt"] = storyboard["cover_prompt"]
            story["image_safe_prompt"] = storyboard["cover_safe_prompt"]
            print(f"لوحة المشاهد: {len(storyboard['scenes'])} مشاهد.")
        except Exception as exc:  # noqa: BLE001 - اللوحة إضافة، لا توقف إنتاج القصة
            print(f"::warning title=لوحة المشاهد::{type(exc).__name__}: {exc}")

    rel = related.published_articles(category=category, kind="story")
    story["article"], story["related_used"] = story_writer.append_related(
        story["article"], rel)

    stamp = datetime.now(timezone.utc)
    digest = hashlib.sha256(
        f"story|{stamp.isoformat()}|{story['title']}".encode("utf-8")
    ).hexdigest()[:12]
    out_dir = OUTPUT_ROOT / digest
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = WORK_ROOT / digest
    images = story_image.build_story_images(story, category, out_dir, work_dir)

    post_text = story["facebook_post"]
    if story["hashtags"]:
        post_text += "\n\n" + " ".join(story["hashtags"])
    write_text_file(out_dir / "facebook_post.txt", post_text)
    write_text_file(out_dir / "first_comment.txt", story["first_comment"])
    write_text_file(out_dir / "rewritten_article.md",
                    f"# {story['title']}\n\n{story['article'].strip()}\n")
    write_text_file(out_dir / "seo.txt", build_seo_text(story))
    write_text_file(out_dir / "editor_notes.txt", build_notes(story))

    result = {
        "kind": "story",
        "generated_at_utc": stamp.isoformat(),
        "title": story["title"],
        "original_title": story["title"],
        "category": category,
        "labels": story_categories.labels_for(category),
        "facebook_post": story["facebook_post"],
        "first_comment": story["first_comment"],
        "hashtags": story["hashtags"],
        "rewritten_article": story["article"],
        "related_articles": story["related_used"],
        "seo": story["seo"],
        "writer": story["writer"],
        "article_stats": story["stats"],
        "story_log": story["story_log"],
        "story_problems": story["problems"],
        "image_file": "facebook_image.jpg",
        "wide_images": {"middle": None, "thumbnail": images["thumbnail"], "errors": []},
        "image_title": story["image_title"],
        "image_text": story["image_title"],
        "image_highlight": story["image_highlight"],
        "image_prompt": story["image_prompt"],
        "storyboard": storyboard,
        "image_note": images["generation"].get("reason", ""),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    publish_output_dir(out_dir, work_dir)
    print("\nاكتمل إنشاء القصة:", out_dir)
    for name in ("facebook_image.jpg", "thumbnail.jpg", "facebook_post.txt",
                 "first_comment.txt", "rewritten_article.md", "seo.txt",
                 "editor_notes.txt", "result.json"):
        if (out_dir / name).exists():
            print("-", out_dir / name)
    return out_dir


def main() -> int:
    try:
        run()
    except StoryError as exc:
        print(f"::error title=فشل إنشاء القصة::{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nتم إيقاف التشغيل.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - أي عطل غير متوقع لا يترك الـ workflow معلقًا
        print(f"::error title=عطل غير متوقع في خط القصص::{type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
