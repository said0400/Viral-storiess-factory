"""
تشغيل خط الإنتاج مع إعادة اختيار قصة أخرى تلقائيًا عند رفض القصة.

- إن أُدخل رابط يدويًا: يعمل كما كان (main.py)، بلا رفض ولا إعادة اختيار.
- في الاختيار التلقائي (AUTO_PICK=true):
    * إن وجد الفحص مخالفات واقعية في المقال (main.StoryRejected) تُسجَّل القصة
      مرفوضة (output/<hash>/result.json) فلا تُختار ثانية، ثم تُختار قصة أخرى فورًا.
    * إن فشل استخراج القصة أو معالجتها (غير تعطل الذكاء الاصطناعي) تُستبدل بأخرى.
    * إن تعطلت كل مزودات الكتابة تُنتظر دقيقة وتُعاد المحاولة على القصة نفسها.
- MAX_STORY_ATTEMPTS (الافتراضي 5): أقصى عدد قصص تُجرَّب قبل التوقف بخطأ واضح.

التشغيل: python run_factory.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import grok
import main as pipeline
import picker


WRITER_FAILURE_PREFIX = "فشلت كتابة المقال"   # تعطل مزودات الكتابة، لا عيب في القصة
WRITER_RETRY_WAIT = 60
MAX_WRITER_RETRIES = 3


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def mark_rejected(url: str, reason: str) -> None:
    """يسجّل القصة مرفوضة حتى لا تُختار ثانية (يظهر لها result.json في مجلدها)."""
    folder = pipeline.output_folder_for(url)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(
        json.dumps({
            "rejected": True,
            "source_url": url,
            "reason": reason,
            "rejected_at_utc": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def switch_story() -> None:
    """يختار قصة أخرى ويحدّث متغيرات التشغيل."""
    item, entry = picker.pick_story()
    os.environ["ARTICLE_URL"] = item["link"]
    os.environ["CONTENT_CATEGORY"] = entry.get("category", "")
    print(f"القصة الجديدة: {item['title']}\nالرابط: {item['link']}")


def run() -> int:
    auto = _flag("AUTO_PICK")
    if auto:
        os.environ["REJECT_ON_ISSUES"] = "1"

    try:
        max_stories = max(1, int(os.getenv("MAX_STORY_ATTEMPTS", "5") or 5))
    except ValueError:
        max_stories = 5

    writer_retries = 0
    story = 1
    while story <= max_stories:
        url = os.getenv("ARTICLE_URL", "").strip()
        print(f"\n===== القصة {story} من {max_stories}: {url} =====")
        try:
            pipeline.main()
            return 0
        except pipeline.StoryRejected as exc:
            if not auto:
                raise
            print(f"::warning title=قصة مرفوضة::{exc}")
            mark_rejected(url, str(exc))
        except pipeline.ProjectError as exc:
            if not auto:
                raise
            if str(exc).startswith(WRITER_FAILURE_PREFIX):
                writer_retries += 1
                if writer_retries >= MAX_WRITER_RETRIES:
                    raise
                print(f"::warning title=تعطل الكتابة::{exc}\n"
                      f"انتظار {WRITER_RETRY_WAIT} ثانية ثم إعادة المحاولة...")
                time.sleep(WRITER_RETRY_WAIT)
                continue                      # القصة نفسها، لا تُحسب محاولة جديدة
            print(f"::warning title=قصة تعذرت معالجتها::{exc}")
            mark_rejected(url, str(exc))

        story += 1
        if story > max_stories:
            break
        try:
            switch_story()
        except (pipeline.ProjectError, grok.GrokError) as exc:
            print(f"::error title=تعذر اختيار قصة أخرى::{exc}", file=sys.stderr)
            return 1

    print(f"::error title=لم تُنتج قصة صالحة::رُفضت {max_stories} قصص متتالية.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(run())
    except (pipeline.ProjectError, grok.GrokError) as exc:
        print(f"\nخطأ: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nتم إيقاف التشغيل.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nخطأ غير متوقع: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
