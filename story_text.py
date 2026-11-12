"""
نصوص القصة المشتركة بين الصوت (story_video.py) ولوحة المشاهد (story_scenes.py):
تنظيف النص للقراءة بصوت عالٍ، وتقسيمه إلى فقرات ومقاطع.

يجب أن يبقى تقسيم الفقرات واحدًا في الوحدتين، لأن لوحة المشاهد تشير إلى الفقرات برقمها
ويحوّل الفيديو الرقم إلى توقيت في الصوت.

AUTHOR_MARK: رمز يبدأ به سطر «مصدر القصة» الذي يلحقه story_writer.py بآخر القصة
(وبه يعرف story_writer ألا يكرر السطر، ويعرف story_video أين ينتهي متن القصة قبل سطور
الختام). وهذا السطر ليس من القصة، فلا يُقرأ بصوت الراوي ولا يُحسب فقرة، كي تبقى أرقام
الفقرات في لوحة المشاهد مطابقة لتوقيتات الصوت.
"""
from __future__ import annotations

import re

RELATED_MARK = "اقرأ أيضا"   # بلا تشكيل، للمقارنة
VIDEO_MARK = "🎧"
AUTHOR_MARK = "✍️"           # يبدأ به سطر «مصدر القصة» (انظر story_writer.AUTHOR_TEMPLATES)

_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def plain(text: str) -> str:
    return _DIACRITICS.sub("", text or "")


def clean_story_text(markdown: str) -> str:
    """
    نص صالح للقراءة بصوت عالٍ: بلا Markdown ولا روابط، وبلا سطور «اقرأ أيضًا»
    والروابط الصوتية وسطر «مصدر القصة».
    """
    lines: list[str] = []
    for raw in (markdown or "").splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith("# "):                       # العنوان الرئيسي (يُقرأ منفصلًا)
            continue
        if RELATED_MARK in plain(line) or VIDEO_MARK in line or AUTHOR_MARK in line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = _URL.sub("", line).replace("**", "").replace("*", "")
        lines.append(line.strip())
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def story_paragraphs(markdown: str) -> list[str]:
    """فقرات القصة كما تُقرأ (العناوين الفرعية فقرات مستقلة). الفهرس هنا هو رقم الفقرة في المشاهد."""
    return [p.strip() for p in re.split(r"\n{2,}", clean_story_text(markdown)) if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!؟?…؛])\s+", paragraph)
    return [p for p in parts if p.strip()]


def split_chunks(text: str, max_words: int = 420) -> list[str]:
    """مقاطع بحدود الفقرات (وبحدود الجمل للفقرة الطويلة جدًا)، كل منها حتى max_words كلمة."""
    units: list[str] = []
    for para in re.split(r"\n{2,}", text.strip()):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) <= max_words:
            units.append(para)
        else:
            buf: list[str] = []
            for sentence in _split_sentences(para):
                if buf and len(" ".join(buf + [sentence]).split()) > max_words:
                    units.append(" ".join(buf))
                    buf = []
                buf.append(sentence)
            if buf:
                units.append(" ".join(buf))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and len("\n\n".join(current + [unit]).split()) > max_words:
            chunks.append("\n\n".join(current))
            current = []
        current.append(unit)
    if current:
        chunks.append("\n\n".join(current))
    return chunks
