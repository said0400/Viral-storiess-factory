"""
كاتب القصص: يولّد قصة أصلية كاملة بـ Gemini (مفتاح الكتابة GEMINI_WRITER_API_KEY).

- التصنيف يفرضه الكود (دوران بين أربعة تصنيفات)، لا النموذج.
- منع التكرار: يُمرَّر للنموذج سجل آخر القصص (الفكرة والصراع والمفاجأة والنهاية)،
  ثم يفحص الكود تشابه الفكرة والمفاجأة مع السجل ويعيد المحاولة مرة عند التشابه.
- الطول: إن جاءت القصة أقصر من STORY_MIN_WORDS (الافتراضي 1300) تُرسل للنموذج ليوسّعها
  (حتى مرتين: نفس الحبكة والمفاجأة والنهاية بمشاهد وحوارات أكثر)، فإن لم تكفِ تُكتب قصة
  جديدة (حتى STORY_MAX_ATTEMPTS محاولات كاملة، الافتراضي 3). الفشل بعد استنفادها فقط.
- لا احتياط من Groq: إن فشل Gemini لا تُنشر قصة ضعيفة، وتعيد المحاولة دورة cron التالية.
- إعدادات أمان Gemini مخففة (BLOCK_ONLY_HIGH) كي لا تُحجب القصص المظلمة بلا داعٍ؛
  وتُغيَّر من STORY_SAFETY_THRESHOLD (مثل BLOCK_NONE).
- البرومبت الإبداعي في prompts/story_style.txt (يُعدَّل دون لمس الكود).
"""
from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path
from typing import Any

import requests

import gemini
import grok
import key_pool
import story_categories
from grok import GrokError
from story_text import AUTHOR_MARK


STYLE_FILE = Path(__file__).resolve().parent / "prompts" / "story_style.txt"
LINK_PLACEHOLDER = "{LINK}"

MAX_HISTORY = 40
MAX_OUTPUT_TOKENS = 24000
REQUEST_TIMEOUT = 300
ATTEMPTS = 5                    # لكل نموذج: يتحمل ازدحام Gemini (503) وحدود المعدل (429)
PARSE_RETRIES = 2               # إعادة التوليد إن لم يكن الرد JSON صالحًا
SIMILARITY_LIMIT = 0.45         # تشابه الفكرة أو المفاجأة (Jaccard) الذي يعدّ تكرارًا
DEFAULT_MIN_WORDS = 1300
DEFAULT_MAX_ATTEMPTS = 4        # محاولات كتابة كاملة للقصة الواحدة
MAX_EXPANSIONS = 3              # توسيعات لقصة قصيرة قبل الاستسلام لقصة جديدة

LOG_KEYS = ("idea", "conflict", "characters", "setting", "opening", "twist", "ending")

SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)

# قيود إبداعية عشوائية تكسر نمطية النموذج بين قصة وأخرى
SETTINGS = (
    "قرية جبلية معزولة في الشتاء", "مدينة ساحلية قديمة في موسم المطر",
    "مستشفى قديم شبه مهجور", "فندق فاخر في موسم السياحة", "حي شعبي مزدحم في الصيف",
    "قطار ليلي طويل", "مزرعة نائية وسط الحقول", "عمارة سكنية يعرف سكانها بعضهم",
    "ميناء تجاري مزدحم", "مدرسة داخلية في الريف", "سوق قديم وأزقة ضيقة",
    "قصر عائلي عريق", "شركة كبرى في برج زجاجي", "صحراء وقافلة أو مخيم",
    "جزيرة صغيرة معزولة", "محطة وقود على طريق سريع ليلًا", "مكتبة أو أرشيف قديم",
    "مقهى عتيق في وسط المدينة", "منجم أو مصنع مهجور", "قاعة مزاد أو معرض فني",
    "حفل زفاف كبير في قاعة أفراح", "بيت ريفي بعد وفاة صاحبه", "مدينة ملاهٍ مغلقة",
    "عيادة أو مختبر خاص", "حي راقٍ تحيط به أسوار وحراس",
)
VOICES = (
    "راوٍ بضمير المتكلم يخفي شيئًا عن القارئ",
    "راوٍ عليم بصوت هادئ يعرف أكثر من الشخصيات",
    "سرد بالتناوب بين شخصيتين تريان الحدث نفسه بشكل مختلف",
    "سرد يبدأ من النهاية ثم يعود إلى البداية ليشرح كيف وصلنا إليها",
    "سرد يقدّم الحدث بعين الشاهد العادي لا البطل",
    "سرد من منظور الشخصية التي تبدو شريرة في ظاهر الأمر",
    "سرد تتخلله وثائق أو رسائل أو تسجيلات تكشف الحقيقة تدريجيًا",
    "سرد خطي سريع بجمل قصيرة وإيقاع متصاعد",
)

HARD_FORMAT = """
أعد JSON صالحًا فقط دون Markdown خارجه، بهذه المفاتيح بالضبط:
{
  "title": "عنوان القصة (حتى 65 حرفًا، جذاب، بلا كشف للمفاجأة)",
  "article": "نص القصة الكامل بصيغة Markdown (من 1500 إلى 2200 كلمة) بلا العنوان الرئيسي",
  "facebook_post": "منشور فيسبوك كاملًا: سطر العنوان بالرمز 🚨 أو ⚠️ ثم المقدمة ثم نقطة القطع ثم سطر التوجيه غير المباشر إلى أول تعليق",
  "first_comment": "التعليق الأول: تتمة المنشور ثم دعوة لقراءة القصة ثم السطر الأخير {LINK} وحدها",
  "hashtags": ["#وسم1", "#وسم2"],
  "image_title": "عنوان قصير للصورة من 3 إلى 8 كلمات",
  "image_highlight": "كلمة أو كلمتان منسوختان حرفيًا من image_title",
  "image_prompt": "English only: one or two sentences describing a dark cinematic still for this story (setting, mood, lighting, one strong visual subject). No text, no letters, no logos.",
  "seo": {
    "focus_keyword": "الكلمة المفتاحية (2 إلى 4 كلمات)",
    "meta_title": "عنوان SEO حتى 60 حرفًا",
    "meta_description": "وصف ميتا بين 140 و160 حرفًا",
    "slug": "english-short-slug",
    "keywords": ["كلمة1", "كلمة2"]
  },
  "story_log": {
    "idea": "الفكرة الأساسية في جملة أو جملتين",
    "conflict": "الصراع الرئيسي",
    "characters": "الشخصيات الرئيسية وصفاتها باختصار",
    "setting": "المكان والزمان",
    "opening": "نوع الافتتاحية (مشهد، سؤال، حوار...)",
    "twist": "المفاجأة الرئيسية",
    "ending": "نوع النهاية"
  }
}
- "hashtags": من 5 إلى 10 وسوم مرتبطة بالقصة.
- "seo.keywords": من 5 إلى 10 كلمات.
- "story_log" وصف دقيق وصادق لما كتبته فعلًا؛ يُحفظ ويُستعمل لمنع تكرار الأفكار لاحقًا.
- كل المفاتيح مطلوبة، والقصة كاملة داخل "article".
""".strip()


SEO_RULES = """
وصف SEO خاص بهذه القصة وحدها (يُقدَّم عند التعارض مع ما سبق):
- "meta_description": بين 140 و160 حرفًا، تأتي الكلمة المفتاحية في أول 70 حرفًا منه، ويحمل
  عنصرًا محددًا من هذه القصة (المكان أو الشيء الغامض أو اللغز الذي تدور حوله) ويترك فجوة فضول
  دون أن يكشف المفاجأة أو النهاية أبدًا. ممنوع الوصف العام الذي يصلح لأي قصة أخرى، وممنوعة
  القوالب مثل: اكتشف، تعرف على، قصة مثيرة، قصة غامضة، في هذه القصة.
- "focus_keyword": عبارة من 2 إلى 4 كلمات خاصة بهذه القصة (لا «قصة رعب» ولا «قصة غامضة»).
- "meta_title": يختلف عن "title" ولا يكرره حرفيًا، حتى 60 حرفًا.
- "keywords": من 6 إلى 10 كلمات وعبارات من عناصر هذه القصة (المكان، الشيء، اللغز، الأجواء)
  مع قليل من الكلمات العامة، لا كلمات عامة وحدها.
- "slug": إنجليزي من 3 إلى 6 كلمات يعبّر عن موضوع هذه القصة تحديدًا.
- مفتاح إضافي داخل كائن "seo" نفسه: "video_summary": من 2 إلى 3 جمل (من 220 إلى 400 حرف)
  تعرّف المستمع بالقصة في فيديو يوتيوب: الأجواء والمكان والسؤال الذي تدور حوله، بلا كشف
  للمفاجأة ولا النهاية، وبصياغة تختلف عن "meta_description".
""".strip()


class StoryError(Exception):
    """خطأ في كتابة القصة أو في مخرجات النموذج."""


# ---------------------------------------------------------------------------
# أدوات نصية
# ---------------------------------------------------------------------------

_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")


def _plain(text: str) -> str:
    return _DIACRITICS.sub("", text or "")


def _clean_line(text: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = text.replace("«", "").replace("»", "").replace('"', "")
    return text[:limit].strip()


def _tokens(text: str) -> set[str]:
    words = re.findall(r"\w+", _plain(text).lower(), flags=re.UNICODE)
    return {w for w in words if len(w) > 2}


def _jaccard(a: str, b: str) -> float:
    sa, sb = _tokens(a), _tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _word_key(word: str) -> str:
    return re.sub(r"[^\w]", "", word, flags=re.UNICODE)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(os.getenv(name, "").strip() or default)))
    except ValueError:
        return default


def _trim_description(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ،,.:؛") + "…"


def _fallback_description(article: str, limit: int = 160) -> str:
    """وصف ميتا من افتتاحية هذه القصة نفسها، يُستعمل فقط إن جاء وصف النموذج فارغًا أو قصيرًا."""
    for block in re.split(r"\n{2,}", article or ""):
        block = block.strip()
        if not block or block.startswith("#"):
            continue
        block = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", block)
        block = block.replace("**", "").replace("*", "")
        sentences = re.split(r"(?<=[.!؟?…])\s+", re.sub(r"\s+", " ", block))
        text = ""
        for sentence in sentences:
            if len(text) + len(sentence) > limit and len(text) >= 110:
                break
            text = f"{text} {sentence}".strip()
        return _trim_description(text, limit)
    return ""


def _slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug[:70] or "story"


# ---------------------------------------------------------------------------
# السجل ومنع التكرار
# ---------------------------------------------------------------------------

def load_history(
    output_root: Path = Path("output"), limit: int = MAX_HISTORY
) -> list[dict[str, str]]:
    """سجل آخر القصص (الأحدث أولًا): العنوان والتصنيف وعناصر story_log."""
    rows = story_categories.load_story_results(output_root)[-limit:]
    history = []
    for row in reversed(rows):
        log = row.get("story_log") if isinstance(row.get("story_log"), dict) else {}
        entry = {"title": str(row.get("title") or ""),
                 "category": str(row.get("category") or "")}
        for key in LOG_KEYS:
            entry[key] = str(log.get(key) or "")
        history.append(entry)
    return history


def find_clash(
    story: dict[str, Any], history: list[dict[str, str]]
) -> tuple[dict[str, str], float] | None:
    """أول قصة سابقة تتشابه معها هذه القصة (عنوان مطابق، أو فكرة/مفاجأة متقاربة)."""
    title = _plain(story.get("title", "")).strip()
    log = story.get("story_log") or {}
    for old in history:
        if title and title == _plain(old.get("title", "")).strip():
            return old, 1.0
        score = max(
            _jaccard(log.get("idea", ""), old.get("idea", "")),
            _jaccard(log.get("twist", ""), old.get("twist", "")),
        )
        if score >= SIMILARITY_LIMIT:
            return old, score
    return None


# ---------------------------------------------------------------------------
# بناء التعليمات
# ---------------------------------------------------------------------------

def _style_rules() -> str:
    for path in (STYLE_FILE, Path("prompts") / "story_style.txt"):
        try:
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except OSError:
            continue
    raise StoryError("ملف التعليمات prompts/story_style.txt غير موجود أو فارغ.")


def _history_block(history: list[dict[str, str]]) -> str:
    if not history:
        return ("لا توجد قصص سابقة بعد. لا تدّعِ أنك تحققت من قصص سابقة، "
                "لكن ابتكر فكرة مستقلة وغير نمطية.")
    lines = []
    for old in history:
        lines.append(
            f"- «{old['title']}» [{old['category']}] فكرة: {old['idea'][:160]} | "
            f"صراع: {old['conflict'][:100]} | مكان: {old['setting'][:80]} | "
            f"افتتاحية: {old['opening'][:60]} | مفاجأة: {old['twist'][:160]} | "
            f"نهاية: {old['ending'][:100]}"
        )
    return (
        "سجل القصص السابقة المنشورة (الأحدث أولًا). لا تكرر أي فكرة أو صراع أو مكان "
        "أو افتتاحية أو مفاجأة أو نهاية منها، ولا تدوّر قصة منها بتفاصيل جديدة:\n"
        + "\n".join(lines)
    )


def build_system(category: str, history: list[dict[str, str]]) -> str:
    info = story_categories.category_info(category)
    return "\n\n".join([
        _style_rules(),
        f"تصنيف هذه القصة (ثابت، لا تغيّره ولا تخلطه بغيره): {category}\n"
        f"نطاقه: {info['description']}",
        _history_block(history),
        HARD_FORMAT,
        SEO_RULES,
    ])


def build_user(rng: random.Random | None = None, extra: str = "") -> str:
    rng = rng or random.Random()
    setting, voice = rng.choice(SETTINGS), rng.choice(VOICES)
    return (
        "اكتب الآن قصة جديدة كاملة وفق التعليمات، بفكرة مختلفة جذريًا عن السجل.\n"
        f"قيد إبداعي لتنويع هذه القصة (التزم به ما دام لا يضعف القصة): "
        f"البيئة: {setting}. زاوية السرد: {voice}.\n"
        "أعد JSON فقط." + extra
    )


# ---------------------------------------------------------------------------
# الاتصال بـ Gemini
# ---------------------------------------------------------------------------

def _models() -> list[str]:
    custom = os.getenv("GEMINI_WRITER_MODEL", "").strip()
    return [custom] if custom else list(gemini.DEFAULT_MODELS)


def _call_gemini_once(system: str, user: str, temperature: float = 0.95,
                      key: str = "") -> tuple[str, str]:
    """يعيد (النص، اسم النموذج) بمفتاح واحد. يرفع StoryError إن فشلت كل النماذج."""
    key = key or (key_pool.load_keys("GEMINI_WRITER_API_KEY") or [""])[0]
    if not key:
        raise StoryError("المفتاح GEMINI_WRITER_API_KEY غير موجود في GitHub Secrets.")

    threshold = os.getenv("STORY_SAFETY_THRESHOLD", "").strip() or "BLOCK_ONLY_HIGH"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
        "safetySettings": [
            {"category": c, "threshold": threshold} for c in SAFETY_CATEGORIES
        ],
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    last = "فشل الاتصال بواجهة Gemini."

    for model in _models():
        url = f"{gemini.GEMINI_BASE_URL}/{model}:generateContent"
        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = requests.post(url, headers=headers, json=body,
                                         timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                last = f"تعذر الاتصال بـ Gemini: {exc}"
            else:
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        last = "أعاد Gemini استجابة ليست JSON صالحًا."
                        break
                    try:
                        return gemini._extract_text(data), model
                    except gemini.GeminiError as exc:
                        last = f"{model}: {exc}"      # حجب أو اقتطاع: جرّب النموذج التالي
                        break
                last = (f"خطأ Gemini HTTP {response.status_code} ({model}): "
                        f"{response.text[:300]}")
                if response.status_code == 429 and "PerDay" in response.text:
                    break                             # حصة اليوم انتهت لهذا النموذج
                if response.status_code in (400, 404):
                    break                             # طلب مرفوض أو نموذج غير موجود
                if response.status_code not in gemini.RETRY_STATUS:
                    raise StoryError(last)
            if attempt < ATTEMPTS:
                wait = min(10 * 2 ** (attempt - 1), 90)
                print(f"تحذير: محاولة Gemini {attempt} فشلت ({last[:140]}) "
                      f"إعادة المحاولة بعد {wait} ثانية.")
                time.sleep(wait)
    raise StoryError(last)


def _call_gemini(system: str, user: str, temperature: float = 0.95) -> tuple[str, str]:
    """
    يعيد (النص، اسم النموذج). يدعم عدة حسابات (GEMINI_WRITER_API_KEY بمفاتيح مفصولة بفاصلة):
    يُستعمل الأول حتى يتوقف (حصة يومية أو مفتاح غير صالح أو فشل متكرر) فيُنتقل إلى التالي.
    """
    keys = key_pool.load_keys("GEMINI_WRITER_API_KEY")
    if not keys:
        raise StoryError("المفتاح GEMINI_WRITER_API_KEY غير موجود في GitHub Secrets.")
    live = key_pool.usable(keys)
    last: StoryError | None = None
    for number, key in enumerate(live, start=1):
        try:
            return _call_gemini_once(system, user, temperature, key)
        except StoryError as exc:
            last = exc
            if key_pool.is_permanent(str(exc)):
                key_pool.mark_dead(key, str(exc))
            if number < len(live):
                print(f"::warning title=حساب Gemini::توقف الحساب {key_pool.mask(key)} "
                      f"({str(exc)[:110]})؛ التبديل إلى الحساب التالي "
                      f"({number + 1} من {len(live)}).")
    raise last or StoryError(
        f"كل حسابات Gemini للكتابة ({len(keys)}) متوقفة: انتهت الحصة اليومية أو المفتاح غير صالح.")


# ---------------------------------------------------------------------------
# التطبيع والتحقق
# ---------------------------------------------------------------------------

def _normalize_comment(value: Any) -> str:
    """التعليق الأول: بلا روابط حقيقية، و{LINK} مرة واحدة في آخره."""
    comment = _URL.sub("", str(value or "")).strip()
    if len(comment) < 40:
        raise StoryError("حقل first_comment مفقود أو قصير جدًا.")
    parts = comment.split(LINK_PLACEHOLDER)
    if len(parts) == 1:
        comment = comment.rstrip() + f"\n\nالقصة كاملة هنا: {LINK_PLACEHOLDER}"
    elif len(parts) > 2:
        comment = parts[0] + LINK_PLACEHOLDER + "".join(parts[1:])
    return comment


def normalize(raw: dict[str, Any], category: str) -> dict[str, Any]:
    title = _clean_line(raw.get("title"), 120)
    article = _URL.sub("", str(raw.get("article") or "")).strip()
    post = _URL.sub("", str(raw.get("facebook_post") or "")).strip()

    if not title:
        raise StoryError("حقل title مفقود أو فارغ.")
    if word_count(article) < 200:
        raise StoryError("نص القصة مفقود أو قصير جدًا.")
    if not post:
        raise StoryError("حقل facebook_post مفقود أو فارغ.")

    article = re.sub(r"^\s*#\s+[^\n]*\n+", "", article, count=1)   # لا نكرر العنوان
    first_comment = _normalize_comment(raw.get("first_comment"))

    tags: list[str] = []
    for item in raw.get("hashtags") or []:
        tag = grok._normalize_hashtag(item)
        if tag and tag not in tags:
            tags.append(tag)

    image_title = _EMOJI.sub("", _clean_line(raw.get("image_title"), 90)).strip() or title
    image_words = {_word_key(w) for w in image_title.split()}
    highlight = [
        w for w in str(raw.get("image_highlight") or "").split()
        if _word_key(w) and _word_key(w) in image_words
    ][:3]

    seo_raw = raw.get("seo") if isinstance(raw.get("seo"), dict) else {}
    meta_description = _clean_line(seo_raw.get("meta_description"), 400)
    if len(meta_description) < 80:
        meta_description = _fallback_description(article) or meta_description
    seo = {
        "focus_keyword": _clean_line(seo_raw.get("focus_keyword"), 60),
        "meta_title": _clean_line(seo_raw.get("meta_title") or title, 70),
        "meta_description": _trim_description(meta_description),
        "slug": _slug(seo_raw.get("slug")),
        "keywords": [
            _clean_line(k, 40) for k in (seo_raw.get("keywords") or [])
            if _clean_line(k, 40)
        ][:10],
        "video_summary": _clean_line(seo_raw.get("video_summary"), 600),
    }

    log_raw = raw.get("story_log") if isinstance(raw.get("story_log"), dict) else {}
    story_log = {k: _clean_line(log_raw.get(k), 300) for k in LOG_KEYS}
    if not story_log["idea"]:
        story_log["idea"] = title

    image_prompt = re.sub(r"\s+", " ", str(raw.get("image_prompt") or "")).strip()[:900]

    return {
        "title": title,
        "article": article,
        "facebook_post": post,
        "first_comment": first_comment,
        "hashtags": tags[:10],
        "image_title": image_title,
        "image_highlight": " ".join(highlight),
        "image_prompt": image_prompt,
        "seo": seo,
        "story_log": story_log,
        "category": category,
    }


DEFAULT_BLOG_NAME = "المسيطرون"      # يُغيَّر من المتغير BLOG_NAME دون لمس الكود

# صيغ متنوعة لسطر مصدر القصة (تُختار بثبات من نص القصة).
AUTHOR_TEMPLATES = (
    "**" + AUTHOR_MARK + " مصدر القصة:** كتبها فريق {name} خصيصًا لقرّاء المدونة، وهي من وحي الخيال.",
    "**" + AUTHOR_MARK + " هذه القصة من تأليف فريق {name}.** أي تشابه مع أشخاص أو وقائع حقيقية غير مقصود.",
    "**" + AUTHOR_MARK + " كتبها لكم فريق {name}**، قصة أصلية من وحي الخيال، ونحب أن نعرف رأيكم فيها.",
)


def build_author_line(article: str = "") -> str:
    """سطر مصدر القصة: أنها كُتبت بقلم فريق المدونة (BLOG_NAME)."""
    name = re.sub(r"[\[\]()*]", "", os.getenv("BLOG_NAME", "").strip()) or DEFAULT_BLOG_NAME
    index = sum((article or "")[:300].encode("utf-8")) % len(AUTHOR_TEMPLATES)
    return AUTHOR_TEMPLATES[index].format(name=name)


def append_related(article: str, related: list[dict[str, str]], limit: int = 2) -> tuple[str, list[dict[str, str]]]:
    """يلحق سطور «اقرأ أيضًا» بقصص سابقة منشورة (روابط حقيقية) في آخر القصة."""
    used: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in related:
        url = str(item.get("url") or "").strip()
        title = re.sub(r"[\[\]()]", "", str(item.get("title") or "")).strip()
        if not url or not title or url in seen:
            continue
        seen.add(url)
        used.append({"title": title, "url": url})
        if len(used) >= limit:
            break
    author = "" if AUTHOR_MARK in article else build_author_line(article)
    if not used:
        return (article.rstrip() + ("\n\n" + author if author else "")), used
    lines = [f"**اقرأ أيضًا:** [{item['title']}]({item['url']})" for item in used]
    head = article.rstrip() + ("\n\n" + author if author else "")
    return head + "\n\n" + "\n\n".join(lines), used


# ---------------------------------------------------------------------------
# الكتابة
# ---------------------------------------------------------------------------

def _write_once(system: str, user: str, category: str) -> tuple[dict[str, Any], str]:
    """رد صالح من النموذج؛ يعيد التوليد إن لم يكن JSON أو نقصت حقوله."""
    last: Exception | None = None
    for number in range(1, PARSE_RETRIES + 2):
        text, model = _call_gemini(system, user)
        try:
            return normalize(grok._extract_json(text), category), model
        except (GrokError, StoryError, ValueError, KeyError, TypeError) as exc:
            last = exc
            print(f"تحذير: رد Gemini غير صالح (المحاولة {number}): {exc}")
    raise StoryError(f"تعذر الحصول على قصة صالحة: {last}")


EXPAND_SYSTEM = """
أنت روائي ومحرر محترف. تتلقى قصة جاءت أقصر من المطلوب، وعليك توسيعها إلى قصة كاملة
من {LOW} إلى {HIGH} كلمة.

القواعد:
- احتفظ بالأحداث الرئيسية وترتيبها وبالشخصيات والمفاجأة الرئيسية والنهاية كما هي، ولا تغيّر الفكرة.
- وسّع بمشاهد وحوارات وتفاصيل حسية (أصوات وإضاءة وروائح وطقس) وصراع داخلي وتصاعد في التوتر
  وإشارات صغيرة تمهّد للمفاجأة، وبإبطاء اللحظات المهمة. لا حشو ولا تكرار ولا تلخيص لما سبق.
- اكتب القصة كاملة من أولها إلى آخرها، لا الإضافات وحدها.
- الفقرات قصيرة، ويمكن استعمال عناوين فرعية بصيغة ## (حتى 6)، وبلا العنوان الرئيسي ولا روابط.
- الفصحى المبسطة نفسها وأسلوب القصة نفسه.
- أعد JSON صالحًا فقط: {"article": "القصة الموسّعة كاملة بصيغة Markdown"}
""".strip()


def _expand_article(article: str, category: str, min_words: int) -> str | None:
    """
    يطلب من النموذج توسيع قصة قصيرة مع بقاء حبكتها. يعيد النص الموسّع، أو None إن لم يزد
    طوله بوضوح (فلا يُستبدل النص الأصلي بما هو أسوأ).
    """
    current = word_count(article)
    low = max(min_words + 200, 1500)
    system = EXPAND_SYSTEM.replace("{LOW}", str(low)).replace("{HIGH}", str(low + 600))
    user = (f"التصنيف: {category}\nالطول الحالي: {current} كلمة؛ المطلوب من {low} إلى "
            f"{low + 600} كلمة.\n\nالقصة الحالية:\n<story>\n{article}\n</story>\n\n"
            "أعد JSON فقط.")
    text, _ = _call_gemini(system, user, temperature=0.8)
    raw = grok._extract_json(text)
    bigger = _URL.sub("", str(raw.get("article") or "")).strip()
    bigger = re.sub(r"^\s*#\s+[^\n]*\n+", "", bigger, count=1)
    if word_count(bigger) < current * 1.1:
        return None
    return bigger


def write_story(
    category: str,
    history: list[dict[str, str]],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """
    يكتب قصة كاملة للتصنيف المحدد، ولا يفشل لمجرد أن أول محاولة جاءت قصيرة:
      - قصة قصيرة: توسَّع (حتى MAX_EXPANSIONS مرة) بدل رفضها.
      - قصة متشابهة مع السجل، أو ما زالت قصيرة بعد التوسيع: تُكتب قصة جديدة،
        حتى STORY_MAX_ATTEMPTS محاولات (الافتراضي 3).
    وتُختار الأفضل بين المحاولات. يفشل فقط إن بقيت أقصر من 75% من الحد الأدنى بعد كل ذلك.
    """
    min_words = _env_int("STORY_MIN_WORDS", DEFAULT_MIN_WORDS, 300, 3000)
    max_attempts = _env_int("STORY_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS, 1, 6)
    system = build_system(category, history)
    extra = ""
    best: tuple[tuple, dict[str, Any]] | None = None

    for attempt in range(1, max_attempts + 1):
        data, model = _write_once(system, build_user(rng, extra), category)

        expansions = 0
        while word_count(data["article"]) < min_words and expansions < MAX_EXPANSIONS:
            expansions += 1
            before = word_count(data["article"])
            try:
                bigger = _expand_article(data["article"], category, min_words)
            except (StoryError, GrokError, ValueError, KeyError, TypeError) as exc:
                print(f"تحذير: تعذر توسيع القصة القصيرة ({exc}).")
                break
            if bigger is None:
                print(f"تحذير: لم يزد التوسيع الطول بوضوح ({before} كلمة).")
                continue
            data["article"] = bigger
            print(f"تم توسيع القصة القصيرة: {before} ← {word_count(bigger)} كلمة.")

        words = word_count(data["article"])
        problems: list[str] = []
        if words < min_words:
            problems.append(f"القصة قصيرة ({words} كلمة)؛ المطلوب من 1500 إلى 2200 كلمة")
        clash = find_clash(data, history)
        if clash:
            old, score = clash
            problems.append(
                f"تشابه ({score:.0%}) مع القصة السابقة «{old['title']}» "
                f"(فكرة: {old['idea'][:120]} | مفاجأة: {old['twist'][:120]})")

        data["writer"] = {"provider": "gemini", "model": model, "attempts": attempt,
                          "expansions": expansions}
        data["stats"] = {"words": words,
                         "reading_minutes": max(1, round(words / 180)),
                         "min_words": min_words}
        data["problems"] = problems

        rank = (not problems, -len(problems), words)
        if best is None or rank > best[0]:
            best = (rank, data)
        if not problems:
            break

        print("تنبيه: " + " | ".join(problems) + "؛ إعادة كتابة.")
        extra = ("\n\nالنسخة السابقة لم تقبل للأسباب التالية، فاكتب قصة جديدة مختلفة "
                 "كليًا، طويلة كاملة من 1700 إلى 2200 كلمة، ولا تكرر الأخطاء: "
                 + " | ".join(problems))

    assert best is not None
    final = best[1]
    if final["stats"]["words"] < int(min_words * 0.75):
        raise StoryError(
            f"القصة أقصر من المقبول بعد {max_attempts} محاولات وتوسيعها "
            f"({final['stats']['words']} كلمة، الحد الأدنى {int(min_words * 0.75)}). "
            "تُعاد المحاولة لاحقًا.")
    return final
