"""
تصنيفات القصص الأربعة وتناوبها: تصنيف 1 ثم 2 ثم 3 ثم 4 ثم يعود إلى 1 وهكذا.

القسم: «قصص الغموض والإثارة» وتحته أربعة تصنيفات فرعية. يحمل كل مقال وسمين في Blogger:
القسم + التصنيف الفرعي (labels_for)، فتظهر القصة في صفحة القسم وصفحة التصنيف.

التصنيف التالي يُحسب من آخر قصة وُلّدت (output/*/result.json التي فيها kind = "story")،
فلا يعتمد على النموذج، ولا يتكرر تصنيف مرتين متتاليتين. والأسماء القديمة (legacy) تُعدّ
مرادفات للجديدة كي لا ينقطع التناوب بسبب قصص وُلّدت قبل تغيير الأسماء.

لا تضع هنا شيئًا آخر: هذا المصدر الموحد لأسماء التصنيفات.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# اسم القسم (الوسم الأب في Blogger)
PARENT_LABEL = "قصص الغموض والإثارة"

STORIES: list[dict[str, Any]] = [
    {
        "name": "قصص الجريمة",
        "legacy": ["قصص الجرائم والتحقيقات البوليسية"],
        "description": (
            "جرائم القتل، الاختفاء الغامض، التحقيقات البوليسية، الأدلة، الأسرار، الخيانة، "
            "الانتقام، الألغاز الجنائية والجرائم المعقدة. خيالية بالكامل أو مستوحاة من أجواء "
            "قضايا واقعية (دون ذكر وقائع أو أسماء حقيقية)."
        ),
        "accent": (205, 32, 44),
        "safe_image": (
            "A rain-soaked city street at night, a lone streetlamp, wet asphalt reflecting "
            "red and blue light, empty and silent, noir atmosphere, cinematic."
        ),
    },
    {
        "name": "قصص الرعب",
        "legacy": ["قصص الرعب والماورائيات والغموض"],
        "description": (
            "الرعب والماورائيات والغموض: الأماكن المهجورة، الظواهر الغريبة، الأسرار المرعبة، "
            "الأحداث الخارقة، البيوت المسكونة، الكائنات الغامضة، والأحداث التي تتحدى التوقعات."
        ),
        "accent": (96, 54, 170),
        "safe_image": (
            "An old abandoned house at dusk surrounded by thick fog, a single lit window, "
            "bare trees, eerie cold blue atmosphere, cinematic."
        ),
    },
    {
        "name": "قصص اجتماعية",
        "legacy": ["القصص الاجتماعية الواقعية والدراما"],
        "description": (
            "قصص اجتماعية واقعية ودراما إنسانية: الخيانة الزوجية، العلاقات العائلية، الظلم، "
            "الفقر، الطمع، الانتقام، التضحية، الأسرار العائلية، الصراعات الإنسانية والمواقف المؤثرة."
        ),
        "accent": (214, 132, 36),
        "safe_image": (
            "A quiet family kitchen table at night under a dim lamp, an empty chair, warm "
            "amber light and deep shadows, melancholic mood, cinematic."
        ),
    },
    {
        "name": "قصص الدهاء",
        "legacy": ["قصص الذكاء والمكر والخدع"],
        "description": (
            "قصص الذكاء والمكر والخدع: عمليات النصب، الخدع الذكية، السرقات المعقدة، التلاعب "
            "النفسي، المؤامرات، الأسرار المالية، الخطط الماكرة والخداع غير المتوقع."
        ),
        "accent": (30, 150, 130),
        "safe_image": (
            "A dim luxurious hotel lobby at night, a leather briefcase on a marble table, "
            "golden light and deep shadows, tense elegant atmosphere, cinematic."
        ),
    },
]

STORY_CATEGORIES: list[str] = [s["name"] for s in STORIES]

_CANONICAL: dict[str, str] = {}
for _item in STORIES:
    _CANONICAL[_item["name"]] = _item["name"]
    for _old in _item["legacy"]:
        _CANONICAL[_old] = _item["name"]


def canonical_category(name: str) -> str:
    """الاسم الحالي للتصنيف (يقبل الأسماء القديمة أيضًا)، أو نص فارغ إن لم يكن تصنيف قصص."""
    return _CANONICAL.get((name or "").strip(), "")


def category_info(name: str) -> dict[str, Any]:
    wanted = canonical_category(name)
    for item in STORIES:
        if item["name"] == wanted:
            return item
    raise KeyError(f"تصنيف قصص غير معروف: {name}")


def labels_for(name: str) -> list[str]:
    """وسوم Blogger للقصة: القسم ثم التصنيف الفرعي."""
    return [PARENT_LABEL, category_info(name)["name"]]


def load_story_results(output_root: Path = Path("output")) -> list[dict[str, Any]]:
    """نتائج القصص المولّدة (kind = story) من الأقدم إلى الأحدث."""
    rows: list[dict[str, Any]] = []
    for path in Path(output_root).glob("*/result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(data, dict) and data.get("kind") == "story"
                and not data.get("rejected")):
            rows.append(data)
    rows.sort(key=lambda d: str(d.get("generated_at_utc") or ""))
    return rows


def next_category(output_root: Path = Path("output"), override: str = "") -> str:
    """
    التصنيف التالي في الدوران. override (من STORY_CATEGORY) يفرض تصنيفًا بعينه
    للتشغيل اليدوي؛ وأي قيمة أخرى (مثل auto أو فارغ) تعني الدوران التلقائي.
    """
    forced = canonical_category(override)
    if forced:
        return forced

    done = [canonical_category(r.get("category", "")) for r in load_story_results(output_root)]
    done = [c for c in done if c]
    if not done:
        return STORY_CATEGORIES[0]
    last = STORY_CATEGORIES.index(done[-1])
    return STORY_CATEGORIES[(last + 1) % len(STORY_CATEGORIES)]
