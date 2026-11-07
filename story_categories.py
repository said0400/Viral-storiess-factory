"""
تصنيفات القصص الأربعة وتناوبها: تصنيف 1 ثم 2 ثم 3 ثم 4 ثم يعود إلى 1 وهكذا.

التصنيف التالي يُحسب من آخر قصة وُلّدت (output/*/result.json التي فيها kind = "story")،
فلا يعتمد على النموذج، ولا يتكرر تصنيف مرتين متتاليتين.

لا تضع هنا شيئًا آخر: هذا المصدر الموحد لأسماء التصنيفات.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# وسم Blogger الأب الذي تحمله كل القصص (صفحة «قصص» في المدونة)
PARENT_LABEL = "قصص"

STORIES: list[dict[str, Any]] = [
    {
        "name": "قصص الجرائم والتحقيقات البوليسية",
        "description": (
            "جرائم القتل، الاختفاء الغامض، التحقيقات، الأدلة، الأسرار، الخيانة، الانتقام، "
            "الألغاز الجنائية والجرائم المعقدة. خيالية بالكامل أو مستوحاة من أجواء قضايا "
            "واقعية (دون ذكر وقائع أو أسماء حقيقية)."
        ),
        "accent": (205, 32, 44),
        "safe_image": (
            "A rain-soaked city street at night, a lone streetlamp, wet asphalt reflecting "
            "red and blue light, empty and silent, noir atmosphere, cinematic."
        ),
    },
    {
        "name": "قصص الرعب والماورائيات والغموض",
        "description": (
            "الأماكن المهجورة، الظواهر الغريبة، الأسرار المرعبة، الأحداث الخارقة، البيوت "
            "المسكونة، الكائنات الغامضة، والأحداث التي تتحدى التوقعات."
        ),
        "accent": (96, 54, 170),
        "safe_image": (
            "An old abandoned house at dusk surrounded by thick fog, a single lit window, "
            "bare trees, eerie cold blue atmosphere, cinematic."
        ),
    },
    {
        "name": "القصص الاجتماعية الواقعية والدراما",
        "description": (
            "الخيانة الزوجية، العلاقات العائلية، الظلم، الفقر، الطمع، الانتقام، التضحية، "
            "الأسرار العائلية، الصراعات الإنسانية والمواقف المؤثرة."
        ),
        "accent": (214, 132, 36),
        "safe_image": (
            "A quiet family kitchen table at night under a dim lamp, an empty chair, warm "
            "amber light and deep shadows, melancholic mood, cinematic."
        ),
    },
    {
        "name": "قصص الذكاء والمكر والخدع",
        "description": (
            "عمليات النصب، الخدع الذكية، السرقات المعقدة، التلاعب النفسي، المؤامرات، "
            "الأسرار المالية، الخطط الماكرة والخداع غير المتوقع."
        ),
        "accent": (30, 150, 130),
        "safe_image": (
            "A dim luxurious hotel lobby at night, a leather briefcase on a marble table, "
            "golden light and deep shadows, tense elegant atmosphere, cinematic."
        ),
    },
]

STORY_CATEGORIES: list[str] = [s["name"] for s in STORIES]


def category_info(name: str) -> dict[str, Any]:
    for item in STORIES:
        if item["name"] == name:
            return item
    raise KeyError(f"تصنيف قصص غير معروف: {name}")


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
    override = (override or "").strip()
    if override in STORY_CATEGORIES:
        return override

    done = [r for r in load_story_results(output_root)
            if r.get("category") in STORY_CATEGORIES]
    if not done:
        return STORY_CATEGORIES[0]
    last = STORY_CATEGORIES.index(done[-1]["category"])
    return STORY_CATEGORIES[(last + 1) % len(STORY_CATEGORIES)]
