"""
مجمّع المفاتيح: أكثر من حساب لكل خدمة، ويُبدَّل إلى التالي كلما توقف الحالي.

الصيغ المقبولة لأي متغير (سر في GitHub) مثل GEMINI_WRITER_API_KEY:
  1) الأسهل ولا يحتاج تعديل ملف الـ workflow: ضع كل المفاتيح في السر نفسه مفصولة بفاصلة
     أو سطر جديد:        key1,key2,key3
  2) أو متغيرات إضافية بأرقام (تُضاف إلى env في الـ workflow): NAME_2 و NAME_3 ... حتى NAME_10.

الترتيب هو ترتيب الاستعمال: يُستعمل الأول حتى يتوقف (انتهاء الحصة اليومية أو مفتاح غير صالح)
فيُنتقل إلى الثاني، وهكذا. المفتاح الذي توقف نهائيًا يُهمَل بقية التشغيل فلا يُهدر عليه طلب.
لا تُطبع المفاتيح كاملة أبدًا (mask تُظهر آخر 4 رموز فقط).
"""
from __future__ import annotations

import os
import re

MAX_EXTRA = 10

_SPLIT = re.compile(r"[,\s;،؛]+")
_DEAD: dict[str, str] = {}

# رسائل تعني أن الحساب نفسه متوقف (حصة يومية، مفتاح خاطئ، صلاحية)، لا مجرد ازدحام عابر
_PERMANENT = re.compile(
    r"PerDay|per day|HTTP (?:401|403)|PERMISSION_DENIED|API_KEY_INVALID|"
    r"API key not valid|UNAUTHENTICATED|SERVICE_DISABLED|billing",
    re.IGNORECASE,
)


def split_keys(raw: str | None) -> list[str]:
    """يفصل نصًا فيه مفتاح أو أكثر (فاصلة أو سطر أو مسافة أو ؛) ويزيل علامات الاقتباس."""
    keys = []
    for part in _SPLIT.split(raw or ""):
        part = part.strip().strip("\"'")
        if part:
            keys.append(part)
    return keys


def load_keys(*names: str) -> list[str]:
    """كل مفاتيح المتغيرات المعطاة بالترتيب (NAME ثم NAME_2 ... NAME_10)، بلا تكرار."""
    keys: list[str] = []
    for name in names:
        variables = [name] + [f"{name}_{i}" for i in range(2, MAX_EXTRA + 1)]
        for variable in variables:
            for key in split_keys(os.getenv(variable, "")):
                if key not in keys:
                    keys.append(key)
    return keys


def usable(keys: list[str]) -> list[str]:
    """المفاتيح التي لم تتوقف في هذا التشغيل."""
    return [k for k in keys if k not in _DEAD]


def mark_dead(key: str, reason: str = "") -> None:
    _DEAD[key] = reason[:120]


def is_dead(key: str) -> bool:
    return key in _DEAD


def reset() -> None:
    _DEAD.clear()


def is_permanent(message: str) -> bool:
    """هل تعني الرسالة أن الحساب متوقف نهائيًا (لا عابرًا)؟"""
    return bool(_PERMANENT.search(message or ""))


def mask(key: str) -> str:
    return "…" + key[-4:] if len(key) > 4 else "…"
