"""
التصنيفات الخمسة: مصدر واحد تستورده كل الملفات (writer.py و picker.py و blogger_publish.py)
حتى لا تختلف القوائم بينها. لا تضع هنا شيئًا آخر.
"""
from __future__ import annotations

CATEGORIES = [
    "غرائب وعجائب",
    "معجزات ونجاة",
    "قصص إنسانية وواقعية",
    "تريند",
    "حول العالم",
]

# تصنيف افتراضي: لا يُترك المقال بلا تصنيف أبدًا
DEFAULT_CATEGORY = CATEGORIES[0]
