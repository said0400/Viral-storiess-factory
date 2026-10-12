"""
المقالات السابقة المنشورة على Blogger، لاستخدامها في الربط الداخلي.

يقرأ output/*/result.json ويختار المقالات التي نُشرت فعلًا (status = LIVE)
وفيها رابط. تُقدَّم مقالات التصنيف نفسه أولًا، ثم الأحدث.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUTPUT_ROOT = Path("output")
MAX_CANDIDATES = 40
MAX_SAME_CATEGORY = 25


def _summary(data: dict[str, Any]) -> str:
    seo = data.get("seo") if isinstance(data.get("seo"), dict) else {}
    text = str(seo.get("meta_description") or "").strip()
    return text[:150]


def published_articles(
    category: str = "",
    exclude_source: str = "",
    limit: int = MAX_CANDIDATES,
) -> list[dict[str, str]]:
    """قائمة [{title, url, summary, category}] لأحدث المقالات المنشورة."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for path in OUTPUT_ROOT.glob("*/result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue

        blog = data.get("blogger") if isinstance(data.get("blogger"), dict) else {}
        url = str(blog.get("url") or "").strip()
        title = str(data.get("title") or "").strip()

        if not url or not title or blog.get("status") != "LIVE":
            continue
        if exclude_source and data.get("source_url") == exclude_source:
            continue  # لا نربط المقال بنفسه عند إعادة المعالجة
        if url in seen:
            continue
        seen.add(url)

        rows.append({
            "title": title,
            "url": url,
            "summary": _summary(data),
            "category": str(data.get("category") or ""),
            "generated_at": str(data.get("generated_at_utc") or ""),
        })

    rows.sort(key=lambda r: r["generated_at"], reverse=True)

    same = [r for r in rows if category and r["category"] == category]
    same = same[:MAX_SAME_CATEGORY]
    same_urls = {r["url"] for r in same}
    rest = [r for r in rows if r["url"] not in same_urls]
    return (same + rest)[:limit]
