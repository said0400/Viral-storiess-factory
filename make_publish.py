"""
النشر عبر Make: يرسل المقال إلى Webhook، وMake ينشره على Blogger ويُرجع رابطه.

- الصورة المصغرة تُوضع أول صورة في المقال، وصورة المنتصف في موضعها داخل النص.
- الصور تُستضاف في مستودع GitHub العام وتُدرج برابط jsDelivr ثابت لكل commit.
- الروابط الداخلية (اقرأ أيضًا) تُحوَّل من Markdown إلى روابط HTML.
- Make يردّ بـ post_id فقط (وحدة Blogger لا تُرجع الرابط)؛ ويُجلب رابط المقال من
  خلاصة المدونة العامة (BLOGGER_BLOG_ID) ويُوضع مكان {LINK} في first_comment.txt.
  وإن ردّ Make برابط (url) يُستعمل مباشرة دون بحث.
- إن كان للمقال post_id محفوظ يُرسل إلى Make للتحديث بدل إنشاء مقال مكرر.
- إن لم يُحدَّد BLOGGER_LABEL يُستعمل تصنيف الذكاء الاصطناعي من result.json.

التشغيل: python make_publish.py (من جذر المستودع، بعد دفع المخرجات).
لا يعتمد على blogger_publish.py ولا على أي سر من أسرار Blogger.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# تصنيف افتراضي: لا يُنشر المقال بلا تسمية أبدًا (مصدر موحد في categories.py)
from categories import DEFAULT_CATEGORY as DEFAULT_LABEL  # noqa: E402

TIMEOUT = 90          # Make قد يتأخر في الرد
REQUEST_TIMEOUT = 20
RETRY_STATUS = {429, 500, 502, 503, 504}


class MakeError(Exception):
    """خطأ في النشر عبر Make."""


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# تحويل Markdown إلى HTML
# ---------------------------------------------------------------------------

def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
                  r'<a href="\2">\1</a>', text)
    return text


def figure(src: str, alt: str) -> str:
    return (
        '<div class="separator" style="clear: both; text-align: center;">'
        f'<img alt="{html.escape(alt, quote=True)}" '
        f'src="{html.escape(src, quote=True)}" '
        'style="max-width: 100%; height: auto;" /></div>'
    )


def markdown_to_html(markdown: str, resolve_src=lambda s: s) -> str:
    out: list[str] = []
    para: list[str] = []

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        image = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", stripped)
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if image:
            flush()
            out.append(figure(resolve_src(image.group(2)), image.group(1)))
        elif heading:
            flush()
            level = min(max(len(heading.group(1)), 2), 4)
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        else:
            para.append(stripped)
    flush()
    return "\n".join(out)


def place_toc(body_html: str) -> str:
    """يضع حاوية جدول المحتويات (tocDiv) بعد المقدمة مباشرة."""
    match = re.search(r'<h[2-4][ >]|<div class="separator"', body_html)
    if not match:
        return body_html
    return (body_html[:match.start()] + '<div id="tocDiv"></div>\n'
            + body_html[match.start():])


def build_post_html(
    markdown: str, thumbnail_url: str | None, thumbnail_alt: str,
    resolve_src=lambda s: s,
) -> str:
    parts = []
    if thumbnail_url:
        parts.append(figure(thumbnail_url, thumbnail_alt))
    parts.append(place_toc(markdown_to_html(markdown, resolve_src)))
    return (
        '<div dir="rtl" style="text-align: right;">\n'
        + "\n".join(parts)
        + "\n</div>"
    )


def _strip_h1(markdown: str) -> str:
    return re.sub(r"^\s*#\s+[^\n]*\n+", "", markdown, count=1)


# ---------------------------------------------------------------------------
# استضافة الصور (المستودع العام + jsDelivr)
# ---------------------------------------------------------------------------

def image_bases(repo: str, sha: str) -> list[str]:
    return [
        f"https://cdn.jsdelivr.net/gh/{repo}@{sha}",
        f"https://raw.githubusercontent.com/{repo}/{sha}",
    ]


def _reachable(url: str, attempts: int = 4, wait: float = 5.0) -> bool:
    for attempt in range(attempts):
        try:
            response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            ok = (response.status_code == 200
                  and response.headers.get("Content-Type", "")
                  .lower().startswith("image/"))
            response.close()
            if ok:
                return True
        except requests.RequestException:
            pass
        if attempt < attempts - 1:
            time.sleep(wait)
    return False


def public_image_url(repo: str, sha: str, rel_path: str) -> str:
    """أول رابط عام يعمل فعلًا (jsDelivr ثم raw)، وإلا خطأ واضح."""
    for base in image_bases(repo, sha):
        url = f"{base}/{rel_path}"
        if _reachable(url):
            return url
    raise MakeError(
        f"تعذر الوصول إلى الصورة علنًا ({rel_path}). تأكد أن المستودع عام "
        "(Public) وأن الملفات دُفعت قبل النشر."
    )


# ---------------------------------------------------------------------------
# التعليق الأول والتصنيفات
# ---------------------------------------------------------------------------

def fill_first_comment(
    out_dir: Path, result: dict[str, Any], url: str | None
) -> None:
    """يضع رابط المقال الحقيقي مكان {LINK} في first_comment.txt."""
    comment = result.get("first_comment")
    if not comment or not url:
        return
    (out_dir / "first_comment.txt").write_text(
        str(comment).replace("{LINK}", url).rstrip() + "\n",
        encoding="utf-8",
    )
    print("تم تحديث first_comment.txt برابط المقال.")


def split_labels(requested: str) -> list[str]:
    out: list[str] = []
    for raw in re.split(r"[,،]", requested or ""):
        name = raw.strip()
        if name and name not in out:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# الإرسال إلى Make
# ---------------------------------------------------------------------------

FEED_URL = "https://www.blogger.com/feeds/{blog}/posts/default/{post}?alt=json"


def lookup_post_url(blog_id: str, post_id: str, attempts: int = 6,
                    wait: float = 5.0) -> str | None:
    """
    رابط المقال المنشور من خلاصة Blogger العامة (بلا مصادقة). وحدة Blogger في Make
    تُرجع post_id فقط، فنسأل الخلاصة عن الرابط. تعيد المحاولة قليلًا لأن المقال
    قد يتأخر ظهوره في الخلاصة ثواني بعد النشر. المسودات ليست علنية فلا رابط لها.
    """
    if not blog_id or not post_id:
        return None
    url = FEED_URL.format(blog=blog_id, post=post_id)
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                entry = response.json().get("entry") or {}
                for link in entry.get("link") or []:
                    href = str(link.get("href") or "")
                    if link.get("rel") == "alternate" and href:
                        return re.sub(
                            r"^http://([^/]*\.blogspot\.[a-z.]+)/",
                            r"https://\1/", href)
        except (requests.RequestException, ValueError, AttributeError):
            pass
        if attempt < attempts - 1:
            time.sleep(wait)
    return None


def post_to_make(payload: dict[str, Any]) -> dict[str, Any]:
    url = os.getenv("MAKE_WEBHOOK_URL", "").strip()
    if not url:
        raise MakeError("سر MAKE_WEBHOOK_URL غير موجود.")
    headers = {"Content-Type": "application/json"}
    secret = os.getenv("MAKE_WEBHOOK_SECRET", "").strip()
    if secret:
        headers["x-make-apikey"] = secret

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last = "فشل الاتصال بـ Make."
    for attempt in (1, 2):
        try:
            r = requests.post(url, headers=headers, data=body,
                              timeout=TIMEOUT)
        except requests.RequestException as exc:
            last = f"تعذر الاتصال بـ Make: {exc}"
        else:
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:
                    raise MakeError(
                        "ردّ Make ليس JSON (غالبًا نصّ Accepted). أضف وحدة "
                        "Webhook response في آخر السيناريو لإرجاع رابط المقال."
                    )
                if not isinstance(data, dict):
                    raise MakeError("ردّ Make غير متوقع (ليس كائن JSON).")
                return data
            last = f"خطأ Make HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in RETRY_STATUS:
                raise MakeError(last)
        if attempt == 1:
            time.sleep(5)
    raise MakeError(last)


def publish(out_dir: Path) -> dict[str, Any]:
    result_path = out_dir / "result.json"
    if not result_path.is_file():
        raise MakeError(f"لا يوجد result.json في {out_dir}.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    previous = result.get("blogger") or {}

    if previous.get("post_id") and not _flag("FORCE_REPROCESS"):
        print(f"المقال منشور سابقًا ({previous.get('url')}); لا حاجة للنشر.")
        fill_first_comment(out_dir, result, previous.get("url"))
        return previous

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    sha = os.getenv("BLOGGER_COMMIT_SHA", "").strip()
    if not repo or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise MakeError(
            "تعذر تحديد المستودع أو commit الصور (GITHUB_REPOSITORY / "
            "BLOGGER_COMMIT_SHA)."
        )

    md_file = out_dir / "rewritten_article.md"
    if not md_file.is_file():
        raise MakeError("rewritten_article.md غير موجود.")
    markdown = _strip_h1(md_file.read_text(encoding="utf-8"))

    workspace = Path(os.getenv("GITHUB_WORKSPACE") or Path.cwd()).resolve()

    def rel(name: str) -> str:
        return Path(os.path.relpath((out_dir / name).resolve(),
                                    workspace)).as_posix()

    urls: dict[str, str] = {}
    for name in ("thumbnail.jpg", "article_middle.jpg"):
        if (out_dir / name).is_file():
            urls[name] = public_image_url(repo, sha, rel(name))

    title = result["title"]
    content = build_post_html(
        markdown, urls.get("thumbnail.jpg"),
        result.get("image_title") or title,
        resolve_src=lambda src: urls.get(src, src),
    )

    requested = (os.getenv("BLOGGER_LABEL", "").strip()
                 or str(result.get("category") or "").strip())
    labels = split_labels(requested) or [DEFAULT_LABEL]
    draft = _flag("BLOGGER_DRAFT")

    payload = {
        "title": title,
        "content": content,
        "labels": labels,
        "labels_csv": ", ".join(labels),
        "draft": draft,
        "status": "DRAFT" if draft else "LIVE",
        "action": "update" if previous.get("post_id") else "insert",
        "post_id": previous.get("post_id") or "",   # للتحديث عبر Router
        "source_url": result.get("source_url"),
        "thumbnail_url": urls.get("thumbnail.jpg", ""),
        "images": urls,
        "seo": result.get("seo"),
    }

    reply = post_to_make(payload)
    post_id = str(reply.get("post_id") or previous.get("post_id") or "").strip()
    post_url = str(reply.get("url") or "").strip() or None

    # related.py يشترط status == "LIVE" حرفيًا لربط المقالات ببعضها؛ نوحّد الصياغة
    raw_status = str(reply.get("status") or "").strip().upper()
    if raw_status == "PUBLISHED":
        raw_status = "LIVE"
    status = raw_status or ("DRAFT" if draft else "LIVE")

    record = {
        "via": "make",
        "post_id": post_id or None,
        "url": post_url,
        "status": status,
        "labels": labels,
        "action": "update" if previous.get("post_id") else "insert",
        "images": urls,
        "commit": sha,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # المسودة ليست علنية فلا رابط لها في الخلاصة العامة
    if not post_url and post_id and status != "DRAFT":
        post_url = lookup_post_url(os.getenv("BLOGGER_BLOG_ID", "").strip(),
                                   post_id)
        record["url"] = post_url

    # نحفظ السجل أولًا (حتى لا يتكرر النشر لاحقًا) ثم نفحص الرابط
    result["blogger"] = record
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")

    if not post_id and not post_url:
        raise MakeError(
            "ردّ Make بلا post_id ولا url. تأكد أن وحدة Webhook response "
            "ترد بحقل post_id (من وحدة Create a post)."
        )

    if not post_url:
        if status == "DRAFT":
            print("::warning title=مقال بحالة مسودة::لا رابط عامًا للمسودة، "
                  "فيبقى {LINK} في first_comment.txt حتى يُنشر المقال.")
            print(f"أُرسل المقال مسودةً عبر Make (post_id={post_id}).")
            return record
        raise MakeError(
            "نُشر المقال لكن تعذّر جلب رابطه من خلاصة المدونة. تأكد من ضبط "
            "BLOGGER_BLOG_ID (رقم المدونة) وأن المدونة عامة؛ ولن يُعبَّأ "
            "التعليق الأول قبل ذلك."
        )

    fill_first_comment(out_dir, result, post_url)
    if status == "DRAFT":
        print("::warning title=مقال بحالة مسودة::رابط التعليق الأول لن يفتحه "
              "الزوار حتى يُنشر المقال.")
    print(f"تم النشر عبر Make: {post_url} ({record['status']})")
    return record


def main() -> int:
    out = os.getenv("OUT_DIR", "").strip()
    if not out:
        print("::error::OUT_DIR غير محدد.", file=sys.stderr)
        return 1
    try:
        publish(Path(out))
    except MakeError as exc:
        print(f"::error title=فشل النشر عبر Make::{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"::error title=عطل غير متوقع في النشر::"
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
