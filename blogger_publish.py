"""
النشر التلقائي على Blogger بعد توليد الحزمة.

- الصورة المصغرة تُوضع أول صورة في المقال (قوالب Blogger تأخذها مصغّرةً)،
  وصورة المنتصف في موضعها داخل النص.
- الصور تُستضاف في مستودع GitHub العام وتُدرج برابط jsDelivr ثابت لكل commit
  (واجهة Blogger لا ترفع الصور).
- التصنيف = تسمية (Label): تُطابق الموجود دون حساسية لحالة الأحرف وإلا تُنشأ.
- إن كان للرابط مقال منشور سابقًا يُحدَّث بدل إنشاء مقال مكرر.
- الروابط الداخلية في المقال (اقرأ أيضًا) تُحوَّل من Markdown إلى روابط HTML.
- التعليق الأول (first_comment.txt) يُعبَّأ برابط المقال الحقيقي بعد النشر.
- إن لم يُحدَّد BLOGGER_LABEL يُستعمل تصنيف الذكاء الاصطناعي المحفوظ في result.json.

التشغيل: python blogger_publish.py (من جذر المستودع، بعد دفع المخرجات).
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


TOKEN_URL = "https://oauth2.googleapis.com/token"
API_BASE = "https://www.googleapis.com/blogger/v3/blogs"
REQUEST_TIMEOUT = 60
RETRY_STATUS = {429, 500, 502, 503, 504}

# تصنيف افتراضي: لا يُنشر المقال بلا تسمية أبدًا (مصدر موحد في categories.py)
from categories import DEFAULT_CATEGORY as DEFAULT_LABEL  # noqa: E402


class BloggerError(Exception):
    """خطأ في النشر على Blogger."""


# ---------------------------------------------------------------------------
# الإعدادات والمصادقة
# ---------------------------------------------------------------------------

def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def credentials() -> dict[str, str]:
    names = ("BLOGGER_CLIENT_ID", "BLOGGER_CLIENT_SECRET",
             "BLOGGER_REFRESH_TOKEN", "BLOGGER_BLOG_ID")
    values = {n: os.getenv(n, "").strip() for n in names}
    missing = [n for n, v in values.items() if not v]
    if missing:
        raise BloggerError(
            "أسرار Blogger ناقصة في GitHub Secrets: " + "، ".join(missing)
        )
    return values


def get_access_token(creds: dict[str, str]) -> str:
    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "client_id": creds["BLOGGER_CLIENT_ID"],
                "client_secret": creds["BLOGGER_CLIENT_SECRET"],
                "refresh_token": creds["BLOGGER_REFRESH_TOKEN"],
                "grant_type": "refresh_token",
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise BloggerError(f"تعذر الاتصال بخادم Google للمصادقة: {exc}") from exc

    if response.status_code != 200:
        try:
            body = response.json()
        except ValueError:
            body = {}
        code = str(body.get("error", response.status_code))
        hint = ""
        if code == "invalid_grant":
            hint = (" ← رمز التحديث منتهٍ أو ملغى. إن كان تطبيق OAuth في وضع "
                    "Testing فصلاحيته 7 أيام؛ انقله إلى In production وأعد "
                    "إنشاء الرمز.")
        elif code == "invalid_client":
            hint = " ← BLOGGER_CLIENT_ID أو BLOGGER_CLIENT_SECRET غير صحيح."
        raise BloggerError(f"فشلت مصادقة Google ({code}){hint}")

    token = response.json().get("access_token")
    if not token:
        raise BloggerError("لم يُعد Google رمز وصول.")
    return token


def _api(method: str, token: str, url: str, **kwargs) -> Any:
    headers = {"Authorization": f"Bearer {token}"}
    last = "فشل الاتصال بـ Blogger."
    for attempt in (1, 2):
        try:
            response = requests.request(
                method, url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs
            )
        except requests.RequestException as exc:
            last = f"تعذر الاتصال بـ Blogger: {exc}"
        else:
            if response.status_code in (200, 201):
                return response.json() if response.content else {}
            detail = response.text[:300]
            last = f"خطأ Blogger HTTP {response.status_code}: {detail}"
            if response.status_code == 403:
                last += (" ← الحساب لا يملك صلاحية الكتابة على هذه المدونة "
                         "أو واجهة Blogger API غير مفعّلة في مشروع Google.")
            elif response.status_code == 404:
                last += " ← تحقق من BLOGGER_BLOG_ID (أو أن المقال حُذف)."
            if response.status_code not in RETRY_STATUS:
                raise BloggerError(last)
        if attempt == 1:
            time.sleep(5)
    raise BloggerError(last)


# ---------------------------------------------------------------------------
# التصنيفات (Labels)
# ---------------------------------------------------------------------------

def existing_labels(token: str, blog_id: str) -> list[str]:
    """تسميات المدونة الحالية، بجمعها من المقالات (لا توجد نقطة مخصصة لها)."""
    labels: dict[str, str] = {}
    page = None
    for _ in range(6):
        params = {"fetchBodies": "false", "maxResults": 500,
                  "fields": "items(labels),nextPageToken"}
        if page:
            params["pageToken"] = page
        data = _api("GET", token, f"{API_BASE}/{blog_id}/posts", params=params)
        for item in data.get("items", []):
            for label in item.get("labels", []) or []:
                labels.setdefault(label.casefold(), label)
        page = data.get("nextPageToken")
        if not page:
            break
    return list(labels.values())


def resolve_labels(requested: str, known: list[str]) -> list[str]:
    """يطابق الأسماء المطلوبة بالموجود (دون حساسية لحالة الأحرف)."""
    index = {k.casefold(): k for k in known}
    result: list[str] = []
    for raw in re.split(r"[,،]", requested or ""):
        name = raw.strip()
        if not name:
            continue
        chosen = index.get(name.casefold(), name)
        if chosen not in result:
            result.append(chosen)
    return result


# ---------------------------------------------------------------------------
# تحويل Markdown إلى HTML لـ Blogger
# ---------------------------------------------------------------------------

def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)
    # روابط Markdown (للربط الداخلي بالمقالات السابقة): [نص](https://...)
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
    """محوّل صغير يكفي بنية المقال: عناوين ## وفقرات وصور ونص غامق/مائل."""
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
    """
    يضع حاوية جدول المحتويات (tocDiv) بعد المقدمة مباشرة، قبل أول عنوان فرعي
    أو أول صورة. قالب المدونة يضيف الجدول أعلى المقال تلقائيًا إلا إذا وجد
    الكلمة tocDiv في نص المقال، فيستعمل موضعنا بدلًا من موضعه.
    """
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
            response = requests.get(url, stream=True, timeout=20)
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
    raise BloggerError(
        f"تعذر الوصول إلى الصورة علنًا ({rel_path}). تأكد أن المستودع عام "
        "(Public) وأن الملفات دُفعت قبل النشر."
    )


# ---------------------------------------------------------------------------
# فحص صورة المقال كما يراها Blogger
# ---------------------------------------------------------------------------

def detect_display_image(token: str, blog_id: str, post_id: str) -> list[str]:
    """
    يسأل Blogger عن صور المقال التي تعرّف عليها (fetchImages). إن كانت القائمة
    فارغة فالقوالب التي تعتمد على صورة Blogger لن تعرض مصغّرة للمقال.
    نعيد المحاولة قليلًا لأن Blogger قد يحسبها بعد النشر بثوانٍ.
    """
    for attempt in range(3):
        data = _api(
            "GET", token, f"{API_BASE}/{blog_id}/posts/{post_id}",
            params={"fetchImages": "true", "fields": "images"},
        )
        images = [i["url"] for i in data.get("images", []) or [] if i.get("url")]
        if images:
            return images
        if attempt < 2:
            time.sleep(6)
    return []


def report_images(images: list[str]) -> None:
    if images:
        print(f"تعرّف Blogger على {len(images)} صورة للمقال: {images[0]}")
    else:
        print(
            "::warning title=صورة المقال::لم يتعرف Blogger على أي صورة في "
            "المقال، فقد لا تظهر مصغّرة في قوائم القالب (آخر المشاركات وغيرها) "
            "رغم ظهورها داخل المقال. السبب المحتمل: الصور مستضافة خارج Blogger."
        )


# ---------------------------------------------------------------------------
# النشر
# ---------------------------------------------------------------------------

def _strip_h1(markdown: str) -> str:
    return re.sub(r"^\s*#\s+[^\n]*\n+", "", markdown, count=1)


def fill_first_comment(
    out_dir: Path, result: dict[str, Any], url: str | None
) -> None:
    """يضع رابط المقال الحقيقي مكان {LINK} في first_comment.txt بعد النشر."""
    comment = result.get("first_comment")
    if not comment or not url:
        return
    (out_dir / "first_comment.txt").write_text(
        str(comment).replace("{LINK}", url).rstrip() + "\n",
        encoding="utf-8",
    )
    print("تم تحديث first_comment.txt برابط المقال.")


def publish(out_dir: Path) -> dict[str, Any]:
    result_path = out_dir / "result.json"
    if not result_path.is_file():
        raise BloggerError(f"لا يوجد result.json في {out_dir}.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    previous = result.get("blogger") or {}

    force = _flag("FORCE_REPROCESS")
    if previous.get("post_id") and not force:
        print(f"المقال منشور سابقًا ({previous.get('url')}); لا حاجة للنشر.")
        fill_first_comment(out_dir, result, previous.get("url"))
        try:
            creds = credentials()
            token = get_access_token(creds)
            images = detect_display_image(
                token, creds["BLOGGER_BLOG_ID"], previous["post_id"]
            )
            previous = dict(previous, blogger_images=images,
                            image_checked_at_utc=datetime.now(
                                timezone.utc).isoformat())
            result["blogger"] = previous
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            report_images(images)
        except BloggerError as exc:
            print(f"تنبيه: تعذر فحص صورة المقال: {exc}")
        return previous

    creds = credentials()
    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    sha = os.getenv("BLOGGER_COMMIT_SHA", "").strip()
    if not repo or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise BloggerError(
            "تعذر تحديد المستودع أو commit الصور (GITHUB_REPOSITORY / "
            "BLOGGER_COMMIT_SHA)."
        )

    md_file = out_dir / "rewritten_article.md"
    if not md_file.is_file():
        raise BloggerError("rewritten_article.md غير موجود.")
    markdown = _strip_h1(md_file.read_text(encoding="utf-8"))

    workspace = Path(os.getenv("GITHUB_WORKSPACE") or Path.cwd()).resolve()

    def rel(name: str) -> str:
        """مسار الملف نسبيًا لجذر المستودع (كما يظهر في رابط CDN)."""
        return Path(
            os.path.relpath((out_dir / name).resolve(), workspace)
        ).as_posix()

    urls: dict[str, str] = {}
    for name in ("thumbnail.jpg", "article_middle.jpg"):
        if (out_dir / name).is_file():
            urls[name] = public_image_url(repo, sha, rel(name))

    title = result["title"]
    thumb_alt = result.get("image_title") or title
    content = build_post_html(
        markdown, urls.get("thumbnail.jpg"), thumb_alt,
        resolve_src=lambda src: urls.get(src, src),
    )

    token = get_access_token(creds)
    blog_id = creds["BLOGGER_BLOG_ID"]

    requested = os.getenv("BLOGGER_LABEL", "").strip()
    if not requested:
        # لا تصنيف يدوي: نستعمل تصنيف الذكاء الاصطناعي المحفوظ في result.json
        requested = str(result.get("category") or "").strip()
    labels: list[str] = []
    if requested:
        try:
            labels = resolve_labels(requested, existing_labels(token, blog_id))
        except BloggerError as exc:
            print(f"تنبيه: تعذر جلب التسميات الحالية ({exc}); تُستخدم كما كُتبت.")
            labels = resolve_labels(requested, [])
    else:
        print("تنبيه: لم يُحدَّد تصنيف؛ يُنشر المقال بلا تسمية.")
    if not labels:
        # شبكة أمان أخيرة: لا يُنشر المقال بلا تصنيف
        labels = [DEFAULT_LABEL]
        print(f"تنبيه: لا تصنيف متاح؛ استُخدم التصنيف الافتراضي «{DEFAULT_LABEL}».")

    draft = _flag("BLOGGER_DRAFT")
    body = {"kind": "blogger#post", "title": title, "content": content,
            "labels": labels}
    fields = "id,url,status,labels"
    action = "insert"
    post: dict[str, Any] | None = None

    if previous.get("post_id"):
        try:
            post = _api("PATCH", token,
                        f"{API_BASE}/{blog_id}/posts/{previous['post_id']}",
                        json=body, params={"fields": fields})
            action = "update"
            if not draft and post.get("status") == "DRAFT":
                post = _api("POST", token,
                            f"{API_BASE}/{blog_id}/posts/{previous['post_id']}"
                            "/publish", params={"fields": fields})
        except BloggerError as exc:
            if "404" not in str(exc):
                raise
            print("المقال السابق غير موجود؛ يُنشأ مقال جديد.")
            post = None

    if post is None:
        post = _api("POST", token, f"{API_BASE}/{blog_id}/posts",
                    json=body,
                    params={"isDraft": str(draft).lower(), "fields": fields})
        action = "insert"

    try:
        display_images = detect_display_image(token, blog_id, post.get("id"))
    except BloggerError as exc:
        print(f"تنبيه: تعذر فحص صورة المقال: {exc}")
        display_images = None

    record = {
        "blogger_images": display_images,
        "post_id": post.get("id"),
        "url": post.get("url"),
        "status": post.get("status"),
        "labels": post.get("labels", labels),
        "action": action,
        "images": urls,
        "commit": sha,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    fill_first_comment(out_dir, result, record["url"])
    if record["status"] == "DRAFT":
        print("::warning title=مقال بحالة مسودة::رابط التعليق الأول لن يفتحه "
              "الزوار حتى يُنشر المقال.")
    result["blogger"] = record
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"تم {'تحديث' if action == 'update' else 'نشر'} المقال: "
          f"{record['url']} ({record['status']})")
    if display_images is not None:
        report_images(display_images)
    return record


def main() -> int:
    out = os.getenv("OUT_DIR", "").strip()
    if not out:
        print("::error::OUT_DIR غير محدد.", file=sys.stderr)
        return 1
    try:
        publish(Path(out))
    except BloggerError as exc:
        print(f"::error title=فشل النشر على Blogger::{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"::error title=عطل غير متوقع في النشر::"
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
