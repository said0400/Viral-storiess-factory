"""
إرسال حزمة المنشور إلى واتساب عبر GREEN-API بعد اكتمال النشر.

يرسل ثلاث رسائل بالترتيب:
  1) صورة فيسبوك (facebook_image.jpg)
  2) نص المنشور مع الهاشتاغات (facebook_post.txt)
  3) نص التعليق الأول وفيه رابط المقال (first_comment.txt)

الأسرار المطلوبة (GitHub Secrets):
  GREENAPI_ID_INSTANCE   رقم الـ instance من لوحة GREEN-API
  GREENAPI_API_TOKEN     apiTokenInstance
  GREENAPI_CHAT_ID       رقمك بالصيغة الدولية بلا + (مثل 2126XXXXXXXX)
                         أو معرّف كامل مثل 2126XXXXXXXX@c.us أو معرّف مجموعة @g.us
متغيرات اختيارية (GitHub Variables):
  GREENAPI_API_URL       عنوان الـ instance (مثل https://7105.api.greenapi.com)
  GREENAPI_MEDIA_URL     عنوان الوسائط (مثل https://7105.media.greenapi.com)

التشغيل: OUT_DIR=output/<hash> python whatsapp_notify.py
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_URL = "https://api.green-api.com"
DEFAULT_MEDIA_URL = "https://media.green-api.com"
REQUEST_TIMEOUT = 120
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
MAX_TEXT = 20000          # الحد الأقصى لنص الرسالة في GREEN-API
GAP_SECONDS = 2           # فاصل بين الرسائل ليبقى ترتيبها صحيحًا


class WhatsAppError(Exception):
    """خطأ في الإرسال عبر GREEN-API."""


# ---------------------------------------------------------------------------
# الإعدادات
# ---------------------------------------------------------------------------

def load_config() -> dict[str, str] | None:
    """يعيد الإعدادات، أو None إن لم يُضبط الإرسال (فيُتخطى بهدوء)."""
    values = {
        "id": os.getenv("GREENAPI_ID_INSTANCE", "").strip(),
        "token": os.getenv("GREENAPI_API_TOKEN", "").strip(),
        "chat": os.getenv("GREENAPI_CHAT_ID", "").strip(),
    }
    if not any(values.values()):
        print("لم تُضبط أسرار GREEN-API؛ يُتخطى الإرسال إلى واتساب.")
        return None

    missing = [
        name for name, key in (
            ("GREENAPI_ID_INSTANCE", "id"),
            ("GREENAPI_API_TOKEN", "token"),
            ("GREENAPI_CHAT_ID", "chat"),
        ) if not values[key]
    ]
    if missing:
        print("::warning title=GREEN-API::أسرار ناقصة، يُتخطى الإرسال: "
              + "، ".join(missing))
        return None

    values["chat"] = normalize_chat_id(values["chat"])
    values["api_url"] = (
        os.getenv("GREENAPI_API_URL", "").strip() or DEFAULT_API_URL
    ).rstrip("/")
    values["media_url"] = (
        os.getenv("GREENAPI_MEDIA_URL", "").strip() or DEFAULT_MEDIA_URL
    ).rstrip("/")
    return values


def normalize_chat_id(value: str) -> str:
    """رقم دولي بلا + يصبح 'رقم@c.us'، والمعرّف الكامل يبقى كما هو."""
    value = value.strip()
    if "@" in value:
        return value
    digits = re.sub(r"\D", "", value)
    if not digits:
        raise WhatsAppError("GREENAPI_CHAT_ID لا يحتوي رقمًا صالحًا.")
    return f"{digits}@c.us"


# ---------------------------------------------------------------------------
# الاتصال
# ---------------------------------------------------------------------------

def _scrub(text: str, cfg: dict[str, str]) -> str:
    """يمنع ظهور الرمز السري في رسائل الخطأ والسجل."""
    for secret in (cfg.get("token"), cfg.get("id")):
        if secret:
            text = text.replace(secret, "***")
    return text


def _hint(status: int) -> str:
    if status in (401, 403):
        return (" ← تحقق من GREENAPI_ID_INSTANCE و GREENAPI_API_TOKEN، "
                "ومن أن الـ instance مفعّل.")
    return ""


def _request(method: str, url: str, cfg: dict[str, str], **kwargs) -> Any:
    last = "فشل الاتصال بـ GREEN-API."
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.request(
                method, url, timeout=REQUEST_TIMEOUT, **kwargs
            )
        except requests.RequestException as exc:
            last = _scrub(f"تعذر الاتصال بـ GREEN-API: {exc}", cfg)
        else:
            if response.status_code == 200:
                try:
                    return response.json() if response.content else {}
                except ValueError:
                    return {}
            body = _scrub(response.text[:300], cfg)
            last = (f"خطأ GREEN-API HTTP {response.status_code}: {body}"
                    + _hint(response.status_code))
            if response.status_code not in RETRY_STATUS:
                raise WhatsAppError(last)
        if attempt < MAX_ATTEMPTS:
            time.sleep(5 * attempt)
    raise WhatsAppError(last)


def check_authorized(cfg: dict[str, str]) -> None:
    """يتأكد أن واتساب مربوط بالـ instance (مسح QR). الفشل في الفحص لا يوقف الإرسال."""
    url = (f"{cfg['api_url']}/waInstance{cfg['id']}"
           f"/getStateInstance/{cfg['token']}")
    try:
        state = _request("GET", url, cfg).get("stateInstance")
    except WhatsAppError as exc:
        print(f"تنبيه: تعذر فحص حالة الـ instance: {exc}")
        return
    if state and state != "authorized":
        raise WhatsAppError(
            f"واتساب غير مربوط بالـ instance (الحالة: {state}). "
            "افتح لوحة GREEN-API وامسح رمز QR."
        )


def _confirm(result: Any, what: str) -> Any:
    """GREEN-API يعيد idMessage عند قبول الرسالة؛ غيابه يعني أنها قد لا تكون أُرسلت."""
    if not isinstance(result, dict) or not result.get("idMessage"):
        print(f"::warning title=واتساب::لم يؤكد GREEN-API إرسال {what} "
              "(لا idMessage في الرد)؛ تحقق من وصولها.")
    return result


def send_text(cfg: dict[str, str], text: str) -> Any:
    url = (f"{cfg['api_url']}/waInstance{cfg['id']}"
           f"/sendMessage/{cfg['token']}")
    return _confirm(_request(
        "POST", url, cfg,
        json={"chatId": cfg["chat"], "message": text[:MAX_TEXT]},
    ), "رسالة نصية")


def send_image(cfg: dict[str, str], path: Path, caption: str) -> Any:
    url = (f"{cfg['media_url']}/waInstance{cfg['id']}"
           f"/sendFileByUpload/{cfg['token']}")
    data = path.read_bytes()
    return _confirm(_request(
        "POST", url, cfg,
        data={"chatId": cfg["chat"], "fileName": path.name,
              "caption": caption},
        files={"file": (path.name, data, "image/jpeg")},
    ), "الصورة")


# ---------------------------------------------------------------------------
# التنفيذ
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def notify(out_dir: Path, cfg: dict[str, str]) -> int:
    post = _read(out_dir / "facebook_post.txt")
    comment = _read(out_dir / "first_comment.txt")
    image = out_dir / "facebook_image.jpg"

    if not post and not comment and not image.is_file():
        raise WhatsAppError(f"لا توجد ملفات للإرسال في {out_dir}.")

    check_authorized(cfg)
    sent = 0

    if image.is_file():
        send_image(
            cfg, image,
            "صورة المنشور — الرسالتان التاليتان: نص المنشور ثم التعليق الأول",
        )
        sent += 1
        print("أُرسلت صورة المنشور.")
    else:
        send_text(cfg, "⚠ لم تُنشأ صورة فيسبوك لهذا المقال (لا صور صالحة). "
                       "الرسالتان التاليتان: نص المنشور ثم التعليق الأول.")
        sent += 1
        print("تنبيه: لا توجد صورة فيسبوك؛ أُرسل إشعار نصي بدلها.")

    if post:
        time.sleep(GAP_SECONDS)
        send_text(cfg, post)
        sent += 1
        print("أُرسل نص المنشور.")

    if comment:
        time.sleep(GAP_SECONDS)
        send_text(cfg, comment)
        sent += 1
        print("أُرسل نص التعليق الأول.")
        if "{LINK}" in comment:
            time.sleep(GAP_SECONDS)
            send_text(cfg, "⚠ تنبيه: لم يُنشر المقال على Blogger، فرابط "
                           "المقال في التعليق الأول لم يُعبَّأ بعد "
                           "(الوسم {LINK}).")
            print("::warning title=واتساب::التعليق الأول ما زال فيه {LINK}.")

    print(f"اكتمل الإرسال إلى واتساب ({sent} رسائل).")
    return sent


def main() -> int:
    out = os.getenv("OUT_DIR", "").strip()
    if not out:
        print("::error::OUT_DIR غير محدد.", file=sys.stderr)
        return 1
    try:
        cfg = load_config()
        if cfg is None:
            return 0
        notify(Path(out), cfg)
    except WhatsAppError as exc:
        print(f"::error title=فشل الإرسال إلى واتساب::{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"::error title=عطل غير متوقع في إرسال واتساب::"
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
