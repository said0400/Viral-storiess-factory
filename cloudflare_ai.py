"""
عميل Cloudflare Workers AI لتوليد صورة من صور مرجعية.

النموذج الافتراضي FLUX.2 [klein] 4B: يدعم حتى 4 صور مرجعية، وتكلفته بحسب
صفحة التسعير الرسمية 5.37 نيورون لكل مربع إدخال 512×512 و26.05 نيورون
لكل مربع إخراج 512×512، أي نحو 100 إلى 180 نيورون للصورة 1280×720،
فتتسع الحصة المجانية (10,000 نيورون يوميًا) لعشرات الصور.
"""
from __future__ import annotations

import base64
import os
import re
import time
from io import BytesIO
from typing import Any

import requests
from PIL import Image, ImageOps

import key_pool


API_BASE = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
# نموذج احتياطي (نص إلى صورة، طلب JSON) يُستعمل حين يرفض النموذج الأول الطلب
DEFAULT_FALLBACK_MODEL = "@cf/black-forest-labs/flux-1-schnell"

MAX_REFERENCES = 4
REFERENCE_MAX_SIDE = 480        # الوثائق: يجب أن تكون الصور أصغر من 512×512
REQUEST_TIMEOUT = 150
MAX_ATTEMPTS = 5          # يتحمل الازدحام: انتظار متصاعد بين المحاولات
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class CloudflareError(Exception):
    """خطأ في واجهة Cloudflare أو في الصورة المُعادة."""


class CloudflareQuotaError(CloudflareError):
    """انتهت الحصة اليومية المجانية (الخطأ 3036): لا جدوى من الإعادة اليوم."""


class CloudflareFlaggedError(CloudflareError):
    """رفض مرشّح المحتوى الطلب أو الناتج (الخطأ 3030)؛ قد تنجح محاولة مختلفة."""


_MISMATCH_WARNED = False


def accounts() -> list[tuple[str, str]]:
    """
    حسابات Cloudflare كأزواج (account_id، token) بالترتيب. تُوضع في السرين نفسيهما مفصولة
    بفاصلة وبالترتيب نفسه:
        CF_ACCOUNT_ID = id1,id2        CF_API_TOKEN = token1,token2
    (أو CF_ACCOUNT_ID_2 و CF_API_TOKEN_2 ... إن أضفتها إلى env في الـ workflow).
    وإن وُجد معرّف واحد مع عدة رموز يُستعمل المعرّف نفسه معها.
    """
    global _MISMATCH_WARNED
    ids = key_pool.load_keys("CF_ACCOUNT_ID")
    tokens = key_pool.load_keys("CF_API_TOKEN")
    if len(ids) == 1 and len(tokens) > 1:
        ids = ids * len(tokens)
    if len(ids) != len(tokens):
        count = min(len(ids), len(tokens))
        if not _MISMATCH_WARNED:
            _MISMATCH_WARNED = True
            print(f"::warning title=حسابات Cloudflare::عدد CF_ACCOUNT_ID ({len(ids)}) لا يساوي "
                  f"عدد CF_API_TOKEN ({len(tokens)})؛ يُستعمل أول {count} فقط بالترتيب.")
        ids, tokens = ids[:count], tokens[:count]
    return list(zip(ids, tokens))


def is_configured() -> bool:
    return bool(accounts())


def model_name() -> str:
    return os.getenv("CF_IMAGE_MODEL", "").strip() or DEFAULT_MODEL


def fallback_model_name() -> str:
    return os.getenv("CF_IMAGE_FALLBACK_MODEL", "").strip() or DEFAULT_FALLBACK_MODEL


def json_payload(model: str, prompt: str, width: int, height: int) -> dict[str, Any]:
    """جسم JSON للنماذج التي لا تستعمل multipart (مثل FLUX.1 schnell)."""
    if "flux-1" in model:
        # الوثائق: الحد الأقصى للبرومبت 2048 حرفًا، وsteps افتراضيها 4
        return {"prompt": prompt[:2048], "steps": 4}
    return {"prompt": prompt, "width": width, "height": height}


def _reference_bytes(image: Image.Image) -> bytes:
    ref = ImageOps.exif_transpose(image).convert("RGB")
    ref.thumbnail(
        (REFERENCE_MAX_SIDE, REFERENCE_MAX_SIDE), Image.Resampling.LANCZOS
    )
    buffer = BytesIO()
    ref.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def build_form(
    prompt: str,
    references: list[Image.Image],
    width: int,
    height: int,
    seed: int | None = None,
) -> dict[str, tuple]:
    """
    حقول multipart/form-data. تُرسل كلها كـ files بقيمة (None, value)
    ليبقى الطلب multipart حتى بلا صور مرجعية، كما تشترط الوثائق.
    """
    form: dict[str, tuple] = {
        "prompt": (None, prompt),
        "width": (None, str(width)),
        "height": (None, str(height)),
    }
    if seed is not None:
        form["seed"] = (None, str(seed))
    for index, ref in enumerate(references[:MAX_REFERENCES]):
        form[f"input_image_{index}"] = (
            f"reference_{index}.jpg", _reference_bytes(ref), "image/jpeg",
        )
    return form


def _error_text(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:400]
    errors = data.get("errors") if isinstance(data, dict) else None
    if errors:
        return "; ".join(
            f"{e.get('code', '')}: {e.get('message', '')}"
            if isinstance(e, dict) else str(e)
            for e in errors
        )[:400]
    return str(data)[:400]


def _decode_image(response: requests.Response) -> Image.Image:
    content_type = response.headers.get("Content-Type", "").lower()

    if content_type.startswith("image/"):
        raw = response.content
    else:
        try:
            data = response.json()
        except ValueError as exc:
            raise CloudflareError("استجابة Cloudflare ليست JSON صالحًا.") from exc

        if isinstance(data, dict) and data.get("success") is False:
            raise CloudflareError(f"رفض Cloudflare الطلب: {_error_text(response)}")

        result: Any = data.get("result", data) if isinstance(data, dict) else data
        encoded = result.get("image") if isinstance(result, dict) else result
        if not isinstance(encoded, str) or not encoded.strip():
            raise CloudflareError("لم تتضمن استجابة Cloudflare صورة.")
        encoded = re.sub(r"^data:image/[^;]+;base64,", "", encoded.strip())
        try:
            raw = base64.b64decode(encoded, validate=False)
        except ValueError as exc:
            raise CloudflareError("تعذر فك ترميز الصورة المُعادة.") from exc

    try:
        with Image.open(BytesIO(raw)) as source:
            source.load()
            return source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise CloudflareError("الملف المُعاد ليس صورة صالحة.") from exc


def _generate_image_once(
    prompt: str,
    references: list[Image.Image],
    width: int = 1280,
    height: int = 720,
    seed: int | None = None,
    model: str | None = None,
    account: str = "",
    token: str = "",
) -> Image.Image:
    """يولّد صورة بحساب واحد بالأبعاد المطلوبة، أو يرفع CloudflareError."""
    if not account or not token:
        first = (accounts() or [("", "")])[0]
        account, token = account or first[0], token or first[1]
    if not account or not token:
        raise CloudflareError(
            "المفتاحان CF_ACCOUNT_ID و CF_API_TOKEN غير موجودين في Secrets."
        )

    chosen_model = model or model_name()
    url = f"{API_BASE}/{account}/ai/run/{chosen_model}"
    headers = {"Authorization": f"Bearer {token}"}
    last_error = "فشل الاتصال بـ Cloudflare."

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if "flux-2" in chosen_model:
            body = {"files": build_form(prompt, references, width, height, seed)}
        else:
            body = {"json": json_payload(chosen_model, prompt, width, height)}
        try:
            response = requests.post(
                url, headers=headers, timeout=REQUEST_TIMEOUT, **body
            )
        except requests.RequestException as exc:
            last_error = f"تعذر الاتصال بـ Cloudflare: {exc}"
        else:
            if response.status_code == 200:
                image = _decode_image(response)
                if image.size != (width, height):
                    image = ImageOps.fit(
                        image, (width, height), Image.Resampling.LANCZOS
                    )
                return image

            last_error = (
                f"خطأ Cloudflare HTTP {response.status_code}: "
                f"{_error_text(response)}"
            )
            if "3036" in last_error or "daily free allocation" in last_error:
                raise CloudflareQuotaError(last_error)
            if "3030" in last_error or "flagged" in last_error.lower():
                raise CloudflareFlaggedError(last_error)
            retryable = (
                response.status_code in RETRY_STATUS
                or "3040" in last_error        # Out of Capacity
            )
            if not retryable:
                raise CloudflareError(last_error)

        if attempt < MAX_ATTEMPTS:
            wait = min(10 * 2 ** (attempt - 1), 90)
            print(f"تحذير: محاولة Cloudflare {attempt} فشلت؛ "
                  f"إعادة المحاولة بعد {wait} ثانية.")
            time.sleep(wait)

    raise CloudflareError(last_error)


def generate_image(
    prompt: str,
    references: list[Image.Image],
    width: int = 1280,
    height: int = 720,
    seed: int | None = None,
    model: str | None = None,
) -> Image.Image:
    """
    يولّد صورة بالأبعاد المطلوبة، أو يرفع CloudflareError. يدعم عدة حسابات (انظر accounts):
    يُستعمل الأول حتى تنتهي حصته اليومية أو يصير رمزه غير صالح، فيُنتقل إلى التالي.
    لا يُرفع CloudflareQuotaError (الذي يوقف سلالم المحاولات) إلا حين تتوقف كل الحسابات.
    رفض مرشّح المحتوى (CloudflareFlaggedError) يمر كما هو لأنه يتكرر في كل حساب.
    """
    pairs = accounts()
    if not pairs:
        raise CloudflareError(
            "المفتاحان CF_ACCOUNT_ID و CF_API_TOKEN غير موجودين في Secrets."
        )
    live = [pair for pair in pairs if not key_pool.is_dead(pair[1])]
    if not live:
        raise CloudflareQuotaError(
            f"كل حسابات Cloudflare ({len(pairs)}) متوقفة: انتهت الحصة اليومية "
            "أو الرمز غير صالح."
        )

    last: CloudflareError | None = None
    for number, (account, token) in enumerate(live, start=1):
        try:
            return _generate_image_once(
                prompt, references, width, height, seed, model, account, token
            )
        except CloudflareQuotaError as exc:
            last = exc
            key_pool.mark_dead(token, str(exc))
        except CloudflareFlaggedError:
            raise
        except CloudflareError as exc:
            if not key_pool.is_permanent(str(exc)):
                raise                       # خطأ غير متعلق بالحساب نفسه
            last = exc
            key_pool.mark_dead(token, str(exc))
        if number < len(live):
            print(f"::warning title=حساب Cloudflare::توقف الحساب {key_pool.mask(token)} "
                  f"({str(last)[:110]})؛ التبديل إلى الحساب التالي "
                  f"({number + 1} من {len(live)}).")
    raise last or CloudflareError("فشل توليد الصورة في كل حسابات Cloudflare.")
