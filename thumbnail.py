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


API_BASE = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"

MAX_REFERENCES = 4
REFERENCE_MAX_SIDE = 480        # الوثائق: يجب أن تكون الصور أصغر من 512×512
REQUEST_TIMEOUT = 150
MAX_ATTEMPTS = 2
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class CloudflareError(Exception):
    """خطأ في واجهة Cloudflare أو في الصورة المُعادة."""


def is_configured() -> bool:
    return bool(
        os.getenv("CF_ACCOUNT_ID", "").strip()
        and os.getenv("CF_API_TOKEN", "").strip()
    )


def model_name() -> str:
    return os.getenv("CF_IMAGE_MODEL", "").strip() or DEFAULT_MODEL


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


def generate_image(
    prompt: str,
    references: list[Image.Image],
    width: int = 1280,
    height: int = 720,
    seed: int | None = None,
) -> Image.Image:
    """يولّد صورة بالأبعاد المطلوبة، أو يرفع CloudflareError."""
    account = os.getenv("CF_ACCOUNT_ID", "").strip()
    token = os.getenv("CF_API_TOKEN", "").strip()
    if not account or not token:
        raise CloudflareError(
            "المفتاحان CF_ACCOUNT_ID و CF_API_TOKEN غير موجودين في Secrets."
        )

    url = f"{API_BASE}/{account}/ai/run/{model_name()}"
    headers = {"Authorization": f"Bearer {token}"}
    last_error = "فشل الاتصال بـ Cloudflare."

    for attempt in range(1, MAX_ATTEMPTS + 1):
        form = build_form(prompt, references, width, height, seed)
        try:
            response = requests.post(
                url, headers=headers, files=form, timeout=REQUEST_TIMEOUT
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
            retryable = (
                response.status_code in RETRY_STATUS
                or "3040" in last_error        # Out of Capacity
            )
            if not retryable:
                raise CloudflareError(last_error)

        if attempt < MAX_ATTEMPTS:
            wait = 6 * attempt
            print(f"تحذير: محاولة Cloudflare {attempt} فشلت؛ "
                  f"إعادة المحاولة بعد {wait} ثانية.")
            time.sleep(wait)

    raise CloudflareError(last_error)
