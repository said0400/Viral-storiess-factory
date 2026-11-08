"""
صور القصة: القصص بلا صور مصدر، فتُولَّد صورة سينمائية داكنة بالذكاء الاصطناعي
(Cloudflare) بمقاس 4:5، ثم تُشتق منها:
  - facebook_image.jpg : 4:5 (أو 3:4) مع التدرج والعنوان والإطار الأبيض (fb_cover.add_overlay)
  - thumbnail.jpg      : 16:9 بقص أفقي للصورة نفسها مع العنوان، وتوضع أول المقال
  - video_frame.jpg    : (في work_dir) القص 16:9 نفسه بلا نص، لفيديو القصة

سلّم التوليد عند الرفض أو التعثر (نفس منطق الصورة المصغرة للأخبار):
  1) البرومبت الكامل  2) إعادة بنتيجة عشوائية مختلفة
  3) نموذج احتياطي    4) نموذج احتياطي بوصف آمن بلا عنف
انتهاء الحصة اليومية يوقف السلّم فورًا. وإن فشل كله تُرسم خلفية داكنة محليًا
(بلون التصنيف) كي تُنشر القصة ولا تتعطل بسبب الصورة.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat

import cloudflare_ai
import fb_cover
import story_categories
from cloudflare_ai import CloudflareError


BASE_SIZE = (1024, 1280)          # 4:5
BLOG_SIZE = (1280, 720)           # 16:9

STYLE_SUFFIX = (
    " Cinematic still, dark moody atmosphere, dramatic low-key lighting, strong "
    "contrast, shallow depth of field, photorealistic, rich detail. Keep the lower "
    "third of the frame darker and uncluttered so a headline can be added later. "
    "No text, no letters, no captions, no logos, no watermarks, no borders, no frames."
)


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}


def _is_blank(image: Image.Image, threshold: float = 6.0) -> bool:
    small = image.convert("L").resize((64, 64))
    return ImageStat.Stat(small).stddev[0] < threshold


def procedural_background(size: tuple[int, int], accent: tuple[int, int, int]) -> Image.Image:
    """خلفية داكنة سينمائية بلا إنترنت: تدرج بلون التصنيف + توهج + تعتيم الأطراف."""
    W, H = size
    top = tuple(int(c * 0.40) for c in accent)
    bottom = (7, 7, 12)
    ramp = Image.linear_gradient("L").resize((W, H))          # أسود أعلى → أبيض أسفل
    img = Image.composite(Image.new("RGB", size, bottom),
                          Image.new("RGB", size, top), ramp)

    glow = Image.new("RGB", size, (0, 0, 0))
    ImageDraw.Draw(glow).ellipse(
        (W * 0.12, H * 0.10, W * 0.88, H * 0.62),
        fill=tuple(int(c * 0.50) for c in accent))
    img = ImageChops.screen(img, glow.filter(ImageFilter.GaussianBlur(W * 0.14)))

    center = Image.new("L", size, 0)
    ImageDraw.Draw(center).ellipse((W * 0.05, H * 0.03, W * 0.95, H * 0.97), fill=255)
    center = center.filter(ImageFilter.GaussianBlur(W * 0.16))
    return Image.composite(img, Image.new("RGB", size, (0, 0, 0)), center)


def _prompts(story: dict[str, Any], info: dict[str, Any]) -> tuple[str, str]:
    scene = (story.get("image_prompt") or "").strip()
    if not scene:
        scene = info["safe_image"]
    return scene + STYLE_SUFFIX, info["safe_image"] + STYLE_SUFFIX


def generate_base(
    story: dict[str, Any], info: dict[str, Any]
) -> tuple[Image.Image | None, dict[str, Any]]:
    """يعيد (الصورة أو None، معلومات المحاولات). لا يرفع أخطاء Cloudflare."""
    notes: list[str] = []
    if not _flag("CF_IMAGE_ENABLED", True):
        return None, {"source": "procedural", "reason": "CF_IMAGE_ENABLED معطّل."}
    if not cloudflare_ai.is_configured():
        return None, {"source": "procedural", "reason": "مفاتيح Cloudflare غير مضبوطة."}

    main_prompt, safe_prompt = _prompts(story, info)
    fallback = cloudflare_ai.fallback_model_name()
    steps = [
        ("الأساسية", None, main_prompt),
        ("إعادة بنتيجة مختلفة", None, main_prompt),
        ("نموذج احتياطي", fallback, main_prompt),
        ("نموذج احتياطي بوصف آمن", fallback, safe_prompt),
    ]
    for number, (label, model, prompt) in enumerate(steps, start=1):
        seed = random.randint(1, 2**31 - 1)
        try:
            image = cloudflare_ai.generate_image(
                prompt, [], BASE_SIZE[0], BASE_SIZE[1], seed, model=model)
            if _is_blank(image):
                raise CloudflareError("الصورة المولّدة فارغة تقريبًا.")
        except cloudflare_ai.CloudflareQuotaError as exc:
            notes.append(f"{label}: انتهت الحصة اليومية")
            print("::warning title=حصة Cloudflare::انتهت الحصة اليومية؛ "
                  "تُرسم خلفية محلية.")
            return None, {"source": "procedural", "reason": " | ".join(notes) or str(exc)}
        except CloudflareError as exc:
            notes.append(f"{label}: {exc}")
            print(f"::warning title=تعذر توليد صورة القصة ({label})::{exc}")
            continue
        return image, {"source": "cloudflare", "model": model or cloudflare_ai.model_name(),
                       "attempt": number, "attempt_label": label,
                       "reason": " | ".join(notes)}
    return None, {"source": "procedural", "reason": " | ".join(notes)}


def build_story_images(
    story: dict[str, Any], category: str, out_dir: Path,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """
    ينشئ facebook_image.jpg و thumbnail.jpg في out_dir ويعيد معلومات عنهما.
    وإن حُدد work_dir يحفظ فيه video_frame.jpg: إطار 16:9 بلا نص لفيديو القصة
    (خارج output/ فلا يدخل المستودع).
    """
    info = story_categories.category_info(category)
    base, meta = generate_base(story, info)
    if base is None:
        base = procedural_background(BASE_SIZE, info["accent"])
        print(f"تنبيه: خلفية محلية لصورة القصة ({meta.get('reason', '')}).")
    base = base.convert("RGB")

    text_enabled = os.getenv("ADD_IMAGE_TEXT", "").strip().lower() not in {
        "0", "false", "no", "off"}
    headline = story["image_title"] if text_enabled else None
    highlight = story.get("image_highlight") or ""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) فيسبوك: 4:5 (أو 3:4) كاملة الإطار
    fb_size = (fb_cover.IMAGE_SIZE, fb_cover.facebook_height())
    picture = ImageOps.fit(base, fb_size, Image.Resampling.LANCZOS)
    fb_image, fb_text = fb_cover.add_overlay(picture, headline, highlight)
    fb_image.save(out_dir / "facebook_image.jpg", "JPEG", quality=93, optimize=True)

    # 2) المدونة: شريط أفقي 16:9 من الصورة نفسها (أعلى قليلًا من الوسط)
    bw, bh = base.size
    crop_h = min(bh, round(bw * BLOG_SIZE[1] / BLOG_SIZE[0]))
    y0 = int((bh - crop_h) * 0.38)
    wide = base.crop((0, y0, bw, y0 + crop_h)).resize(BLOG_SIZE, Image.Resampling.LANCZOS)
    wide_image, wide_text = fb_cover.add_overlay(wide, headline, highlight)
    wide_image.save(out_dir / "thumbnail.jpg", "JPEG", quality=92, optimize=True)

    # 3) إطار الفيديو: القص نفسه بلا نص، بدقة أعلى ليتحمل التحريك
    video_frame = None
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        video_frame = work_dir / "video_frame.jpg"
        wide.resize((1920, 1080), Image.Resampling.LANCZOS).save(
            video_frame, "JPEG", quality=95)

    return {
        "facebook": {"file": "facebook_image.jpg", "size": list(fb_size), "text": fb_text},
        "thumbnail": {"file": "thumbnail.jpg", "size": list(BLOG_SIZE), "text": wide_text,
                      "source": meta.get("source"), "model": meta.get("model"),
                      "reason": meta.get("reason", "")},
        "video_frame": str(video_frame) if video_frame else None,
        "generation": meta,
    }
