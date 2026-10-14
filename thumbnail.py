"""
الصورة الثانية (صورة المقال): صورة مصغرة احترافية 16:9.

الوضع الافتراضي (hybrid): Cloudflare يولّد الخلفية والأجواء فقط (بلا أشخاص)،
وتُقص الوجوه الحقيقية من صور المقال وتوضع فوقها، ثم يُكتب العنوان. فتبقى
الملامح حقيقية. إن تعذّر Cloudflare تُستخدم خلفية مولّدة محليًا.
THUMBNAIL_MODE=ai_full يعيد التوليد الكامل من الصور المرجعية (قد يغيّر الملامح)،
ويتدرّج عبر 5 محاولات بالذكاء الاصطناعي (_try_ai_full_ladder) قبل أي احتياط محلي.

إن تعذّر القص كله تُستخدم الصورة الأولى (الدمج) كصورة للمقال مع العنوان،
وتبقى الصورة الأولى بلا نص.
"""
from __future__ import annotations

import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
    ImageFont,
    ImageOps,
    ImageStat,
)

import cloudflare_ai
import thumbnail_design as design
from cloudflare_ai import CloudflareError
from grok import default_gallery

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_TEXT_OK = True
except ImportError:
    ARABIC_TEXT_OK = False


THUMB_SIZE = (1280, 720)
FONT_FILENAME = "IBMPlexSansArabic-Bold.ttf"
FONT_FALLBACKS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",)
TEXT_ACCENT = "#FFD400"


class ThumbnailError(Exception):
    """خطأ متوقع أثناء إنشاء الصورة المصغرة."""


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# كتابة العنوان على صورة 16:9
# ---------------------------------------------------------------------------

def find_font_path():
    candidates = [os.getenv("FONT_PATH", "").strip()]
    here = Path(__file__).resolve().parent
    candidates += [here / "fonts" / FONT_FILENAME,
                   Path("fonts") / FONT_FILENAME, *FONT_FALLBACKS]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def clean_headline(text):
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]", "",
                  text or "")
    text = text.replace("#", " ").replace("«", "").replace("»", "")
    text = text.replace('"', "").replace("“", "").replace("”", "")
    text = "".join(ch for ch in text
                   if not unicodedata.category(ch).startswith("C") or ch == " ")
    return re.sub(r"\s+", " ", text).strip()


def shape_word(word):
    return get_display(arabic_reshaper.reshape(word), base_dir="R")


def layout_headline(words, font_path, width):
    """أكبر خط يتسع في سطرين (ثم ثلاثة) لصورة عرضها width."""
    max_w = width * 0.90
    shaped = [shape_word(w) for w in words]
    result = None

    for max_lines, min_frac in ((2, 0.046), (3, 0.038)):
        for size in range(int(width * 0.064), int(width * min_frac) - 1, -2):
            font = ImageFont.truetype(
                font_path, size, layout_engine=ImageFont.Layout.BASIC)
            space = font.getlength(" ") + size * 0.10
            widths = [font.getlength(s) for s in shaped]

            lines, line_widths, cur, cur_w = [], [], [], 0.0
            for i, wd in enumerate(widths):
                add = wd if not cur else space + wd
                if cur and cur_w + add > max_w:
                    lines.append(cur)
                    line_widths.append(cur_w)
                    cur, cur_w = [i], wd
                else:
                    cur.append(i)
                    cur_w += add
            if cur:
                lines.append(cur)
                line_widths.append(cur_w)

            result = {"font": font, "size": size, "space": space,
                      "shaped": shaped, "widths": widths,
                      "lines": lines, "line_widths": line_widths}
            if len(lines) <= max_lines and max(line_widths) <= max_w:
                return result
    return result


def zone_penalty(zone, keep):
    zx0, zy0, zx1, zy1 = zone
    total = 0.0
    for (bx0, by0, bx1, by1), weight in keep:
        area = max(1.0, (bx1 - bx0) * (by1 - by0))
        ow = max(0.0, min(zx1, bx1) - max(zx0, bx0))
        oh = max(0.0, min(zy1, by1) - max(zy0, by0))
        total += weight * ow * oh / area
    return total


def calmer_zone(image: Image.Image) -> str:
    """الشريط الأقل تفاصيلًا (أعلى أو أسفل) لوضع النص فيه."""
    gray = image.convert("L").resize((320, 180))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    top = ImageStat.Stat(edges.crop((0, 0, 320, 54))).mean[0]
    bottom = ImageStat.Stat(edges.crop((0, 126, 320, 180))).mean[0]
    return "bottom" if bottom <= top * 1.15 else "top"


def add_headline(canvas, headline, highlight, keep=None, preferred_zone=None):
    """يكتب العنوان على canvas مباشرة ويعيد معلومات عنه، أو None."""
    if not ARABIC_TEXT_OK:
        print("::warning title=النص العربي::مكتبتا arabic-reshaper و "
              "python-bidi غير مثبتتين؛ تُخطي الكتابة على الصورة.")
        return None
    font_path = find_font_path()
    if not font_path:
        print("::warning title=النص العربي::لم يُعثر على خط عربي.")
        return None

    words = clean_headline(headline).split()[:24]
    if not words:
        return None

    W, H = canvas.size
    lay = layout_headline(words, font_path, W)
    if lay is None:
        return None

    size = lay["size"]
    line_h = int(size * 1.28)
    pad = int(size * 0.55)
    n_lines = len(lay["lines"])
    block_h = line_h * n_lines + pad * 2

    zones = {"bottom": (0, H - block_h, W, H), "top": (0, 0, W, block_h)}
    if preferred_zone in zones:
        zone_name = preferred_zone
    else:
        zone_name = min(("bottom", "top"),
                        key=lambda z: zone_penalty(zones[z], keep or []))
    zx0, zy0, zx1, zy1 = zones[zone_name]

    gh = int(block_h * 1.5)
    alpha = Image.linear_gradient("L").resize((W, gh))
    alpha = alpha.point(lambda v: int(v * 0.88))
    if zone_name == "bottom":
        canvas.paste((0, 0, 0), (0, H - gh, W, H), alpha)
    else:
        canvas.paste((0, 0, 0), (0, 0, W, gh), ImageOps.flip(alpha))

    draw = ImageDraw.Draw(canvas)
    bar_w, bar_h = int(W * 0.12), max(5, int(size * 0.12))
    bar_y = (zy0 + pad - int(size * 0.30) - bar_h if zone_name == "bottom"
             else zy1 - pad + int(size * 0.30))
    draw.rounded_rectangle(((W - bar_w) // 2, bar_y, (W + bar_w) // 2,
                            bar_y + bar_h), radius=bar_h // 2,
                           fill=TEXT_ACCENT)

    def norm(word):
        return re.sub(r"[^\w]", "", word, flags=re.UNICODE)

    marked = {norm(w) for w in clean_headline(highlight).split()} - {""}
    stroke = max(3, size // 16)
    y = zy0 + pad

    for line, line_w in zip(lay["lines"], lay["line_widths"]):
        x = (W + line_w) / 2
        for idx in line:
            x -= lay["widths"][idx]
            color = TEXT_ACCENT if norm(words[idx]) in marked else "white"
            draw.text((x, y), lay["shaped"][idx], font=lay["font"],
                      fill=color, stroke_width=stroke, stroke_fill="black")
            x -= lay["space"]
        y += line_h

    return {"text": " ".join(words), "zone": zone_name,
            "font": Path(font_path).name, "font_size": size,
            "lines": n_lines}


# ---------------------------------------------------------------------------
# الصور المرجعية والبرومبت
# ---------------------------------------------------------------------------

def reference_images(images, plan, limit=4):
    """مقتطعات حول العنصر الرئيسي من الصور الفوتوغرافية التي اختارها التحليل."""
    def valid(seq):
        return [i for i in seq
                if 0 <= i.get("index", -1) < len(images)
                and i.get("kind", "photo") == "photo"
                and i.get("subject_box")]

    # صور المعرض الفوتوغرافية أولًا (أوسع)، ثم صور الخطة الرئيسية
    items = valid(plan.get("gallery", [])) or valid(plan.get("images", []))
    if not items:
        return [images[i] for i in range(min(limit, len(images)))]

    refs = []
    for item in items[:limit]:
        img = images[item["index"]]
        W, H = img.size
        l, t, r, b = item["subject_box"]
        padx, pady = (r - l) * 0.08, (b - t) * 0.08
        box = (max(0, round((l - padx) * W)), max(0, round((t - pady) * H)),
               min(W, round((r + padx) * W)), min(H, round((b + pady) * H)))
        if box[2] - box[0] < 32 or box[3] - box[1] < 32:
            refs.append(img)
        else:
            refs.append(img.crop(box))
    return refs


PROMPT_TEMPLATE = (
    "Professional YouTube thumbnail, 16:9 widescreen. Compose one striking "
    "image using the people and objects from the reference photos{refs}. "
    "Keep every person's face, identity, age, skin tone, hair and clothing "
    "exactly as in the references: do not change, beautify or invent faces, "
    "people or objects. {brief}"
    "Cinematic lighting, strong contrast, vivid but natural saturated colors, "
    "crisp focus on the main subject, soft blurred background, a clear focal "
    "point following the rule of thirds, subtle vignette. Keep the lower "
    "fifth of the frame calm and uncluttered so a headline can be added "
    "later. Photorealistic. No text, no letters, no captions, no logos, no "
    "watermarks, no borders, no frames."
)


def build_prompt(plan, n_refs):
    refs = ""
    if n_refs > 1:
        refs = f" (image 0 is the main subject; images 1 to {n_refs - 1} " \
               "are supporting subjects)"
    brief = (plan.get("thumbnail_brief") or "").strip()
    brief = (brief + " ") if brief else ""
    return PROMPT_TEMPLATE.format(refs=refs, brief=brief)


# ---------------------------------------------------------------------------
# الواجهة العامة
# ---------------------------------------------------------------------------

def _is_blank(image: Image.Image, threshold: float = 8.0) -> bool:
    small = image.convert("L").resize((64, 36))
    return ImageStat.Stat(small).stddev[0] < threshold


def _save(image, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "JPEG", quality=92, optimize=True)


BACKGROUND_PROMPT = (
    "Cinematic dramatic background plate for a YouTube thumbnail, 16:9 "
    "widescreen. {setting}Dark moody atmosphere with {palette} rim light and "
    "soft glow, shallow depth of field, bokeh, volumetric light rays, rich "
    "contrast. Keep the center and the lower area wide and open for "
    "foreground subjects and a headline. No people, no faces, no figures, no "
    "hands, no text, no letters, no logos, no watermarks. Photorealistic, "
    "high detail."
)


def build_background_prompt(plan, color_name: str) -> str:
    setting = (plan.get("thumbnail_background") or "").strip()
    setting = (setting + " ") if setting else "An abstract moody studio backdrop. "
    palette = f"{color_name} with a contrasting warm accent"
    return BACKGROUND_PROMPT.format(setting=setting, palette=palette)


def _cloudflare_skip_reason(sensitive: bool) -> str:
    if not _flag("CF_IMAGE_ENABLED", True):
        return "الإنشاء عبر Cloudflare معطّل (CF_IMAGE_ENABLED)."
    if sensitive and _flag("CF_SKIP_SENSITIVE", False):
        return "قصة حساسة: تُخطّى الصورة المولّدة بالذكاء الاصطناعي."
    if not cloudflare_ai.is_configured():
        return "مفاتيح Cloudflare غير مضبوطة."
    return ""


def _gallery_items(images, plan):
    items = plan.get("gallery") or default_gallery(len(images))
    return [i for i in items
            if 0 <= i.get("index", -1) < len(images) and i.get("subject_box")]


def _try_hybrid(images, plan, headline, highlight, destination,
                prompt_file, cf_skip) -> dict[str, Any]:
    """خلفية (Cloudflare أو محلية) + وجوه حقيقية مقصوصة + العنوان."""
    items = _gallery_items(images, plan)
    people, card = design.extract_subjects(images, items)
    if not people and not card:
        raise ThumbnailError("لم يُقص أي عنصر من الصور.")

    used = [images[i["index"]] for i in items]
    accent, color_name = design.dominant_accent(used)

    background, bg_source, bg_reason = None, "local", ""
    if cf_skip:
        bg_reason = cf_skip
    else:
        try:
            prompt = build_background_prompt(plan, color_name)
            if prompt_file:
                prompt_file.parent.mkdir(parents=True, exist_ok=True)
                prompt_file.write_text(
                    f"mode: hybrid (background only)\n"
                    f"model: {cloudflare_ai.model_name()}\n\n{prompt}\n",
                    encoding="utf-8")
            generated = cloudflare_ai.generate_image(
                prompt, [], THUMB_SIZE[0], THUMB_SIZE[1])
            if _is_blank(generated, threshold=4.0):   # خلفية داكنة مقبولة
                raise CloudflareError("الخلفية المولّدة فارغة تقريبًا.")
            background = design.prepare_background(generated, THUMB_SIZE)
            bg_source = "cloudflare"
        except (CloudflareError, OSError, ValueError) as exc:
            bg_reason = str(exc)
            print(f"::warning title=فشل خلفية Cloudflare::{exc}")

    if background is None:
        background = design.procedural_background(THUMB_SIZE, accent)

    canvas, keep = design.compose(background, people, card, accent)
    text_info = add_headline(canvas, headline, highlight, keep=keep)
    _save(canvas, destination)

    return {
        "file": destination.name,
        "source": "hybrid_cloudflare" if bg_source == "cloudflare"
        else "hybrid_local",
        "background": bg_source,
        "model": cloudflare_ai.model_name() if bg_source == "cloudflare" else None,
        "reason": bg_reason,
        "subjects": {"people": len(people), "card": bool(card)},
        "text": text_info,
    }


def _try_ai_full(images, plan, headline, highlight, destination,
                 prompt_file, cf_skip) -> dict[str, Any]:
    """الوضع القديم: توليد كامل من صور مرجعية (قد يغيّر الملامح)."""
    if cf_skip:
        raise ThumbnailError(cf_skip)

    refs = reference_images(images, plan)
    prompt = build_prompt(plan, len(refs))
    if prompt_file:
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(
            f"mode: ai_full\nmodel: {cloudflare_ai.model_name()}\n"
            f"references: {len(refs)}\n\n{prompt}\n", encoding="utf-8")

    generated = cloudflare_ai.generate_image(
        prompt, refs, THUMB_SIZE[0], THUMB_SIZE[1])
    if _is_blank(generated):
        raise CloudflareError("الصورة المولّدة فارغة تقريبًا.")

    text_info = add_headline(generated, headline, highlight,
                             preferred_zone=calmer_zone(generated))
    _save(generated, destination)
    return {"file": destination.name, "source": "cloudflare_full",
            "model": cloudflare_ai.model_name(), "reason": "",
            "text": text_info}


# ---------------------------------------------------------------------------
# التوليد الكامل بالذكاء الاصطناعي دائمًا: تدرّج في المحاولات قبل أي احتياط
# ---------------------------------------------------------------------------

SCENE_PROMPT = (
    "Professional YouTube thumbnail, 16:9 widescreen, dramatic cinematic "
    "composition. {brief}{setting}Strong contrast, vivid saturated colors, "
    "soft blurred background, shallow depth of field, a clear focal point "
    "following the rule of thirds, subtle vignette. Keep the lower fifth of "
    "the frame calm and uncluttered so a headline can be added later. "
    "Photorealistic. No text, no letters, no captions, no logos, no "
    "watermarks, no borders, no frames."
)


def build_scene_prompt(plan, headline: str = "") -> str:
    """وصف مشهد بلا صور مرجعية: من وصف التحليل البصري، أو من عنوان القصة."""
    brief = (plan.get("thumbnail_brief") or "").strip()
    setting = (plan.get("thumbnail_background") or "").strip()
    if not brief and not setting and headline:
        brief = f"Story theme (Arabic headline, for context only): {headline}."
    brief = (brief + " ") if brief else ""
    setting = (setting + " ") if setting else ""
    return SCENE_PROMPT.format(brief=brief, setting=setting)


def _try_ai_full_ladder(images, plan, headline, highlight, destination,
                        prompt_file, cf_skip) -> dict[str, Any]:
    """
    يولّد الصورة كاملة بالذكاء الاصطناعي ويتدرّج عند الرفض أو التعثر:
      1) الصور المرجعية + البرومبت الأساسي
      2) إعادة بنتيجة عشوائية مختلفة (مرشّح المحتوى يخطئ أحيانًا)
      3) وصف المشهد بلا صور مرجعية
      4) نموذج Cloudflare احتياطي (FLUX.1 schnell)
      5) النموذج الاحتياطي بوصف آمن بلا أشخاص
    انتهاء الحصة اليومية يوقف التدرّج فورًا (كل النماذج تتشارك الحصة)،
    فيرفع CloudflareQuotaError ليتولى create_thumbnail الاحتياط المحلي.
    """
    if cf_skip:
        raise ThumbnailError(cf_skip)

    refs = reference_images(images, plan)
    main_prompt = build_prompt(plan, len(refs))
    scene_prompt = build_scene_prompt(plan, headline)
    safe_prompt = build_background_prompt(plan, "deep blue")
    fallback_model = cloudflare_ai.fallback_model_name()

    steps = [
        ("المحاولة الأساسية بالصور المرجعية", None, refs, main_prompt),
        ("إعادة بنتيجة عشوائية مختلفة", None, refs, main_prompt),
        ("وصف المشهد بلا صور مرجعية", None, [], scene_prompt),
        ("نموذج احتياطي", fallback_model, [], scene_prompt),
        ("نموذج احتياطي بوصف آمن", fallback_model, [], safe_prompt),
    ]

    errors: list[str] = []
    for number, (label, model, step_refs, prompt) in enumerate(steps, start=1):
        seed = random.randint(1, 2**31 - 1)
        try:
            generated = cloudflare_ai.generate_image(
                prompt, step_refs, THUMB_SIZE[0], THUMB_SIZE[1], seed,
                model=model)
            if _is_blank(generated):
                raise CloudflareError("الصورة المولّدة فارغة تقريبًا.")
        except cloudflare_ai.CloudflareQuotaError:
            print("::warning title=حصة Cloudflare::انتهت الحصة اليومية؛ "
                  "يُستعمل الاحتياط المحلي.")
            raise
        except CloudflareError as exc:
            errors.append(f"{label}: {exc}")
            print(f"::warning title=تعذر التوليد ({label})::{exc}")
            continue

        if prompt_file:
            prompt_file.parent.mkdir(parents=True, exist_ok=True)
            prompt_file.write_text(
                f"mode: ai_full (محاولة {number}: {label})\n"
                f"model: {model or cloudflare_ai.model_name()}\n"
                f"references: {len(step_refs)}\n\n{prompt}\n",
                encoding="utf-8")

        text_info = add_headline(generated, headline, highlight,
                                 preferred_zone=calmer_zone(generated))
        _save(generated, destination)
        return {"file": destination.name, "source": "cloudflare_full",
                "model": model or cloudflare_ai.model_name(),
                "attempt": number, "attempt_label": label,
                "reason": " | ".join(errors), "text": text_info}

    raise ThumbnailError(
        "فشلت كل محاولات التوليد بالذكاء الاصطناعي: " + " | ".join(errors))


def create_thumbnail(
    images: list[Image.Image],
    plan: dict[str, Any],
    headline: str,
    highlight: str,
    collage: dict[str, Any] | None,
    destination: Path,
    prompt_file: Path | None = None,
    sensitive: bool = False,
) -> dict[str, Any]:
    """
    ينشئ thumbnail.jpg. source: hybrid_cloudflare | hybrid_local |
    cloudflare_full | collage_fallback.
    """
    mode = os.getenv("THUMBNAIL_MODE", "").strip().lower() or "hybrid"
    cf_skip = _cloudflare_skip_reason(sensitive)
    attempt = _try_ai_full_ladder if mode == "ai_full" else _try_hybrid
    reason = ""

    try:
        return attempt(images, plan, headline, highlight, destination,
                       prompt_file, cf_skip)
    except (ThumbnailError, CloudflareError, design.CutoutError,
            OSError, ValueError) as exc:
        reason = str(exc)
        print(f"::warning title=تعذر تصميم الصورة المصغرة::{exc}")
    except Exception as exc:  # أي عطل غير متوقع لا يوقف المشروع
        reason = f"{type(exc).__name__}: {exc}"
        print(f"::warning title=عطل غير متوقع في الصورة المصغرة::{reason}")

    # --- الاحتياط: الصورة الأولى (الدمج) كصورة للمقال مع العنوان ---
    if not collage or not Path(collage["path"]).is_file():
        raise ThumbnailError("لا توجد صورة دمج احتياطية لصنع صورة المقال.")

    with Image.open(collage["path"]) as source:
        base = source.convert("RGB")
    text_info = add_headline(base, headline, highlight,
                             keep=collage.get("keepouts", []))
    _save(base, destination)
    return {"file": destination.name, "source": "collage_fallback",
            "model": None, "reason": reason, "text": text_info}
