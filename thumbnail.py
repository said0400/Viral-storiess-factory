"""
الصورة الثانية (صورة المقال): صورة مصغرة احترافية 16:9 تُولَّد عبر Cloudflare
من الصور المرجعية، ثم يُكتب عليها عنوان الصورة.

إن فشل Cloudflare (أو لم يُضبط، أو كانت القصة حساسة) تُستخدم الصورة الأولى
(الدمج) كصورة للمقال ويُضاف عليها العنوان، وتبقى الصورة الأولى بلا نص.
"""
from __future__ import annotations

import os
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
from cloudflare_ai import CloudflareError

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
    items = [i for i in plan.get("images", [])
             if 0 <= i.get("index", -1) < len(images)
             and i.get("kind", "photo") == "photo"]
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

def _is_blank(image: Image.Image) -> bool:
    small = image.convert("L").resize((64, 36))
    return ImageStat.Stat(small).stddev[0] < 8.0


def _save(image, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "JPEG", quality=92, optimize=True)


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
    ينشئ thumbnail.jpg: Cloudflare أولًا، وإلا الصورة الأولى (الدمج) + العنوان.
    يعيد معلومات المصدر: source = cloudflare | collage_fallback.
    """
    info: dict[str, Any] = {"file": destination.name, "source": None,
                            "model": None, "reason": "", "text": None}

    skip_reason = ""
    if not _flag("CF_IMAGE_ENABLED", True):
        skip_reason = "الإنشاء عبر Cloudflare معطّل (CF_IMAGE_ENABLED)."
    elif sensitive and _flag("CF_SKIP_SENSITIVE", True):
        skip_reason = "قصة حساسة: تُخطّى الصورة المولّدة بالذكاء الاصطناعي."
    elif not cloudflare_ai.is_configured():
        skip_reason = "مفاتيح Cloudflare غير مضبوطة."

    if not skip_reason:
        try:
            refs = reference_images(images, plan)
            prompt = build_prompt(plan, len(refs))
            if prompt_file:
                prompt_file.parent.mkdir(parents=True, exist_ok=True)
                prompt_file.write_text(
                    f"model: {cloudflare_ai.model_name()}\n"
                    f"references: {len(refs)}\n\n{prompt}\n",
                    encoding="utf-8")

            generated = cloudflare_ai.generate_image(
                prompt, refs, THUMB_SIZE[0], THUMB_SIZE[1])
            if _is_blank(generated):
                raise CloudflareError("الصورة المولّدة فارغة تقريبًا.")

            zone = calmer_zone(generated)
            text_info = add_headline(generated, headline, highlight,
                                     preferred_zone=zone)
            _save(generated, destination)
            info.update(source="cloudflare", model=cloudflare_ai.model_name(),
                        text=text_info)
            return info
        except (CloudflareError, OSError, ValueError) as exc:
            info["reason"] = str(exc)
            print(f"::warning title=فشل Cloudflare::{exc}")
    else:
        info["reason"] = skip_reason
        print(f"تنبيه: {skip_reason}")

    # --- الاحتياط: الصورة الأولى (الدمج) كصورة للمقال مع العنوان ---
    if not collage or not Path(collage["path"]).is_file():
        raise ThumbnailError("لا توجد صورة دمج احتياطية لصنع صورة المقال.")

    with Image.open(collage["path"]) as source:
        base = source.convert("RGB")
    text_info = add_headline(base, headline, highlight,
                             keep=collage.get("keepouts", []))
    _save(base, destination)
    info.update(source="collage_fallback", text=text_info)
    return info
