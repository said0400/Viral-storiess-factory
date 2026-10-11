"""
تصميم الصورة المصغرة بالكود من الصور الحقيقية:
خلفية (من Cloudflare أو مولّدة محليًا) + أشخاص مقصوصون من صور المقال
+ بطاقة للقطات الشاشة + توهج وحدود وظل. الوجوه تبقى حقيقية دون تغيير.
"""
from __future__ import annotations

import colorsys
import os
import random
from typing import Any

import numpy as np
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)


class CutoutError(Exception):
    """تعذر قص عنصر من صورته."""


PEOPLE_MODEL = "u2net_human_seg"      # أفضل للأشخاص والشعر
OBJECT_MODEL = "isnet-general-use"    # أفضل للأشياء
_SESSIONS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# القص
# ---------------------------------------------------------------------------

def _session(model: str):
    if model not in _SESSIONS:
        try:
            from rembg import new_session
        except ImportError as exc:
            raise CutoutError("مكتبة rembg غير مثبتة.") from exc
        try:
            _SESSIONS[model] = new_session(model)
        except Exception as exc:  # فشل تنزيل النموذج أو تحميله
            raise CutoutError(f"تعذر تحميل نموذج القص {model}: {exc}") from exc
    return _SESSIONS[model]


def _model_for(item: dict[str, Any]) -> str:
    forced = os.getenv("REMBG_MODEL", "").strip()
    if forced:
        return forced
    return PEOPLE_MODEL if item.get("avoid_boxes") else OBJECT_MODEL


def clean_alpha(rgba: Image.Image, open_px: int = 3, feather: float = 1.2):
    """يزيل الشظايا المنفصلة ويسد الثقوب ويُنعّم الحواف."""
    try:
        from scipy import ndimage as ndi
    except ImportError:
        return rgba

    solid = np.array(rgba.getchannel("A")) > 127
    opened = ndi.binary_opening(
        solid, structure=np.ones((open_px * 2 + 1,) * 2)
    )
    labels, count = ndi.label(opened)
    if count:
        sizes = ndi.sum(opened, labels, range(1, count + 1))
        main = labels == (1 + int(np.argmax(sizes)))
        # نعيد حواف الشعر الرفيعة الملاصقة للجسم فقط
        keep = ndi.binary_dilation(main, iterations=open_px + 2) & solid
    else:
        keep = solid
    keep = ndi.binary_fill_holes(keep)

    mask = Image.fromarray((keep * 255).astype("uint8")).filter(
        ImageFilter.GaussianBlur(feather)
    )
    out = rgba.copy()
    out.putalpha(mask)
    return out


def fade_borders(rgba: Image.Image) -> Image.Image:
    """
    الحواف التي يلامسها الشخص عند حدود الصورة الأصلية (كتف أو شعر خارج الإطار)
    تظهر كخط مستقيم قبيح؛ نُخفيها بتدرج. الحافة السفلية تُترك لأنها تلتصق
    بأسفل الصورة المصغرة.
    """
    alpha = np.array(rgba.getchannel("A")).astype("float32")
    h, w = alpha.shape
    band_x, band_y = max(40, int(w * 0.14)), max(30, int(h * 0.07))

    def touches(edge: np.ndarray) -> bool:
        return float((edge > 127).mean()) > 0.15

    ramp = np.ones_like(alpha)
    if touches(alpha[:, 0]):
        ramp *= np.clip(np.arange(w) / band_x, 0, 1)[None, :]
    if touches(alpha[:, -1]):
        ramp *= np.clip((w - 1 - np.arange(w)) / band_x, 0, 1)[None, :]
    if touches(alpha[0, :]):
        ramp *= np.clip(np.arange(h) / band_y, 0, 1)[:, None]

    out = rgba.copy()
    out.putalpha(Image.fromarray((alpha * ramp).astype("uint8")))
    return out


def bust_box(item: dict[str, Any], W: int, H: int):
    """مقتطع من الرأس حتى الصدر إن عُرف موضع الوجه، وإلا العنصر كله."""
    sb = item["subject_box"]
    sx0, sy0, sx1, sy1 = sb[0] * W, sb[1] * H, sb[2] * W, sb[3] * H

    face = None
    for b in item.get("avoid_boxes", []):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if sb[0] <= cx <= sb[2] and sb[1] <= cy <= sb[3]:
            face = (b[0] * W, b[1] * H, b[2] * W, b[3] * H)
            break

    if face:
        fx0, fy0, fx1, fy1 = face
        fw, fh = fx1 - fx0, fy1 - fy0
        cx = (fx0 + fx1) / 2
        half = 2.0 * max(fw, 0.8 * fh)
        box = [cx - half, fy0 - 0.8 * fh, cx + half, fy1 + 2.4 * fh]
        mx, my = 0.06 * (sx1 - sx0), 0.06 * (sy1 - sy0)
        box = [max(box[0], sx0 - mx), max(box[1], sy0 - my),
               min(box[2], sx1 + mx), min(box[3], sy1 + my)]
    else:
        box = [sx0, sy0, sx1, sy1]

    box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
    box = tuple(int(round(v)) for v in box)
    if box[2] - box[0] < 80 or box[3] - box[1] < 80:
        raise CutoutError("منطقة العنصر صغيرة جدًا للقص.")
    return box, face


def cut_subject(image: Image.Image, item: dict[str, Any]) -> dict[str, Any]:
    """يقص الشخص أو الشيء. يعيد {"image": RGBA مقصوص، "face": صندوق|None}."""
    try:
        from rembg import remove
    except ImportError as exc:
        raise CutoutError("مكتبة rembg غير مثبتة.") from exc

    W, H = image.size
    box, face = bust_box(item, W, H)
    crop = image.convert("RGB").crop(box)

    try:
        raw = remove(crop, session=_session(_model_for(item)),
                     post_process_mask=True)
    except CutoutError:
        raise
    except Exception as exc:
        raise CutoutError(f"فشل القص: {exc}") from exc

    cut = clean_alpha(raw.convert("RGBA"))
    alpha = np.array(cut.getchannel("A"))
    coverage = float((alpha > 127).mean())
    if not 0.06 <= coverage <= 0.97:
        raise CutoutError(f"قناع القص غير معقول (تغطية {coverage:.0%}).")

    rel_face = None
    if face:
        rel_face = (face[0] - box[0], face[1] - box[1],
                    face[2] - box[0], face[3] - box[1])
        fx, fy = int((rel_face[0] + rel_face[2]) / 2), int((rel_face[1] + rel_face[3]) / 2)
        if not (0 <= fy < alpha.shape[0] and 0 <= fx < alpha.shape[1]
                and alpha[fy, fx] > 200):
            raise CutoutError("القص فقد الوجه؛ يُتخطى هذا العنصر.")

    bbox = cut.getchannel("A").point(lambda v: 255 if v > 10 else 0).getbbox()
    if not bbox:
        raise CutoutError("القناع فارغ.")
    cut = fade_borders(cut.crop(bbox))
    if rel_face:
        rel_face = (rel_face[0] - bbox[0], rel_face[1] - bbox[1],
                    rel_face[2] - bbox[0], rel_face[3] - bbox[1])
    return {"image": cut, "face": rel_face}


def extract_subjects(images, items, max_people: int = 2):
    """(قائمة الأشخاص المقصوصين، بطاقة لقطة الشاشة أو None)."""
    people: list[dict[str, Any]] = []
    card = None
    for item in items:
        if not 0 <= item.get("index", -1) < len(images):
            continue
        image = images[item["index"]]
        kind = item.get("kind", "photo")
        if kind == "screenshot":
            if card is None:
                card = {"image": image}
            continue
        if kind == "photo" and len(people) < max_people:
            try:
                people.append(cut_subject(image, item))
            except CutoutError as exc:
                print(f"تنبيه: تخطي عنصر في الصورة المصغرة: {exc}")
    return people, card


# ---------------------------------------------------------------------------
# الألوان والخلفية
# ---------------------------------------------------------------------------

def _hue_name(h: float) -> str:
    deg = h * 360
    for limit, name in ((15, "red"), (45, "orange"), (70, "yellow"),
                        (165, "green"), (200, "teal"), (260, "blue"),
                        (300, "purple"), (345, "magenta")):
        if deg < limit:
            return name
    return "red"


def dominant_accent(images: list[Image.Image]):
    """
    لون التمييز: أبرز لون مشبع في الصور. وإن كان لون بشرة (برتقالي)
    نأخذ المتمم (أزرق مخضر) على طريقة الألوان السينمائية.
    تعيد ((r,g,b), اسم_اللون).
    """
    hist = np.zeros(36)
    for img in images:
        hsv = np.array(img.convert("RGB").resize((64, 64)).convert("HSV")) / 255.0
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        weight = np.where((s > 0.25) & (v > 0.25), s * v, 0.0)
        bins = np.minimum((h * 36).astype(int), 35)
        np.add.at(hist, bins.ravel(), weight.ravel())

    hue = (int(np.argmax(hist)) + 0.5) / 36 if hist.sum() > 0 else 0.6
    if 0.02 <= hue <= 0.14:               # لون بشرة
        hue = (hue + 0.5) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255)), _hue_name(hue)


def _complement(rgb):
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    r, g, b = colorsys.hsv_to_rgb((h + 0.5) % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _falloff(w: int, h: int, power: float = 2.0) -> np.ndarray:
    """قناع بيضاوي: 1 في المركز و0 عند الحافة بلا حدود ظاهرة."""
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - (w - 1) / 2) / (w / 2)) ** 2
                + ((yy - (h - 1) / 2) / (h / 2)) ** 2)
    return np.clip(1.0 - d, 0.0, 1.0) ** power


def _glow(canvas: Image.Image, cx, cy, rx, ry, color, strength):
    w, h = max(2, int(rx * 2)), max(2, int(ry * 2))
    mask = Image.fromarray((_falloff(w, h) * strength * 255).astype("uint8"))
    canvas.paste(Image.new("RGB", (w, h), color),
                 (int(cx - rx), int(cy - ry)), mask)


def _vignette(img: Image.Image, strength: float = 0.85) -> Image.Image:
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - (w - 1) / 2) / (w / 2)) ** 2
                + ((yy - (h - 1) / 2) / (h / 2)) ** 2)
    alpha = np.clip((d - 0.62) / 0.75, 0.0, 1.0) ** 1.6 * strength * 255
    out = img.copy()
    out.paste((0, 0, 0), mask=Image.fromarray(alpha.astype("uint8")))
    return out


def procedural_background(size, accent, seed: int = 11) -> Image.Image:
    """خلفية داكنة سينمائية: تدرج + توهجان + دوائر ضبابية + حبيبات."""
    W, H = size
    rnd = random.Random(seed)

    y = np.linspace(0, 1, H)[:, None, None]
    top = np.array([c * 0.34 for c in accent])[None, None, :]
    bottom = np.array([9, 9, 16])[None, None, :]
    arr = (top * (1 - y) + bottom * y) * np.ones((1, W, 1))
    img = Image.fromarray(arr.clip(0, 255).astype("uint8"), "RGB")

    comp = _complement(accent)
    _glow(img, W * 0.24, H * 0.55, W * 0.50, H * 0.85, accent, 0.80)
    _glow(img, W * 0.80, H * 0.42, W * 0.46, H * 0.80, comp, 0.55)

    bokeh = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(bokeh)
    for _ in range(18):
        r = rnd.randint(18, 70)
        x, yy = rnd.randint(0, W), rnd.randint(0, int(H * 0.8))
        color = rnd.choice([accent, comp, (255, 255, 255)])
        dim = rnd.uniform(0.10, 0.28)
        draw.ellipse((x - r, yy - r, x + r, yy + r),
                     fill=tuple(int(c * dim) for c in color))
    bokeh = bokeh.filter(ImageFilter.GaussianBlur(7))
    img = ImageChops.screen(img, bokeh)

    noise = Image.effect_noise(size, 22).convert("RGB")
    return Image.blend(img, noise, 0.035)


def prepare_background(bg: Image.Image, size) -> Image.Image:
    """يهيّئ خلفية Cloudflare: المقاس، تعتيم خفيف، وضبابية بسيطة."""
    bg = ImageOps.fit(bg.convert("RGB"), size, Image.Resampling.LANCZOS)
    luminance = np.array(bg.convert("L")).mean()
    if luminance > 105:
        bg = ImageEnhance.Brightness(bg).enhance(105 / luminance)
    return bg.filter(ImageFilter.GaussianBlur(1.6))


# ---------------------------------------------------------------------------
# التركيب
# ---------------------------------------------------------------------------

def _grade(rgb: Image.Image) -> Image.Image:
    rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
    rgb = ImageEnhance.Color(rgb).enhance(1.12)
    return ImageEnhance.Sharpness(rgb).enhance(1.25)


def _draw_person(canvas, sub, cx, target_h, accent, keep):
    W, H = canvas.size
    img = sub["image"]
    scale = min(target_h / img.height, 0.60 * W / img.width)
    w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    person = img.resize((w, h), Image.Resampling.LANCZOS)

    alpha = person.getchannel("A")
    rgb = _grade(person.convert("RGB"))

    x = int(min(max(cx - w / 2, -0.12 * w), W - 0.88 * w))
    y = int(H - h + 0.02 * h)

    _glow(canvas, x + w / 2, y + h * 0.40, w * 0.95, h * 0.80, accent, 0.85)

    shadow = Image.new("L", canvas.size, 0)
    shadow.paste(alpha.point(lambda v: int(v * 0.65)), (x + 10, y + 14))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    canvas.paste((0, 0, 0), mask=shadow)

    outline = Image.new("L", canvas.size, 0)
    solid = alpha.point(lambda v: 255 if v > 200 else 0)
    outline.paste(solid.filter(ImageFilter.MaxFilter(9)), (x, y))
    canvas.paste((255, 255, 255), mask=outline.filter(ImageFilter.GaussianBlur(1)))

    canvas.paste(rgb, (x, y), alpha)

    face = sub.get("face")
    if face:
        keep.append(((x + face[0] * scale, y + face[1] * scale,
                      x + face[2] * scale, y + face[3] * scale), 6.0))
    else:
        keep.append(((x, y, x + w, y + 0.4 * h), 4.0))


def _draw_card(canvas, image, cx, cy, target_h, angle, keep):
    W, H = canvas.size
    img = image.convert("RGB")
    s = min(target_h / img.height, 0.46 * W / img.width)
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    shot = img.resize((w, h), Image.Resampling.LANCZOS)

    border = 10
    card = Image.new("RGBA", (w + 2 * border, h + 2 * border), (0, 0, 0, 0))
    ImageDraw.Draw(card).rounded_rectangle(
        (0, 0, card.width - 1, card.height - 1), radius=20,
        fill=(255, 255, 255, 255))
    inner = Image.new("L", (w, h), 0)
    ImageDraw.Draw(inner).rounded_rectangle((0, 0, w - 1, h - 1), radius=12, fill=255)
    card.paste(shot, (border, border), inner)

    card = card.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    x, y = int(cx - card.width / 2), int(cy - card.height / 2)

    shadow = Image.new("L", canvas.size, 0)
    shadow.paste(card.getchannel("A").point(lambda v: int(v * 0.7)),
                 (x + 12, y + 16))
    canvas.paste((0, 0, 0), mask=shadow.filter(ImageFilter.GaussianBlur(18)))
    canvas.paste(card.convert("RGB"), (x, y), card.getchannel("A"))
    keep.append(((x, y, x + card.width, y + card.height), 3.0))


def compose(background: Image.Image, people, card, accent):
    """
    يركّب العناصر على الخلفية. يعيد (الصورة، صناديق لا يجوز تغطيتها بالنص).
    - شخصان: واحد يسارًا وواحد يمينًا.
    - شخص + لقطة: الشخص يسارًا والبطاقة يمينًا.
    - شخص وحده: يمينًا. - لقطة وحدها: في اليمين مائلة قليلًا.
    """
    canvas = background.convert("RGB").copy()
    W, H = canvas.size
    keep: list = []

    if len(people) >= 2:
        _draw_person(canvas, people[0], 0.27 * W, 1.00 * H, accent, keep)
        _draw_person(canvas, people[1], 0.75 * W, 0.92 * H, accent, keep)
    elif len(people) == 1 and card:
        _draw_person(canvas, people[0], 0.28 * W, 1.00 * H, accent, keep)
        _draw_card(canvas, card["image"], 0.72 * W, 0.46 * H, 0.66 * H, -4, keep)
    elif len(people) == 1:
        _draw_person(canvas, people[0], 0.68 * W, 1.02 * H, accent, keep)
    elif card:
        _draw_card(canvas, card["image"], 0.60 * W, 0.46 * H, 0.70 * H, -4, keep)

    canvas = _vignette(canvas)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.05)
    return canvas, keep
