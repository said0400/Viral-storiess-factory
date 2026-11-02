"""
صورة فيسبوك: صورة واحدة بمقاس 4:5 (1080×1350) أو 3:4 (1080×1440).

- أفضل صورة تملأ الإطار كله بقص ذكي حول العنصر المهم (بلا دائرة ولا كولاج).
- أسفل الصورة تدرّج أسود كثيف عند الحافة السفلية يخفّ تدريجيًا نحو الأعلى
  حتى يصير شفافًا، والعنوان العربي مكتوب فوقه.
- إطار أبيض رقيق يحيط بالصورة كلها.

وحدة مستقلة: لا تستورد من السكربت الرئيسي. تُستدعى بالتوقيع نفسه:
    create_design(images, plan, destination, headline=None, highlight=None)
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_TEXT_OK = True
except ImportError:
    ARABIC_TEXT_OK = False

IMAGE_SIZE = 1080
FONT_FILENAME = "IBMPlexSansArabic-Bold.ttf"
FONT_FALLBACKS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",)
TEXT_ACCENT = "#FFD400"

FRAME_INSET = 24        # بعد الإطار عن حافة الصورة (بكسل)
FRAME_WIDTH = 6         # سماكة الإطار
TEXT_ZONE = 0.30        # أقصى ارتفاع يشغله العنوان من الصورة
VERTICAL_FOCUS = 0.42   # موضع مركز العنصر المهم من ارتفاع النافذة (يترك الأسفل للنص)
GRADIENT_MAX_ALPHA = 0.95
GRADIENT_CURVE = 1.6    # أكبر = يخفّ التدرج أسرع نحو الأعلى


class CoverError(ValueError):
    """خطأ في تصميم الصورة (يلتقطه السكربت الرئيسي ضمن ValueError)."""


# ---------------------------------------------------------------------------
# أدوات عامة
# ---------------------------------------------------------------------------

def facebook_height() -> int:
    """4:5 افتراضيًا (1080×1350)، أو 3:4 (1080×1440) عبر FB_IMAGE_RATIO."""
    ratio = (os.getenv("FB_IMAGE_RATIO", "").strip()
             .replace(".", ":").replace("/", ":"))
    if ratio == "3:4":
        return round(IMAGE_SIZE * 4 / 3)
    return round(IMAGE_SIZE * 5 / 4)


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_headline(text: str | None) -> str:
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]", "",
                  text or "")
    text = text.replace("#", " ").replace("«", "").replace("»", "")
    text = text.replace('"', "").replace("“", "").replace("”", "")
    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("C") or ch == " "
    )
    return _clean_text(text)


def find_font_path() -> str | None:
    candidates = [os.getenv("FONT_PATH", "").strip()]
    here = Path(__file__).resolve().parent
    candidates += [
        here / "fonts" / FONT_FILENAME,
        Path("fonts") / FONT_FILENAME,
        *FONT_FALLBACKS,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def shape_word(word: str) -> str:
    return get_display(arabic_reshaper.reshape(word), base_dir="R")


# ---------------------------------------------------------------------------
# القص الذكي
# ---------------------------------------------------------------------------

def smart_window(image, box, aspect, padding=0.12):
    """أكبر نافذة بنسبة الخانة متمركزة على العنصر، ويدخل العنصر فيها إن اتسع."""
    W, H = image.size
    l, t, r, b = box
    bl, br, bt, bb = l * W, r * W, t * H, b * H
    cx, cy = (bl + br) / 2, (bt + bb) / 2

    if W / H > aspect:
        win_w, win_h = H * aspect, H
    else:
        win_w, win_h = W, W / aspect

    x0 = min(max(cx - win_w / 2, 0), W - win_w)
    y0 = min(max(cy - win_h / 2, 0), H - win_h)

    if br - bl <= win_w:
        x0 = min(max(x0, br - win_w), bl)
    if bb - bt <= win_h:
        y0 = min(max(y0, bb - win_h), bt)
    x0 = min(max(x0, 0), W - win_w)
    y0 = min(max(y0, 0), H - win_h)
    return (x0, y0, x0 + win_w, y0 + win_h)


def cover_window(image, anchor, aspect):
    """
    نافذة القص: كما smart_window، لكن إن كان في الصورة حرية رأسية يُرفع العنصر
    المهم نحو الأعلى (VERTICAL_FOCUS) ليبقى أسفل الصورة للعنوان، دون قطع أعلاه.
    """
    W, H = image.size
    x0, y0, x1, y1 = smart_window(image, anchor, aspect)
    wh = y1 - y0
    if wh < H - 1:
        cy = (anchor[1] + anchor[3]) / 2 * H
        top = anchor[1] * H
        ny = cy - VERTICAL_FOCUS * wh
        ny = min(ny, top - 0.04 * wh)          # لا نقطع أعلى العنصر
        ny = min(max(ny, 0), H - wh)
        y0 = ny
    return (x0, y0, x1, y0 + wh)


def crop_window(image, win, size):
    W, H = image.size
    x0 = max(0, min(W - 1, round(win[0])))
    y0 = max(0, min(H - 1, round(win[1])))
    x1 = max(x0 + 1, min(W, round(win[2])))
    y1 = max(y0 + 1, min(H, round(win[3])))
    return ImageOps.fit(image.crop((x0, y0, x1, y1)), size,
                        method=Image.Resampling.LANCZOS)


def polish_soft(tile):
    """تحسين خفيف: لا نبالغ في الحدة فتُظهر عيوب الصور الضعيفة."""
    tile = ImageEnhance.Contrast(tile).enhance(1.04)
    tile = ImageEnhance.Color(tile).enhance(1.07)
    return ImageEnhance.Sharpness(tile).enhance(1.08)


def choose_anchor(image, item, aspect):
    """إن اتسع العنصر الرئيسي في النافذة فهو المرجع، وإلا فالتفصيل المهم."""
    W, H = image.size
    if W / H > aspect:
        max_w, max_h = H * aspect, H
    else:
        max_w, max_h = W, W / aspect
    sb = item["subject_box"]
    fits = ((sb[2] - sb[0]) * W <= max_w * 1.02
            and (sb[3] - sb[1]) * H <= max_h * 1.02)
    if fits:
        return sb
    return item.get("detail_box") or sb


# ---------------------------------------------------------------------------
# اختيار أفضل صورة
# ---------------------------------------------------------------------------

def _coverage(image, item, win):
    """نسبة ما تحفظه النافذة من العنصر الرئيسي (1.0 = كامل)."""
    W, H = image.size
    sb = item["subject_box"]
    sl, st, sr, sbm = sb[0] * W, sb[1] * H, sb[2] * W, sb[3] * H
    area = (sr - sl) * (sbm - st)
    if area <= 0:
        return 1.0
    iw = max(0.0, min(sr, win[2]) - max(sl, win[0]))
    ih = max(0.0, min(sbm, win[3]) - max(st, win[1]))
    return iw * ih / area


def _text_overlap(image, item, win):
    """نسبة تغطية صناديق الوجوه (avoid_boxes) بمنطقة النص السفلية."""
    W, H = image.size
    zone = (win[0], win[1] + (1 - TEXT_ZONE) * (win[3] - win[1]), win[2], win[3])
    total = 0.0
    for b in item.get("avoid_boxes") or []:
        bx0, by0, bx1, by1 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        area = max(1.0, (bx1 - bx0) * (by1 - by0))
        ow = max(0.0, min(zone[2], bx1) - max(zone[0], bx0))
        oh = max(0.0, min(zone[3], by1) - max(zone[1], by0))
        total += ow * oh / area
    return min(total, 1.0)


def _candidates(images, plan):
    """صور الخطة أولًا (الأهم أولًا) ثم صور المعرض غير المكررة."""
    found, seen = [], set()
    for pos, item in enumerate(plan.get("images") or []):
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(images) and idx not in seen:
            seen.add(idx)
            found.append((0.35 if pos == 0 else 0.15, item))
    for item in plan.get("gallery") or []:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(images) and idx not in seen:
            seen.add(idx)
            found.append((0.0, item))
    return found


def pick_cover(images, plan, size):
    """يعيد (العنصر، النافذة) لأفضل صورة فوتوغرافية: تغطية + دقة + ابتعاد الوجوه عن النص."""
    W, H = size
    aspect = W / H
    cands = _candidates(images, plan)
    if not cands:
        raise CoverError("خطة التصميم لا تشير إلى صور متاحة.")

    photos = [(b, it) for b, it in cands if it.get("kind", "photo") == "photo"]
    if not photos:
        return cands[0][1], None        # لا صورة فوتوغرافية: مسار الصورة المركّبة

    best, best_score = None, -1e9
    for bonus, item in photos:
        img = images[item["index"]]
        win = cover_window(img, choose_anchor(img, item, aspect), aspect)
        score = (
            2.0 * _coverage(img, item, win)
            + 1.5 * min(1.0, (win[2] - win[0]) / IMAGE_SIZE)   # الدقة الفعلية
            - 1.0 * _text_overlap(img, item, win)
            + bonus
        )
        if score > best_score:
            best, best_score = (item, win), score
    return best


def build_framed_asis(img, size):
    """صورة مركّبة أو لقطة شاشة: تُعرض كاملة دون قص فوق خلفية ضبابية."""
    W, H = size
    small = (W // 4, H // 4)
    bg = ImageOps.fit(img, small, Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(8))
    bg = bg.resize(size, Image.Resampling.BICUBIC)
    bg = ImageEnhance.Brightness(bg).enhance(0.5)

    avail_w, avail_h = int(W * 0.88), int(H * (1 - TEXT_ZONE) - H * 0.06)
    scale = min(avail_w / img.width, avail_h / img.height)
    fg = img.resize((max(1, round(img.width * scale)),
                     max(1, round(img.height * scale))),
                    Image.Resampling.LANCZOS)
    bg.paste(fg, ((W - fg.width) // 2, int(H * 0.05)))
    return bg


# ---------------------------------------------------------------------------
# العنوان
# ---------------------------------------------------------------------------

def layout_for_band(words, font_path, width, band_h):
    """أكبر خط يتسع فيه العنوان داخل ارتفاع band_h (حتى 4 أسطر)."""
    max_w = width * 0.88
    shaped = [shape_word(w) for w in words]
    result = None

    for size in range(int(width * 0.080), int(width * 0.036) - 1, -2):
        font = ImageFont.truetype(font_path, size,
                                  layout_engine=ImageFont.Layout.BASIC)
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

        line_h = int(size * 1.28)
        bar_h = max(6, int(size * 0.10))
        gap = int(size * 0.35)
        content_h = bar_h + gap + line_h * len(lines)

        result = {
            "font": font, "size": size, "space": space, "shaped": shaped,
            "widths": widths, "lines": lines, "line_widths": line_widths,
            "line_h": line_h, "bar_h": bar_h, "gap": gap,
            "content_h": content_h,
        }
        if (len(lines) <= 4 and max(line_widths) <= max_w
                and content_h <= band_h * 0.92):
            return result
    return result


def add_overlay(picture, headline, highlight):
    """تدرّج أسود سفلي + العنوان + إطار أبيض رقيق. يعيد (الصورة، معلومات النص)."""
    W, H = picture.size
    canvas = picture.copy()
    lay, words, font_path = None, [], None

    if headline:
        if not ARABIC_TEXT_OK:
            print("::warning title=النص العربي::مكتبتا arabic-reshaper و "
                  "python-bidi غير مثبتتين؛ تُخطى الكتابة على الصورة.")
        else:
            font_path = find_font_path()
            if not font_path:
                print("::warning title=النص العربي::لم يُعثر على خط عربي.")
            else:
                words = clean_headline(headline).split()[:24]
                if words:
                    lay = layout_for_band(words, font_path, W,
                                          int(H * TEXT_ZONE))

    bottom_pad = FRAME_INSET + FRAME_WIDTH + 34
    info = None

    if lay:
        grad_h = min(H, lay["content_h"] + bottom_pad + int(H * 0.16))
        alpha = Image.linear_gradient("L").resize(
            (W, grad_h), Image.Resampling.BILINEAR)
        alpha = alpha.point(
            lambda v: int(255 * GRADIENT_MAX_ALPHA
                          * (v / 255) ** GRADIENT_CURVE))
        canvas.paste((0, 0, 0), (0, H - grad_h, W, H), alpha)

        draw = ImageDraw.Draw(canvas)
        top = H - bottom_pad - lay["content_h"]

        bar_w = int(W * 0.12)
        draw.rounded_rectangle(
            ((W - bar_w) // 2, top, (W + bar_w) // 2, top + lay["bar_h"]),
            radius=lay["bar_h"] // 2, fill=TEXT_ACCENT)

        def norm(word):
            return re.sub(r"[^\w]", "", word, flags=re.UNICODE)

        marked = {norm(w) for w in clean_headline(highlight).split()} - {""}
        stroke = max(2, lay["size"] // 28)
        y = top + lay["bar_h"] + lay["gap"]
        for line, line_w in zip(lay["lines"], lay["line_widths"]):
            x = (W + line_w) / 2            # الكتابة من اليمين إلى اليسار
            for idx in line:
                x -= lay["widths"][idx]
                color = TEXT_ACCENT if norm(words[idx]) in marked else "white"
                draw.text((x, y), lay["shaped"][idx], font=lay["font"],
                          fill=color, stroke_width=stroke, stroke_fill="black")
                x -= lay["space"]
            y += lay["line_h"]

        info = {
            "text": " ".join(words), "zone": "bottom_gradient",
            "font": Path(font_path).name, "font_size": lay["size"],
            "lines": len(lay["lines"]), "size": [W, H],
        }

    # الإطار الأبيض الرقيق يحيط بالصورة كلها (يمرّ فوق التدرج أيضًا)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (FRAME_INSET, FRAME_INSET, W - FRAME_INSET - 1, H - FRAME_INSET - 1),
        outline="white", width=FRAME_WIDTH)
    return canvas, info


# ---------------------------------------------------------------------------
# الواجهة العامة
# ---------------------------------------------------------------------------

def create_design(images, plan, destination: Path, headline=None,
                  highlight=None):
    if not images:
        raise CoverError("لا توجد صور صالحة للتصميم.")

    size = (IMAGE_SIZE, facebook_height())
    item, win = pick_cover(images, plan, size)
    img = images[item["index"]]

    if win is None:
        picture = build_framed_asis(img, size)
        mode = "asis_framed"
    else:
        if win[2] - win[0] < IMAGE_SIZE * 0.5:
            print(f"تحذير: الصورة المختارة صغيرة ({img.width}×{img.height}) "
                  "وستُكبَّر أكثر من ضعفين؛ قد تظهر أقل حدة.")
        picture = polish_soft(crop_window(img, win, size))
        mode = "cover"

    text_enabled = os.getenv("ADD_IMAGE_TEXT", "").strip().lower() not in {
        "0", "false", "no", "off"}
    canvas, info = add_overlay(
        picture, headline if (headline and text_enabled) else None,
        highlight or "")

    # تسجيل ما حدث في خطة التحليل المحفوظة داخل result.json
    plan["layout_applied"] = f"single_{mode}"
    plan["cover_image_index"] = item["index"]
    if info:
        plan["headline"] = info
        print(f"النص على الصورة: {info['text']} (تدرّج سفلي)")
    print(f"صورة فيسبوك: الصورة رقم {item['index']} ({mode}) "
          f"{size[0]}×{size[1]}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, "JPEG", quality=93, optimize=True)
    return destination
