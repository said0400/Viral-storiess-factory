"""
الصورة الأولى: دمج صور المقال في صورة عريضة 16:9 بلا أي نص.
وحدة مستقلة عن الصورة المربعة، ولا تعدّل شيئًا فيها.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
    ImageStat,
)


WIDE_SIZE = (1280, 720)
SS = 2                       # دقة مضاعفة لنعومة الحواف
GUTTER = 8
GUTTER_COLOR = "#FFFFFF"


class CollageError(Exception):
    """خطأ متوقع أثناء إنشاء الصورة العريضة."""


# ---------------------------------------------------------------------------
# أدوات القص
# ---------------------------------------------------------------------------

def smart_window(image, box, aspect, mode="cover",
                 padding=0.12, min_frac=0.2):
    """نافذة قص بنسبة الخانة تحتوي العنصر المهم."""
    W, H = image.size
    l, t, r, b = box
    bl, br, bt, bb = l * W, r * W, t * H, b * H
    cx, cy = (bl + br) / 2, (bt + bb) / 2

    if W / H > aspect:
        max_w, max_h = H * aspect, H
    else:
        max_w, max_h = W, W / aspect

    if mode == "cover":
        win_w, win_h = max_w, max_h
    else:
        bw = max((br - bl) * (1 + 2 * padding), 1.0)
        bh = max((bb - bt) * (1 + 2 * padding), 1.0)
        win_w = bw if bw / bh > aspect else bh * aspect
        win_w = min(max(win_w, max_w * min_frac), max_w)
        win_h = win_w / aspect

    x0 = min(max(cx - win_w / 2, 0), W - win_w)
    y0 = min(max(cy - win_h / 2, 0), H - win_h)

    if br - bl <= win_w:
        x0 = min(max(x0, br - win_w), bl)
    if bb - bt <= win_h:
        y0 = min(max(y0, bb - win_h), bt)
    x0 = min(max(x0, 0), W - win_w)
    y0 = min(max(y0, 0), H - win_h)

    return (x0, y0, x0 + win_w, y0 + win_h)


def crop_window(image, win, size):
    W, H = image.size
    x0 = max(0, min(W - 1, round(win[0])))
    y0 = max(0, min(H - 1, round(win[1])))
    x1 = max(x0 + 1, min(W, round(win[2])))
    y1 = max(y0 + 1, min(H, round(win[3])))
    return ImageOps.fit(
        image.crop((x0, y0, x1, y1)), size,
        method=Image.Resampling.LANCZOS,
    )


def polish(tile):
    tile = ImageEnhance.Contrast(tile).enhance(1.06)
    tile = ImageEnhance.Color(tile).enhance(1.10)
    return ImageEnhance.Sharpness(tile).enhance(1.15)


def choose_anchor(image, item, aspect):
    """العنصر الرئيسي إن اتسع، وإلا التفصيل المهم (العينان مثلًا)."""
    W, H = image.size
    if W / H > aspect:
        max_w, max_h = H * aspect, H
    else:
        max_w, max_h = W, W / aspect
    sb = item["subject_box"]
    fits = (
        (sb[2] - sb[0]) * W <= max_w * 1.02
        and (sb[3] - sb[1]) * H <= max_h * 1.02
    )
    return sb if fits else (item.get("detail_box") or sb)


def window_coverage(image, item, w, h):
    """نسبة ما تحفظه الخانة من العنصر الرئيسي (1.0 = كامل)."""
    W, H = image.size
    aspect = w / h
    win = smart_window(image, choose_anchor(image, item, aspect),
                       aspect, "cover")
    sl, st = item["subject_box"][0] * W, item["subject_box"][1] * H
    sr, sb = item["subject_box"][2] * W, item["subject_box"][3] * H
    area = (sr - sl) * (sb - st)
    if area <= 0:
        return 1.0
    iw = max(0.0, min(sr, win[2]) - max(sl, win[0]))
    ih = max(0.0, min(sb, win[3]) - max(st, win[1]))
    return iw * ih / area


def window_is_flat(image, win, threshold=14.0):
    crop = image.crop(tuple(int(v) for v in win)).convert("L").resize((64, 64))
    return ImageStat.Stat(crop).stddev[0] < threshold


# ---------------------------------------------------------------------------
# التفصيل المكبر (الدائرة)
# ---------------------------------------------------------------------------

def make_detail_window(image, box, scale, ch):
    """
    نافذة تفصيل مربعة بتكبير بين 1.5x و3.2x.
    scale: عدد بكسلات الصورة النهائية لكل بكسل من الأصل.
    تعيد (النافذة، قطر الدائرة بالبكسل).
    """
    W, H = image.size
    tight = smart_window(image, box, 1.0, "tight", padding=0.08,
                         min_frac=0.05)
    need = tight[2] - tight[0]

    d_px = min(0.42 * ch, max(0.30 * ch, 1.6 * need * scale))
    dw = min(max(need, d_px / (3.2 * scale)), d_px / (1.5 * scale))
    dw = min(dw, W, H)

    cx = (box[0] + box[2]) / 2 * W
    cy = (box[1] + box[3]) / 2 * H
    x0 = min(max(cx - dw / 2, 0), W - dw)
    y0 = min(max(cy - dw / 2, 0), H - dw)
    return (x0, y0, x0 + dw, y0 + dw), int(d_px)


def pick_detail(image, item, scale, ch):
    sb = item["subject_box"]
    cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
    hw, hh = (sb[2] - sb[0]) * 0.2, (sb[3] - sb[1]) * 0.2
    centered = [max(0.0, cx - hw), max(0.0, cy - hh),
                min(1.0, cx + hw), min(1.0, cy + hh)]
    for box in (item["detail_box"], centered):
        win, d_px = make_detail_window(image, box, scale, ch)
        if not window_is_flat(image, win):
            return win, d_px
    return None, None


def paste_inset(canvas, tile, x, y, shape, border):
    d = tile.width
    mask = Image.new("L", (d, d), 0)
    mdraw = ImageDraw.Draw(mask)
    if shape == "circle":
        mdraw.ellipse((0, 0, d - 1, d - 1), fill=255)
    else:
        mdraw.rounded_rectangle((0, 0, d - 1, d - 1),
                                radius=int(d * 0.07), fill=255)

    shadow = Image.new("L", canvas.size, 0)
    shadow.paste(mask, (x + 6 * SS, y + 8 * SS))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12 * SS))
    shadow = shadow.point(lambda v: int(v * 0.55))
    canvas.paste((0, 0, 0), (0, 0, canvas.width, canvas.height), shadow)

    canvas.paste(tile, (x, y), mask)
    draw = ImageDraw.Draw(canvas)
    box = (x, y, x + d - 1, y + d - 1)
    if shape == "circle":
        draw.ellipse(box, outline="white", width=border)
    else:
        draw.rounded_rectangle(box, radius=int(d * 0.07),
                               outline="white", width=border)


def _draw_ring_and_inset(canvas, detail, d, x, y, ring, plan, keep):
    ring_cx, ring_cy, ring_r = ring
    ring_r = max(ring_r, 34 * SS)
    ins_cx, ins_cy = x + d / 2, y + d / 2
    CW, CH = canvas.size

    if plan["show_ring"] and 0 <= ring_cx <= CW and 0 <= ring_cy <= CH:
        draw = ImageDraw.Draw(canvas)
        vx, vy = ins_cx - ring_cx, ins_cy - ring_cy
        dist = (vx ** 2 + vy ** 2) ** 0.5
        if dist > ring_r + d / 2 + 20 * SS:
            ux, uy = vx / dist, vy / dist
            draw.line(
                (ring_cx + ux * ring_r, ring_cy + uy * ring_r,
                 ins_cx - ux * d / 2, ins_cy - uy * d / 2),
                fill="white", width=4 * SS,
            )
        draw.ellipse(
            (ring_cx - ring_r, ring_cy - ring_r,
             ring_cx + ring_r, ring_cy + ring_r),
            outline="white", width=5 * SS,
        )

    paste_inset(canvas, detail, x, y, plan["inset_shape"], 8 * SS)

    keep.append(((x, y, x + d, y + d), 5.0))
    keep.append(((ring_cx - ring_r, ring_cy - ring_r,
                  ring_cx + ring_r, ring_cy + ring_r), 2.0))


# ---------------------------------------------------------------------------
# التخطيطات
# ---------------------------------------------------------------------------

def build_single_cover(canvas, images, plan, keep):
    """صورة أفقية الطابع: الخلفية قص 16:9 والتفصيل في دائرة بزاوية."""
    CW, CH = canvas.size
    item = plan["images"][0]
    img = images[item["index"]]
    W, H = img.size
    aspect = CW / CH

    main_win = smart_window(img, choose_anchor(img, item, aspect),
                            aspect, "cover")
    mx0, my0, mx1, _ = main_win
    mw = mx1 - mx0
    scale = CW / mw
    canvas.paste(polish(crop_window(img, main_win, (CW, CH))), (0, 0))

    def to_canvas(b):
        return ((b[0] * W - mx0) * scale, (b[1] * H - my0) * scale,
                (b[2] * W - mx0) * scale, (b[3] * H - my0) * scale)

    avoid = [to_canvas(b) for b in item.get("avoid_boxes", [])]
    for box in avoid:
        keep.append((box, 6.0))

    dwin, d = pick_detail(img, item, scale, CH)
    if dwin is None:
        return
    dcx, dcy = (dwin[0] + dwin[2]) / 2, (dwin[1] + dwin[3]) / 2
    dw = dwin[2] - dwin[0]
    detail = polish(crop_window(img, dwin, (d, d)))

    ring_cx, ring_cy = (dcx - mx0) * scale, (dcy - my0) * scale
    ring_r = dw * scale / 2

    sl, st, sr, sb = to_canvas(item["subject_box"])
    scx, scy = (sl + sr) / 2, (st + sb) / 2

    grid_w, grid_h = 256, 144
    edges = canvas.convert("L").resize((grid_w, grid_h)).filter(
        ImageFilter.FIND_EDGES)
    kx, ky = grid_w / CW, grid_h / CH

    def edge_density(px, py):
        box = (max(0, int(px * kx)), max(0, int(py * ky)),
               min(grid_w, int((px + d) * kx) + 1),
               min(grid_h, int((py + d) * ky) + 1))
        if box[2] <= box[0] or box[3] <= box[1]:
            return 1.0
        return min(1.0, ImageStat.Stat(edges.crop(box)).mean[0] / 40.0)

    def overlap_frac(px, py, ax0, ay0, ax1, ay1):
        ow = max(0, min(px + d, ax1) - max(px, ax0))
        oh = max(0, min(py + d, ay1) - max(py, ay0))
        return ow * oh / (d * d)

    m = int(CH * 0.045)
    corners = [(CW - d - m, m), (m, m),
               (CW - d - m, CH - d - m), (m, CH - d - m)]

    def penalty(pos):
        px, py = pos
        dist = ((px + d / 2 - scx) ** 2 + (py + d / 2 - scy) ** 2) ** 0.5
        return (
            3.0 * overlap_frac(px, py, sl, st, sr, sb)
            + 2.5 * sum(overlap_frac(px, py, *b) for b in avoid)
            + 4.0 * overlap_frac(px, py, ring_cx - ring_r, ring_cy - ring_r,
                                 ring_cx + ring_r, ring_cy + ring_r)
            + edge_density(px, py)
            + (0.3 if py > CH / 2 else 0.0)
            - 0.2 * dist / CW
        )

    x, y = min(corners, key=penalty)
    print(f"الصورة العريضة: تكبير {d / (dw * scale):.2f}x")
    _draw_ring_and_inset(canvas, detail, d, x, y,
                         (ring_cx, ring_cy, ring_r), plan, keep)


def build_single_fit(canvas, images, plan, keep):
    """
    صورة طولية: كاملة في الوسط فوق خلفية مموّهة، والتفصيل المكبر في
    الجانب الفارغ. تعيد False إن لم يتسع الجانب.
    """
    CW, CH = canvas.size
    item = plan["images"][0]
    img = images[item["index"]]
    W, H = img.size

    scale = CH / H
    fw = int(W * scale)
    side = (CW - fw) // 2
    if side < 0.26 * CH:
        return False

    small = (CW // 4, CH // 4)
    bg = ImageOps.fit(img, small, Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(8 * SS))
    bg = bg.resize((CW, CH), Image.Resampling.BICUBIC)
    bg = ImageEnhance.Brightness(bg).enhance(0.55)
    canvas.paste(bg, (0, 0))

    fg = polish(img.resize((fw, CH), Image.Resampling.LANCZOS))
    ox = side
    canvas.paste(fg, (ox, 0))

    def to_canvas(b):
        return (ox + b[0] * fw, b[1] * CH, ox + b[2] * fw, b[3] * CH)

    for b in item.get("avoid_boxes", []):
        keep.append((to_canvas(b), 6.0))

    dwin, d = pick_detail(img, item, scale, CH)
    if dwin is None:
        return True

    d = int(min(d, side * 0.9))
    dcx, dcy = (dwin[0] + dwin[2]) / 2, (dwin[1] + dwin[3]) / 2
    dw = dwin[2] - dwin[0]
    detail = polish(crop_window(img, dwin, (d, d)))

    ring_cx, ring_cy = ox + dcx * scale, dcy * scale
    ring_r = dw * scale / 2

    on_right = ring_cx >= CW / 2
    x = (CW - side + (side - d) // 2) if on_right else (side - d) // 2
    y = int(min(max(ring_cy - d / 2, CH * 0.06), CH - d - CH * 0.06))

    print(f"الصورة العريضة (طولية): تكبير {d / (dw * scale):.2f}x")
    _draw_ring_and_inset(canvas, detail, d, x, y,
                         (ring_cx, ring_cy, ring_r), plan, keep)
    return True


def build_asis(canvas, images, plan, keep):
    """صورة مركّبة أو لقطة: تُعرض كاملة فوق خلفية مموّهة."""
    CW, CH = canvas.size
    item = plan["images"][0]
    img = images[item["index"]]
    W, H = img.size

    if 1.6 <= W / H <= 1.95:
        win = smart_window(img, item["subject_box"], CW / CH, "cover")
        canvas.paste(polish(crop_window(img, win, (CW, CH))), (0, 0))
        return

    small = (CW // 4, CH // 4)
    bg = ImageOps.fit(img, small, Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(8 * SS))
    bg = bg.resize((CW, CH), Image.Resampling.BICUBIC)
    bg = ImageEnhance.Brightness(bg).enhance(0.5)
    canvas.paste(bg, (0, 0))

    s = min(CW / W, CH / H)
    fg = img.resize((max(1, round(W * s)), max(1, round(H * s))),
                    Image.Resampling.LANCZOS)
    canvas.paste(fg, ((CW - fg.width) // 2, (CH - fg.height) // 2))


def panel_boxes(layout, CW, CH, main_side="left"):
    g = GUTTER * SS
    if layout == "two_panel":
        half = (CW - g) // 2
        return [(0, 0, half, CH), (half + g, 0, CW, CH)]

    if layout == "three_panel":
        left_w = (CW - g) // 2
        h = (CH - g) // 2
        boxes = [(0, 0, left_w, CH),
                 (left_w + g, 0, CW, h),
                 (left_w + g, h + g, CW, CH)]
        if main_side == "right":
            boxes = [(CW - x1, y0, CW - x0, y1) for x0, y0, x1, y1 in boxes]
        return boxes

    half_w, half_h = (CW - g) // 2, (CH - g) // 2
    return [(0, 0, half_w, half_h), (half_w + g, 0, CW, half_h),
            (0, half_h + g, half_w, CH), (half_w + g, half_h + g, CW, CH)]


def paste_tile(canvas, image, item, box, keep):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    win = smart_window(image, choose_anchor(image, item, w / h),
                       w / h, "cover")
    canvas.paste(polish(crop_window(image, win, (w, h))), (x0, y0))

    W, H = image.size
    ww, wh = win[2] - win[0], win[3] - win[1]
    for b in item.get("avoid_boxes", []):
        keep.append(((x0 + (b[0] * W - win[0]) / ww * w,
                      y0 + (b[1] * H - win[1]) / wh * h,
                      x0 + (b[2] * W - win[0]) / ww * w,
                      y0 + (b[3] * H - win[1]) / wh * h), 6.0))


def build_panels(canvas, images, plan, keep):
    CW, CH = canvas.size
    boxes = panel_boxes(plan["layout"], CW, CH, plan.get("main_side", "left"))
    for item, box in zip(plan["images"], boxes):
        paste_tile(canvas, images[item["index"]], item, box, keep)


def _panels_too_lossy(images, plan, CW, CH):
    """هل تفقد الألواح جزءًا كبيرًا من العناصر المهمة؟"""
    layout, items = plan["layout"], plan["images"]
    if layout not in ("two_panel", "three_panel", "four_grid"):
        return False
    boxes = panel_boxes(layout, CW, CH, plan.get("main_side", "left"))
    score = sum(
        window_coverage(images[i["index"]], i, b[2] - b[0], b[3] - b[1])
        for i, b in zip(items, boxes)
    )
    average = score / max(1, len(items))
    return average < (0.75 if layout == "two_panel" else 0.7)


# ---------------------------------------------------------------------------
# الواجهة العامة
# ---------------------------------------------------------------------------

def create_wide_collage(
    images: list[Image.Image],
    plan: dict[str, Any],
    destination: Path,
    size: tuple[int, int] = WIDE_SIZE,
) -> dict[str, Any]:
    """
    ينشئ الصورة العريضة بلا نص ويحفظها في destination.
    يعيد {"path", "layout", "size", "keepouts"} حيث keepouts صناديق
    (بإحداثيات الصورة النهائية) لا يجوز أن يغطيها نص يُضاف لاحقًا.
    """
    if not images:
        raise CollageError("لا توجد صور صالحة للدمج.")

    plan = dict(plan)
    plan.setdefault("show_ring", True)
    plan.setdefault("inset_shape", "circle")
    plan["images"] = [
        i for i in plan.get("images", [])
        if 0 <= i.get("index", -1) < len(images)
    ]
    if not plan["images"]:
        raise CollageError("خطة الدمج لا تشير إلى صور متاحة.")

    CW, CH = size[0] * SS, size[1] * SS
    layout = plan["layout"]

    if layout in ("two_panel", "three_panel", "four_grid"):
        needed = {"two_panel": 2, "three_panel": 3, "four_grid": 4}[layout]
        if len(plan["images"]) < needed:
            layout = "single_inset"
        elif _panels_too_lossy(images, plan, CW, CH):
            print("تنبيه: الألواح العريضة تفقد العنصر المهم؛ "
                  "التحول إلى صورة واحدة مع تفصيل مكبر.")
            layout = "single_inset"
        plan["layout"] = layout

    if layout != "as_is" and len(plan["images"]) == 1:
        plan["layout"] = layout = "single_inset"

    canvas = Image.new("RGB", (CW, CH), GUTTER_COLOR)
    keep: list = []

    if layout == "as_is":
        build_asis(canvas, images, plan, keep)
    elif layout == "single_inset":
        plan["images"] = plan["images"][:1]
        item = plan["images"][0]
        img = images[item["index"]]
        W, H = img.size
        cover_window = choose_anchor(img, item, CW / CH)
        subject_fits = cover_window is item["subject_box"]
        portrait_like = W / H < 1.25

        done = False
        if portrait_like and not subject_fits:
            done = build_single_fit(canvas, images, plan, keep)
        if not done:
            keep.clear()
            build_single_cover(canvas, images, plan, keep)
    else:
        build_panels(canvas, images, plan, keep)

    final = canvas.resize(size, Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    final.save(destination, "JPEG", quality=92, optimize=True)

    scaled = [
        (tuple(v / SS for v in box), weight) for box, weight in keep
    ]
    return {
        "path": str(destination),
        "layout": layout,
        "size": list(size),
        "keepouts": scaled,
    }


# ---------------------------------------------------------------------------
# صورة المعرض: جمع 1 إلى 4 صور من المقال بلا نص ولا دوائر
# ---------------------------------------------------------------------------

def _fit_blur_tile(canvas, image, box):
    """الصورة كاملة داخل الخانة فوق خلفية مموّهة. تعيد مستطيل الصورة."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    W, H = image.size

    small = (max(8, w // 4), max(8, h // 4))
    bg = ImageOps.fit(image, small, Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(8 * SS))
    bg = bg.resize((w, h), Image.Resampling.BICUBIC)
    bg = ImageEnhance.Brightness(bg).enhance(0.55)
    canvas.paste(bg, (x0, y0))

    s = min(w / W, h / H)
    fw, fh = max(1, round(W * s)), max(1, round(H * s))
    fg = image.resize((fw, fh), Image.Resampling.LANCZOS)
    ox, oy = x0 + (w - fw) // 2, y0 + (h - fh) // 2
    canvas.paste(fg, (ox, oy))
    return ox, oy, fw, fh


def paste_gallery_tile(canvas, image, item, box, keep):
    """قص ذكي للصور الجيدة، وعرض كامل للقطات الشاشة والصور التي ستُقص بشدة."""
    w, h = box[2] - box[0], box[3] - box[1]
    coverage = window_coverage(image, item, w, h)

    if item.get("kind") == "screenshot" or coverage < 0.72:
        ox, oy, fw, fh = _fit_blur_tile(canvas, image, box)
        for b in item.get("avoid_boxes", []):
            keep.append(((ox + b[0] * fw, oy + b[1] * fh,
                          ox + b[2] * fw, oy + b[3] * fh), 6.0))
    else:
        paste_tile(canvas, image, item, box, keep)


def create_gallery_collage(
    images: list[Image.Image],
    gallery: list[dict[str, Any]],
    destination: Path,
    size: tuple[int, int] = WIDE_SIZE,
) -> dict[str, Any]:
    """
    يجمع صور المقال (1 إلى 4) في صورة 16:9 بلا نص: صورة واحدة، أو لوحتان،
    أو ثلاث (كبيرة ولوحتان)، أو أربع في شبكة 2×2.
    """
    items = [
        g for g in (gallery or [])
        if 0 <= g.get("index", -1) < len(images) and g.get("subject_box")
    ][:4]
    if not items:
        raise CollageError("لا توجد صور صالحة لمعرض المقال.")

    CW, CH = size[0] * SS, size[1] * SS
    count = len(items)
    layout = {1: "single", 2: "two_panel", 3: "three_panel",
              4: "four_grid"}[count]

    canvas = Image.new("RGB", (CW, CH), GUTTER_COLOR)
    keep: list = []

    if count == 1:
        paste_gallery_tile(canvas, images[items[0]["index"]], items[0],
                           (0, 0, CW, CH), keep)
    else:
        for item, box in zip(items, panel_boxes(layout, CW, CH)):
            paste_gallery_tile(canvas, images[item["index"]], item, box, keep)

    final = canvas.resize(size, Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    final.save(destination, "JPEG", quality=92, optimize=True)

    return {
        "path": str(destination),
        "layout": layout,
        "count": count,
        "images": [i["index"] for i in items],
        "size": list(size),
        "keepouts": [(tuple(v / SS for v in b), wt) for b, wt in keep],
    }
