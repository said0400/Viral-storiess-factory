"""
لوحة مشاهد القصة: صور مرتبطة بأحداث القصة نفسها لا صور عامة.

1) make_storyboard: بعد كتابة القصة يقرأ Gemini فقراتها المرقّمة ويصف لحظات بصرية محددة
   منها (الشخصيات والمكان والأشياء من نص القصة)، مع وصف آمن بلا عنف لكل مشهد يُستعمل
   إن رفض مرشّح Cloudflare الوصف الأصلي، ووصف لصورة الغلاف.
2) generate_scene_images: يولّد صورة 16:9 لكل مشهد (سلّم محاولات)، وأي مشهد يفشل يرث
   أقرب صورة ناجحة؛ وإن انتهت حصة Cloudflare يكتفي بما ولّده.
3) scene_segments: يحوّل رقم فقرة كل مشهد إلى لحظة في الصوت المروي فتظهر صورة كل مشهد
   حين تُروى أحداثه.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import cloudflare_ai
import grok
import story_categories
import story_image
import story_text
import story_writer
from cloudflare_ai import CloudflareError
from grok import GrokError
from story_writer import StoryError

SCENE_SIZE = (1280, 720)
FRAME_SIZE = (1920, 1080)
DEFAULT_SCENES = 6
MIN_SCENE_SECONDS = 6.0

STORYBOARD_SYSTEM = """
You are the art director of a dark-fiction storytelling channel. You receive an Arabic short
story split into numbered paragraphs. Create a storyboard: {N} key visual moments of THIS exact
story, to be rendered by an image model (FLUX) and shown in order while the story is narrated.

Return valid JSON only, no Markdown:
{
  "characters": "visual description of the recurring characters (age, build, hair, clothing) and the main location",
  "cover_prompt": "...",
  "cover_safe_prompt": "...",
  "scenes": [
    {"paragraph": 0, "prompt": "...", "safe_prompt": "..."}
  ]
}

Rules:
- Everything must come from the story itself: its real setting, objects, clues, weather and characters.
  Never generic stock imagery (no "dark room", "spooky forest" unless the story really has it).
- Every "prompt" is English, 45 to 80 words, ONE clear subject and action, with concrete visual
  details: who (repeat the characters' visual description in every prompt so they look the same in
  all scenes), where, what is happening, the key object, camera angle, light and mood.
  Cinematic, photorealistic, dark moody lighting. No text, letters, captions, logos or watermarks.
  No real people, celebrities or brands.
- "safe_prompt": the SAME moment rephrased with no violence, blood, weapons, corpses, nudity or
  gore; express the tension through atmosphere, shadows, silhouettes, objects and aftermath.
- "paragraph": the integer index of the paragraph where that moment happens. Scenes must be in story
  order, spread across the whole story (first scene = paragraph 0, last scene near the ending), with
  at least 3 paragraphs between scenes when the story allows. Do not reveal the final twist in a scene
  that comes before the twist is told.
- "cover_prompt": ONE striking vertical (portrait 4:5) poster composition of the story's central
  subject, intriguing but without spoiling the ending. "cover_safe_prompt" is its non-violent version.
- Exactly {N} scenes.
""".strip()


def _clean(text: Any, limit: int = 900) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def _as_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def normalize_storyboard(raw: dict[str, Any], n_paragraphs: int) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in raw.get("scenes") or []:
        if not isinstance(item, dict):
            continue
        index = _as_index(item.get("paragraph"))
        prompt = _clean(item.get("prompt"))
        if index is None or len(prompt) < 30:
            continue
        index = max(0, min(n_paragraphs - 1, index))
        if index in seen:
            continue
        seen.add(index)
        scenes.append({"paragraph": index, "prompt": prompt,
                       "safe_prompt": _clean(item.get("safe_prompt"))})
    scenes.sort(key=lambda s: s["paragraph"])
    if len(scenes) < 2:
        raise StoryError("لوحة المشاهد ناقصة (أقل من مشهدين صالحين).")
    scenes[0]["paragraph"] = 0

    cover = _clean(raw.get("cover_prompt"))
    if len(cover) < 30:
        cover = scenes[0]["prompt"]
    return {
        "characters": _clean(raw.get("characters"), 600),
        "cover_prompt": cover,
        "cover_safe_prompt": _clean(raw.get("cover_safe_prompt")),
        "scenes": scenes[:10],
    }


def make_storyboard(title: str, article_markdown: str, category: str,
                    want: int | None = None) -> dict[str, Any]:
    """لوحة مشاهد من نص القصة بـ Gemini (مفتاح الكتابة). يرفع StoryError عند الفشل."""
    paragraphs = story_text.story_paragraphs(article_markdown)
    if len(paragraphs) < 4:
        raise StoryError("القصة أقصر من أن تُقسَّم إلى مشاهد.")
    want = want or story_writer._env_int("STORY_SCENES", DEFAULT_SCENES, 3, 10)
    system = STORYBOARD_SYSTEM.replace("{N}", str(want))
    numbered = "\n\n".join(f"[{i}] {p[:420]}" for i, p in enumerate(paragraphs))
    user = f"Story title: {title}\nCategory: {category}\n\nParagraphs:\n{numbered}"

    last: Exception | None = None
    for attempt in (1, 2):
        text, _ = story_writer._call_gemini(system, user, temperature=0.6)
        try:
            board = normalize_storyboard(grok._extract_json(text), len(paragraphs))
            board["paragraphs"] = len(paragraphs)
            return board
        except (GrokError, StoryError, ValueError, KeyError, TypeError) as exc:
            last = exc
            print(f"تحذير: لوحة المشاهد غير صالحة (المحاولة {attempt}): {exc}")
    raise StoryError(f"تعذر إنشاء لوحة المشاهد: {last}")


# ---------------------------------------------------------------------------
# صور المشاهد
# ---------------------------------------------------------------------------

def _render_scene(scene: dict[str, Any], info: dict[str, Any]) -> Any:
    """سلّم محاولات لمشهد واحد. يعيد صورة PIL أو None، ويرفع CloudflareQuotaError."""
    fallback = cloudflare_ai.fallback_model_name()
    safe = scene.get("safe_prompt") or info["safe_image"]
    steps = [
        (None, scene["prompt"]),
        (None, scene["prompt"]),
        (fallback, scene["prompt"]),
        (fallback, safe),
    ]
    for model, prompt in steps:
        try:
            image = cloudflare_ai.generate_image(
                prompt + story_image.STYLE_SUFFIX, [], SCENE_SIZE[0], SCENE_SIZE[1],
                random.randint(1, 2**31 - 1), model=model)
            if story_image._is_blank(image):
                raise CloudflareError("الصورة المولّدة فارغة تقريبًا.")
            return image
        except cloudflare_ai.CloudflareQuotaError:
            raise
        except CloudflareError as exc:
            print(f"::warning title=تعذر توليد صورة مشهد::{str(exc)[:160]}")
    return None


def generate_scene_images(storyboard: dict[str, Any], category: str,
                          work_dir: Path) -> list[dict[str, Any]]:
    """
    يولّد صورة لكل مشهد ويحفظها في work_dir/scenes. يعيد قائمة بنفس ترتيب المشاهد:
    {"paragraph", "path", "generated"}. المشهد الفاشل يرث أقرب صورة ناجحة؛ وإن فشلت
    كلها يعيد قائمة فارغة (يرجع الفيديو إلى إطار واحد).
    """
    if not story_image._flag("CF_IMAGE_ENABLED", True) or not cloudflare_ai.is_configured():
        print("تنبيه: Cloudflare غير مفعّل؛ لا صور مشاهد.")
        return []

    info = story_categories.category_info(category)
    rows: list[dict[str, Any]] = []
    quota = False
    for number, scene in enumerate(storyboard["scenes"], start=1):
        image = None
        if not quota:
            print(f"  مشهد {number}/{len(storyboard['scenes'])} (فقرة {scene['paragraph']})...")
            try:
                image = _render_scene(scene, info)
            except cloudflare_ai.CloudflareQuotaError:
                quota = True
                print("::warning title=حصة Cloudflare::انتهت الحصة؛ تُكرَّر الصور المتاحة.")
        rows.append({"paragraph": scene["paragraph"], "image": image,
                     "generated": image is not None})

    if not any(r["generated"] for r in rows):
        return []
    previous = None
    for row in rows:                                   # الفاشل يرث السابق
        if row["image"] is not None:
            previous = row["image"]
        else:
            row["image"] = previous
    following = None
    for row in reversed(rows):                         # الفاشل في البداية يرث اللاحق
        if row["image"] is not None:
            following = row["image"]
        else:
            row["image"] = following

    folder = work_dir / "scenes"
    folder.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        path = folder / f"scene_{index:02d}.jpg"
        row["image"].convert("RGB").resize(FRAME_SIZE).save(path, "JPEG", quality=93)
        row["path"] = path
        del row["image"]
    return rows


# ---------------------------------------------------------------------------
# التوقيت: من رقم الفقرة إلى لحظة في الصوت
# ---------------------------------------------------------------------------

def paragraph_start_times(paragraphs: list[str], title: str, chunks: list[str],
                          chunk_seconds: list[float], pause: float) -> list[float]:
    """لحظة بدء قراءة كل فقرة (بالثواني) بالتناسب مع عدد الكلمات داخل مقطعها الصوتي."""
    offsets, acc = [], len(title.split()) if title.strip() else 0
    for paragraph in paragraphs:
        offsets.append(acc)
        acc += len(paragraph.split())

    starts, clock, counts = [], 0.0, []
    for chunk, seconds in zip(chunks, chunk_seconds):
        starts.append(clock)
        counts.append(len(chunk.split()))
        clock += seconds + pause
    cumulative = [0]
    for count in counts:
        cumulative.append(cumulative[-1] + count)

    def at(word: int) -> float:
        for j in range(len(chunks)):
            if word < cumulative[j + 1] or j == len(chunks) - 1:
                fraction = min(max((word - cumulative[j]) / max(1, counts[j]), 0.0), 1.0)
                return starts[j] + fraction * chunk_seconds[j]
        return 0.0

    return [at(offset) for offset in offsets]


def scene_segments(scenes: list[dict[str, Any]], starts: list[float],
                   total_seconds: float,
                   min_seconds: float = MIN_SCENE_SECONDS) -> list[dict[str, Any]]:
    """
    مقاطع الفيديو: لكل مشهد {"scene", "start", "end", "seconds"}. المشاهد المتقاربة جدًا
    (أقل من min_seconds) تُدمج في سابقها، فتمتد الصورة السابقة.
    """
    times = [starts[min(s["paragraph"], len(starts) - 1)] for s in scenes]
    times[0] = 0.0
    keep = [0]
    for i in range(1, len(times)):
        if (times[i] - times[keep[-1]] >= min_seconds
                and total_seconds - times[i] >= min_seconds):
            keep.append(i)
    segments = []
    for position, scene_index in enumerate(keep):
        start = times[scene_index]
        end = times[keep[position + 1]] if position + 1 < len(keep) else total_seconds
        segments.append({"scene": scene_index, "start": start, "end": end,
                         "seconds": end - start})
    return segments


def scenes_enabled() -> bool:
    return os.getenv("STORY_SCENES", "").strip().lower() not in {"0", "off", "no", "false"}
