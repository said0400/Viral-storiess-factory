"""
فيديو القصة: يحوّل نص القصة إلى صوت بـ Gemini TTS، ويعرض عليه صور مشاهد القصة نفسها
(من لوحة المشاهد) بتحريك بسيط، ثم ينشره على يوتيوب عبر Make ويضع رابطه في القصة.

المراحل:
  1) تنظيف النص وتقسيمه إلى مقاطع (حد المخرج الصوتي في الطلب الواحد) ثم تحويل كل مقطع.
  2) صور المشاهد: لكل مشهد من لوحة المشاهد (result.json -> storyboard) صورة 16:9 تظهر في
     الوقت الذي تُروى فيه أحداثه، بتكبير/انسياب بطيء وانتقال عبر الأسود. وإن لم توجد لوحة
     أو فشل توليد الصور يُستعمل إطار الغلاف وحده بتحريك دوري.
  3) موسيقى خلفية اختيارية من مجلد music/ (تُكرَّر إن كانت أقصر من الفيديو وتُخفَّض إلى
     STORY_MUSIC_VOLUME في المئة، افتراضيًا 12) تُدمج مع الصوت المروي.
     الدمج بـ ffmpeg (مقاطع تُضم بالنسخ + الصوت مع ضبط الشدة)، ثم التحقق من أن الصوت
     موجود ومسموع في الملف النهائي؛ وإن لم يكن يُعاد الدمج بإعادة الترميز، وإن فشل يفشل
     الإنتاج برسالة واضحة بدل نشر فيديو صامت.
  4) رفع story.mp4 إلى GitHub Releases (رابط عام) فيُرسَل الرابط إلى Make لا الملف.
  5) Make يرفع إلى يوتيوب. نقرأ رابط الفيديو من رد Make (youtube_url أو video_id)،
     وإلا نراقب خلاصة RSS لقناتك (YOUTUBE_CHANNEL_ID) حتى يظهر الفيديو بعنوانه.
  6) نضع رابط الفيديو في القصة: بعد مقدمتها ووسطها وآخرها، ونحفظه في result.json.

STORY_TTS_STYLE / STORY_TTS_OUTRO: النص الافتراضي إن تُركا فارغين، و none يعطّلهما.
STORY_VIDEO: off = تعطيل | local = صوت وفيديو للفحص فقط (بلا رفع) | full = كل المراحل.
الافتراضي (فارغ): full إن وُجد السر MAKE_VIDEO_WEBHOOK_URL وإلا local.

الفيديو إضافة اختيارية: أي فشل هنا يُسجَّل تحذيرًا ولا يمنع نشر القصة.

التشغيل: OUT_DIR=output/<hash> WORK_DIR=story_work/<hash> python story_video.py
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import wave
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

import story_categories
import story_scenes
from story_text import (  # clean_story_text و split_chunks متاحتان أيضًا من هذه الوحدة
    RELATED_MARK,
    clean_story_text,
    plain as _plain,
    split_chunks,
    story_paragraphs,
)


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_TTS_MODELS = ("gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts")
DEFAULT_VOICE = "Charon"
DEFAULT_STYLE = "اقرأ القصة التالية كراوٍ محترف بصوت هادئ ومشوق وبفصحى واضحة:"
DEFAULT_OUTRO = "انتهت القصة. شكرًا لاستماعكم."

REQUEST_TIMEOUT = 300
TTS_ATTEMPTS = 5
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
BLOCKED_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "BLOCKLIST",
                   "SPII", "RECITATION", "OTHER"}
SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
)

CHUNK_WORDS = 420            # نحو 3 دقائق صوت: دون حد المخرج الصوتي للطلب الواحد
CHUNK_GAP_SECONDS = 6        # فاصل بين طلبات TTS لتفادي حد الطلبات في الدقيقة
PAUSE_SECONDS = 0.5          # صمت بين المقاطع
CLIP_SECONDS = 60            # طول التحريك الدوري لوضع الإطار الواحد
FPS = 15
VIDEO_SIZE = (1280, 720)
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"   # شدة صوت موحدة مناسبة ليوتيوب
MIN_AUDIBLE_DB = -55.0                         # أقل متوسط شدة نعدّه مسموعًا
MUSIC_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
DEFAULT_MUSIC_PERCENT = 12                     # شدة الموسيقى قياسًا بالصوت المروي (10 إلى 15 مناسبة)
DEFAULT_ZOOM_PERCENT = 10                      # مقدار تكبير الصورة خلال المشهد


class VideoError(Exception):
    """خطأ في إنتاج الفيديو أو نشره."""


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(os.getenv(name, "").strip() or default)))
    except ValueError:
        return default


def _text_setting(name: str, default: str) -> str:
    """متغير نصي: الفارغ أو غير المضبوط = الافتراضي، والقيمة none/off تعطّله."""
    raw = os.getenv(name, "").strip()
    if raw.lower() in {"none", "off", "-"}:
        return ""
    return raw or default


def video_mode() -> str:
    raw = os.getenv("STORY_VIDEO", "").strip().lower()
    if raw in {"off", "0", "false", "no"}:
        return "off"
    if raw in {"local", "full"}:
        return raw
    return "full" if os.getenv("MAKE_VIDEO_WEBHOOK_URL", "").strip() else "local"


# ---------------------------------------------------------------------------
# 1) الصوت: Gemini TTS
# ---------------------------------------------------------------------------

def _tts_keys() -> list[str]:
    """مفاتيح Gemini بالترتيب (لكل مفتاح حصته): مفتاح TTS ثم مفتاح تحليل الصور ثم الكتابة."""
    keys: list[str] = []
    for name in ("GEMINI_TTS_API_KEY", "GEMINI_API_KEY", "GEMINI_WRITER_API_KEY"):
        value = os.getenv(name, "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def _tts_models() -> list[str]:
    custom = os.getenv("GEMINI_TTS_MODEL", "").strip()
    return [custom] if custom else list(DEFAULT_TTS_MODELS)


def _extract_audio(data: dict[str, Any]) -> tuple[bytes, int]:
    """يعيد (PCM 16 بت أحادي، معدل العيّنة)."""
    feedback = data.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        raise VideoError(f"حجب Gemini الطلب: {feedback.get('blockReason')}.")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise VideoError("استجابة Gemini لا تحتوي مرشحات.")
    candidate = candidates[0]
    reason = candidate.get("finishReason")
    if reason in BLOCKED_REASONS:
        raise VideoError(f"أوقف Gemini التحويل الصوتي: {reason}.")
    if reason == "MAX_TOKENS":
        raise VideoError("انتهت رموز Gemini قبل اكتمال الصوت (المقطع طويل).")
    for part in (candidate.get("content") or {}).get("parts") or []:
        inline = part.get("inlineData") or part.get("inline_data") or {}
        if inline.get("data"):
            rate_match = re.search(r"rate=(\d+)", str(inline.get("mimeType") or
                                                      inline.get("mime_type") or ""))
            rate = int(rate_match.group(1)) if rate_match else 24000
            return base64.b64decode(inline["data"]), rate
    raise VideoError("لم يُرجع Gemini صوتًا.")


def tts_chunk(text: str) -> tuple[bytes, int]:
    """يحوّل مقطعًا واحدًا إلى صوت. يجرّب المفاتيح والنماذج ويعيد المحاولة عند الازدحام."""
    keys = _tts_keys()
    if not keys:
        raise VideoError("لا مفتاح Gemini للصوت (GEMINI_TTS_API_KEY أو GEMINI_API_KEY "
                         "أو GEMINI_WRITER_API_KEY).")
    voice = os.getenv("GEMINI_TTS_VOICE", "").strip() or DEFAULT_VOICE
    style = _text_setting("STORY_TTS_STYLE", DEFAULT_STYLE)
    prompt = f"{style}\n\n{text}" if style else text
    threshold = os.getenv("STORY_SAFETY_THRESHOLD", "").strip() or "BLOCK_ONLY_HIGH"
    use_safety = True
    last = "فشل الاتصال بواجهة Gemini."

    for key in keys:
        for model in _tts_models():
            url = f"{GEMINI_BASE_URL}/{model}:generateContent"
            for attempt in range(1, TTS_ATTEMPTS + 1):
                body: dict[str, Any] = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {"voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": voice}}},
                    },
                }
                if use_safety:
                    body["safetySettings"] = [
                        {"category": c, "threshold": threshold}
                        for c in SAFETY_CATEGORIES]
                try:
                    response = requests.post(
                        url, headers={"Content-Type": "application/json",
                                      "x-goog-api-key": key},
                        json=body, timeout=REQUEST_TIMEOUT)
                except requests.RequestException as exc:
                    last = f"تعذر الاتصال بـ Gemini: {exc}"
                else:
                    if response.status_code == 200:
                        try:
                            return _extract_audio(response.json())
                        except (VideoError, ValueError) as exc:
                            last = f"{model}: {exc}"
                            break                         # حجب أو لا صوت: النموذج التالي
                    last = (f"خطأ Gemini HTTP {response.status_code} ({model}): "
                            f"{response.text[:300]}")
                    if response.status_code == 429 and "PerDay" in response.text:
                        break                             # حصة اليوم لهذا النموذج/المفتاح
                    if response.status_code == 400 and use_safety:
                        use_safety = False                # قد لا يقبل النموذج safetySettings
                        continue
                    if response.status_code in (400, 404):
                        break
                    if response.status_code not in RETRY_STATUS:
                        raise VideoError(last)
                if attempt < TTS_ATTEMPTS:
                    wait = min(10 * 2 ** (attempt - 1), 90)
                    hinted = re.search(r'retryDelay"?\s*:\s*"?(\d+(?:\.\d+)?)s', last)
                    if hinted:
                        wait = max(wait, min(int(float(hinted.group(1))) + 2, 120))
                    print(f"تحذير: محاولة الصوت {attempt} فشلت ({last[:120]}) "
                          f"إعادة المحاولة بعد {wait} ثانية.")
                    time.sleep(wait)
    raise VideoError(last)


def synthesize(text: str, wav_path: Path, title: str = "") -> dict[str, Any]:
    """
    يحوّل النص كله إلى WAV واحد (مقاطع مفصولة بصمت قصير). يعيد معلومات تشمل نصوص
    المقاطع ومدة كل منها، لتحويل أرقام فقرات المشاهد إلى توقيت في الصوت.
    """
    outro = _text_setting("STORY_TTS_OUTRO", DEFAULT_OUTRO)
    spoken = (f"{title.strip()}.\n\n" if title.strip() else "") + text
    if outro:
        spoken += "\n\n" + outro
    chunks = split_chunks(spoken, _env_int("STORY_TTS_CHUNK_WORDS", CHUNK_WORDS, 100, 700))
    if not chunks:
        raise VideoError("لا نص للتحويل إلى صوت.")
    print(f"الصوت: {len(chunks)} مقاطع.")

    pcm_parts: list[bytes] = []
    seconds_each: list[float] = []
    rate = 0
    for number, chunk in enumerate(chunks, start=1):
        print(f"  تحويل المقطع {number}/{len(chunks)} ({len(chunk.split())} كلمة)...")
        pcm, chunk_rate = tts_chunk(chunk)
        if not pcm:
            raise VideoError(f"المقطع {number} بلا صوت.")
        if rate and chunk_rate != rate:
            raise VideoError(f"معدلات العيّنة مختلفة بين المقاطع ({rate} و{chunk_rate}).")
        rate = chunk_rate
        pcm_parts.append(pcm)
        seconds_each.append(len(pcm) / 2 / rate)
        if number < len(chunks):
            time.sleep(CHUNK_GAP_SECONDS)

    silence = b"\x00\x00" * int(rate * PAUSE_SECONDS)
    pcm_all = silence.join(pcm_parts)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm_all)
    seconds = len(pcm_all) / 2 / rate
    print(f"مدة الصوت: {seconds / 60:.1f} دقيقة.")
    return {"chunks": len(chunks), "chunk_texts": chunks, "chunk_seconds": seconds_each,
            "seconds": seconds, "rate": rate}


# ---------------------------------------------------------------------------
# 2) الفيديو: ffmpeg
# ---------------------------------------------------------------------------

def ffmpeg_bin() -> str:
    found = os.getenv("FFMPEG_BIN", "").strip() or shutil.which("ffmpeg")
    if not found:
        raise VideoError("ffmpeg غير مثبت (تثبّته خطوة الـ workflow).")
    return found


def _run(cmd: list[str], what: str) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoError(f"فشل ffmpeg ({what}): {result.stderr[-400:]}")
    return result.stderr


def _db(text: str, key: str) -> float | None:
    match = re.search(rf"{key}:\s*(-?[\d.]+|-inf) dB", text)
    if not match:
        return None
    return -120.0 if match.group(1) == "-inf" else float(match.group(1))


def audio_report(path: Path) -> dict[str, Any]:
    """هل في الملف مسار صوت؟ وما شدته (متوسط وأعلى)؟ (ffmpeg volumedetect)"""
    result = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-i", str(path), "-vn", "-af", "volumedetect",
         "-f", "null", "-"], capture_output=True, text=True)
    streams = [line.strip()[:110] for line in result.stderr.splitlines() if "Stream #" in line]
    return {"has_audio": any("Audio:" in s for s in streams),
            "mean": _db(result.stderr, "mean_volume"),
            "max": _db(result.stderr, "max_volume"), "streams": streams}


def audible(report: dict[str, Any]) -> bool:
    return bool(report["has_audio"] and report["mean"] is not None
                and report["mean"] > MIN_AUDIBLE_DB)


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def media_seconds(path: Path) -> float:
    """مدة ملف وسائط بالثواني (من مخرج ffmpeg -i)."""
    result = subprocess.run([ffmpeg_bin(), "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise VideoError(f"تعذر قراءة مدة {path.name}.")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def zoom_amount() -> float:
    """مقدار التكبير في المشهد (كسر): STORY_ZOOM_PERCENT بين 2 و30، الافتراضي 10."""
    return _env_int("STORY_ZOOM_PERCENT", DEFAULT_ZOOM_PERCENT, 2, 30) / 100


def music_gain() -> float:
    """معامل شدة الموسيقى: STORY_MUSIC_VOLUME بالمئة بين 2 و40، الافتراضي 12."""
    return _env_int("STORY_MUSIC_VOLUME", DEFAULT_MUSIC_PERCENT, 2, 40) / 100


def _playable(path: Path) -> bool:
    try:
        return media_seconds(path) > 1.0
    except VideoError:
        return False


def pick_music(output_root: Path) -> Path | None:
    """
    موسيقى القصة من مجلد music/ (STORY_MUSIC_DIR): ملف محدد بـ STORY_MUSIC_FILE، وإلا
    تناوب بين الملفات الصالحة بحسب عدد القصص السابقة فتتنوع الخلفية. STORY_MUSIC=off تعطّلها.
    """
    if os.getenv("STORY_MUSIC", "").strip().lower() in {"off", "0", "false", "no", "none"}:
        return None
    folder = Path(os.getenv("STORY_MUSIC_DIR", "").strip() or "music")
    chosen = os.getenv("STORY_MUSIC_FILE", "").strip()
    if chosen:
        for candidate in (Path(chosen), folder / chosen):
            if candidate.is_file() and _playable(candidate):
                return candidate
        print(f"::warning title=الموسيقى::الملف STORY_MUSIC_FILE={chosen} غير موجود أو غير صالح.")
        return None
    files = []
    if folder.is_dir():
        files = [f for f in sorted(folder.iterdir())
                 if f.suffix.lower() in MUSIC_EXTENSIONS and _playable(f)]
    if not files:
        print(f"لا موسيقى خلفية (ضع ملفات mp3/wav في المجلد {folder}/ لتفعيلها).")
        return None
    done = len(story_categories.load_story_results(output_root))
    return files[done % len(files)]


def write_concat_list(clips: list[Path], list_path: Path) -> Path:
    list_path.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips),
                         encoding="utf-8")
    return list_path


def _audio_graph(seconds: float, music: Path | None, gain: float) -> str:
    """
    مخطط الصوت: الصوت المروي بشدة موحدة (استريو). مع الموسيقى: تُضبط شدتها أيضًا ثم تُضرب
    بـ gain (مثل 0.12) فتبقى في الخلفية، وتظهر تدريجيًا وتختفي تدريجيًا، ثم تُمزج وتُحمى
    من التشبّع. تُكرَّر الموسيقى القصيرة بـ -stream_loop عند الإدخال.
    """
    voice = f"[1:a]{LOUDNORM},aformat=channel_layouts=stereo[v]"
    if music is None:
        return f"{voice};[v]anull[a]"
    fade_out_at = max(0.0, seconds - 4.0)
    return (f"{voice};"
            f"[2:a]{LOUDNORM},aformat=channel_layouts=stereo,volume={gain:.3f},"
            f"afade=t=in:st=0:d=2,afade=t=out:st={fade_out_at:.2f}:d=4[m];"
            "[v][m]amix=inputs=2:duration=first:dropout_transition=0,volume=2,"
            "alimiter=limit=0.95[a]")


def _mux(concat_list: Path, wav: Path, out_mp4: Path, seconds: float,
         music: Path | None, gain: float) -> dict[str, Any]:
    """نسخ الفيديو أولًا ثم إعادة ترميزه إن لم يكن الصوت مسموعًا في الناتج."""
    attempts = [
        ("نسخ الفيديو", ["-c:v", "copy"]),
        ("إعادة ترميز الفيديو", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                                 "-pix_fmt", "yuv420p"]),
    ]
    report: dict[str, Any] = {}
    for label, video_codec in attempts:
        out_mp4.unlink(missing_ok=True)
        cmd = [ffmpeg_bin(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
               "-i", str(wav)]
        if music is not None:
            cmd += ["-stream_loop", "-1", "-i", str(music)]
        cmd += ["-filter_complex", _audio_graph(seconds, music, gain),
                "-map", "0:v:0", "-map", "[a]", *video_codec,
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                "-t", f"{seconds:.3f}", "-movflags", "+faststart", str(out_mp4)]
        _run(cmd, label)
        report = audio_report(out_mp4)
        print(f"الصوت في الفيديو ({label}): متوسط {report['mean']} dB، أعلى {report['max']} dB")
        if audible(report):
            return report
        print(f"::warning title=صوت الفيديو::لا صوت مسموع بعد «{label}»؛ نجرّب طريقة أخرى.")
    raise VideoError("الفيديو بلا صوت مسموع رغم سلامة الصوت المولّد. المسارات: "
                     + " | ".join(report.get("streams", [])) + f" | متوسط الشدة {report.get('mean')}")


def finalize_video(concat_list: Path, wav: Path, out_mp4: Path, seconds: float,
                   music: Path | None = None, gain: float | None = None) -> dict[str, Any]:
    """
    يضم مقاطع الفيديو (concat) مع الصوت المروي وموسيقى الخلفية الاختيارية. الصوت: AAC
    استريو 48 كيلوهرتز بشدة موحدة. يتحقق أن الصوت مسموع، ويفشل بوضوح بدل أن يعيد فيديو
    صامتًا. وإن تعذر دمج الموسيقى (ملف تالف مثلًا) يُدمج الصوت المروي وحده.
    """
    source = audio_report(wav)
    if not audible(source):
        raise VideoError(f"الصوت المولّد صامت تقريبًا (متوسط الشدة {source['mean']} dB).")
    gain = music_gain() if gain is None else gain

    for use_music in ([music, None] if music is not None else [None]):
        try:
            report = _mux(concat_list, wav, out_mp4, seconds, use_music, gain)
        except VideoError as exc:
            if use_music is None:
                raise
            print(f"::warning title=الموسيقى::تعذر دمج الموسيقى ({str(exc)[:160]})؛ "
                  "يُدمج الصوت المروي وحده.")
            continue
        report["music"] = use_music.name if use_music is not None else None
        if use_music is not None:
            print(f"الموسيقى: {use_music.name} بشدة {gain:.0%} من الصوت المروي (تُكرَّر تلقائيًا).")
        return report
    raise VideoError("تعذر دمج الصوت.")


def make_clip(frame: Path, clip_path: Path, seconds: int = CLIP_SECONDS,
              fps: int = FPS) -> None:
    """تحريك دوري لإطار واحد: تكبير بطيء ثم عودة (يبدأ وينتهي بالإطار نفسه)."""
    n = seconds * fps
    w, h = VIDEO_SIZE
    zoom = f"1+{zoom_amount() * 0.6:.3f}*(1-cos(2*PI*on/{n}))"
    x = f"iw/2-(iw/zoom/2)+sin(2*PI*on/{n})*iw*0.010"
    y = f"ih/2-(ih/zoom/2)+(cos(2*PI*on/{n})-1)*ih*0.008"
    vf = (f"scale={w * 3}:{h * 3}:flags=lanczos,"
          f"zoompan=z='{zoom}':x='{x}':y='{y}':d=1:s={w}x{h}:fps={fps},"
          "format=yuv420p")
    _run([ffmpeg_bin(), "-y", "-loop", "1", "-framerate", str(fps), "-i", str(frame),
          "-vf", vf, "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "23", "-g", str(fps * 2), "-r", str(fps), str(clip_path)], "التحريك")


def make_video(frame: Path, wav: Path, out_mp4: Path,
               clip_seconds: int = CLIP_SECONDS, music: Path | None = None) -> dict[str, Any]:
    """وضع الإطار الواحد: مقطع دوري يُكرَّر (قائمة concat) حتى نهاية الصوت."""
    seconds = wav_seconds(wav)
    clip = out_mp4.parent / "clip.mp4"
    make_clip(frame, clip, seconds=clip_seconds)
    repeats = math.ceil(seconds / clip_seconds) + 1
    listing = write_concat_list([clip] * repeats, out_mp4.parent / "concat.txt")
    try:
        return finalize_video(listing, wav, out_mp4, seconds, music)
    finally:
        clip.unlink(missing_ok=True)
        listing.unlink(missing_ok=True)


def make_scene_clip(image: Path, out: Path, seconds: float, index: int,
                    fps: int = FPS) -> None:
    """مقطع مشهد واحد: تكبير أو ابتعاد بطيء مع انسياب جانبي، يظهر ويختفي عبر الأسود."""
    n = max(2, round(seconds * fps))
    w, h = VIDEO_SIZE
    zoom_in = index % 2 == 0
    amount = zoom_amount()
    zoom = (f"1.04+{amount:.3f}*on/{n}" if zoom_in
            else f"{1.04 + amount:.3f}-{amount:.3f}*on/{n}")
    direction = 1 if (index // 2) % 2 == 0 else -1
    x = f"iw/2-(iw/zoom/2)+(on/{n}-0.5)*iw*0.025*{direction}"
    y = "ih/2-(ih/zoom/2)"
    fade = min(0.7, seconds / 4)
    vf = (f"scale={w * 3}:{h * 3}:flags=lanczos,"
          f"zoompan=z='{zoom}':x='{x}':y='{y}':d=1:s={w}x{h}:fps={fps},"
          f"fade=t=in:st=0:d={fade:.2f},fade=t=out:st={max(0.0, seconds - fade):.2f}:d={fade:.2f},"
          "format=yuv420p")
    _run([ffmpeg_bin(), "-y", "-loop", "1", "-framerate", str(fps), "-i", str(image),
          "-vf", vf, "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "23", "-g", str(fps * 2), "-r", str(fps), str(out)], f"مشهد {index + 1}")


def make_slideshow(segments: list[dict[str, Any]], wav: Path, out_mp4: Path,
                   music: Path | None = None) -> dict[str, Any]:
    """
    segments: [{"image": Path, "seconds": float}] بترتيب القصة. تُحوَّل المشاهد إلى مقاطع
    بالتوازي، وتُضم بالنسخ، ثم يُركَّب الصوت. يمدّ آخر مقطع ثانية إضافية لتغطية الذيل.
    """
    seconds = wav_seconds(wav)
    folder = out_mp4.parent / "scene_clips"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    jobs = []
    for index, segment in enumerate(segments):
        extra = 1.0 if index == len(segments) - 1 else 0.0
        jobs.append((segment["image"], folder / f"clip_{index:02d}.mp4",
                     segment["seconds"] + extra, index))
    workers = max(1, min(len(jobs), os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(make_scene_clip, *job) for job in jobs]:
            future.result()
    listing = write_concat_list([job[1] for job in jobs], out_mp4.parent / "concat.txt")
    try:
        return finalize_video(listing, wav, out_mp4, seconds, music)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        listing.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3) الاستضافة والنشر
# ---------------------------------------------------------------------------

def _gh(args: list[str]) -> str:
    try:
        result = subprocess.run(["gh", *args], capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise VideoError("أداة gh غير متوفرة.") from exc
    if result.returncode != 0:
        raise VideoError(f"فشل gh {' '.join(args[:2])}: {result.stderr.strip()[:300]}")
    return result.stdout


def upload_release(mp4: Path, repo: str, tag: str, title: str) -> str:
    """يرفع الفيديو إلى GitHub Releases ويعيد رابط تنزيل عام (يحتاج مستودعًا عامًا)."""
    if not repo:
        raise VideoError("GITHUB_REPOSITORY غير محدد.")
    _gh(["release", "create", tag, str(mp4), "--repo", repo,
         "--title", title[:120], "--notes", "فيديو قصة مسموعة (مؤقت، يُحذف تلقائيًا)."])
    return f"https://github.com/{repo}/releases/download/{tag}/{mp4.name}"


def cleanup_old_releases(repo: str, prefix: str = "story-video-", days: int = 3) -> int:
    """يحذف إصدارات الفيديو القديمة (best-effort) كي لا تتراكم."""
    removed = 0
    try:
        rows = json.loads(_gh(["release", "list", "--repo", repo, "--limit", "100",
                               "--json", "tagName,createdAt"]) or "[]")
    except (VideoError, ValueError):
        return 0
    limit = datetime.now(timezone.utc) - timedelta(days=days)
    for row in rows:
        tag = str(row.get("tagName") or "")
        try:
            created = datetime.fromisoformat(str(row["createdAt"]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if tag.startswith(prefix) and created < limit:
            try:
                _gh(["release", "delete", tag, "--repo", repo, "--yes", "--cleanup-tag"])
                removed += 1
            except VideoError:
                pass
    return removed


def post_video_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    """
    يرسل رابط الفيديو إلى Make مرة واحدة فقط (لا إعادة محاولة: قد تكرر الرفع على يوتيوب).
    يعيد رد Make إن كان JSON، وإلا {} (مثل الرد النصي Accepted أو انتهاء المهلة).
    """
    url = os.getenv("MAKE_VIDEO_WEBHOOK_URL", "").strip()
    if not url:
        raise VideoError("سر MAKE_VIDEO_WEBHOOK_URL غير موجود.")
    headers = {"Content-Type": "application/json"}
    secret = (os.getenv("MAKE_VIDEO_WEBHOOK_SECRET", "").strip()
              or os.getenv("MAKE_WEBHOOK_SECRET", "").strip())
    if secret:
        headers["x-make-apikey"] = secret
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        response = requests.post(url, headers=headers, data=body, timeout=60)
    except requests.Timeout:
        print("تنبيه: انتهت مهلة رد Make (قد يكون يرفع الفيديو)؛ نراقب قناة يوتيوب.")
        return {}
    except requests.RequestException as exc:
        raise VideoError(f"تعذر الاتصال بـ Make: {exc}") from exc
    if response.status_code != 200:
        raise VideoError(f"خطأ Make HTTP {response.status_code}: {response.text[:300]}")
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def url_from_reply(reply: dict[str, Any]) -> str | None:
    """رابط يوتيوب من رد Make: youtube_url مباشرة، أو video_id."""
    direct = str(reply.get("youtube_url") or "").strip()
    if direct.startswith("http"):
        return direct
    video_id = str(reply.get("video_id") or reply.get("youtube_id") or "").strip()
    if re.fullmatch(r"[\w-]{6,20}", video_id):
        return f"https://www.youtube.com/watch?v={video_id}"
    return None


def _norm_title(text: str) -> str:
    return re.sub(r"\s+", " ", _plain(text)).strip().lower()


def find_in_feed(xml_text: str, title: str, since: datetime) -> str | None:
    """رابط الفيديو من خلاصة RSS للقناة: العنوان مطابق ونُشر بعد since."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    atom, yt = "{http://www.w3.org/2005/Atom}", "{http://www.youtube.com/xml/schemas/2015}"
    wanted = _norm_title(title)
    for entry in root.iter(f"{atom}entry"):
        entry_title = (entry.findtext(f"{atom}title") or "")
        video_id = (entry.findtext(f"{yt}videoId") or "").strip()
        published = (entry.findtext(f"{atom}published") or "").strip()
        if not video_id or _norm_title(entry_title) != wanted:
            continue
        try:
            when = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if when < since - timedelta(minutes=5):
                continue
        except ValueError:
            pass
        return f"https://www.youtube.com/watch?v={video_id}"
    return None


def wait_for_video(channel_id: str, title: str, since: datetime,
                   minutes: int, interval: int = 30) -> str | None:
    """يراقب خلاصة القناة حتى يظهر الفيديو أو تنتهي المهلة."""
    feed = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    deadline = time.time() + minutes * 60
    while True:
        try:
            response = requests.get(feed, timeout=30)
            if response.status_code == 200:
                found = find_in_feed(response.text, title, since)
                if found:
                    return found
        except requests.RequestException:
            pass
        if time.time() + interval > deadline:
            return None
        time.sleep(interval)


# ---------------------------------------------------------------------------
# 4) روابط الفيديو داخل القصة
# ---------------------------------------------------------------------------

def _intro_end(body: list[str]) -> int:
    """
    موضع رابط الاستماع: بعد مقدمة القصة. المقدمة هي الفقرات التي تسبق أول عنوان فرعي
    إن جاء خلال أول ثلاث فقرات، وإلا فالفقرة الأولى وحدها. يعيد فهرس الكتلة التي
    يوضع الرابط قبلها.
    """
    first = 0
    while first < len(body) and body[first].lstrip().startswith("#"):
        first += 1                                    # عناوين تسبق المقدمة نفسها
    for index in range(first + 1, min(first + 4, len(body))):
        if body[index].lstrip().startswith("#"):
            return index
    return min(first + 1, len(body))


def insert_video_links(markdown: str, url: str) -> str:
    """
    يضيف رابط «القصة مسموعة» بعد مقدمة القصة، ثم في وسطها، ثم في آخرها
    (قبل سطور «اقرأ أيضًا»).
    """
    if url in markdown:
        return markdown
    top = f"**🎧 [استمع إلى هذه القصة مسموعة بالفيديو على يوتيوب]({url})**"
    mid = f"🎧 [تفضّل الاستماع؟ شاهد القصة مسموعة على يوتيوب]({url})"
    end = f"**🎧 [استمع إلى القصة كاملة مسموعة على يوتيوب]({url})**"

    blocks = re.split(r"\n{2,}", markdown.strip())
    start = 1 if blocks and blocks[0].lstrip().startswith("# ") else 0
    stop = len(blocks)
    while stop > start and RELATED_MARK in _plain(blocks[stop - 1]):
        stop -= 1
    body = blocks[start:stop]
    if not body:
        return markdown

    intro = _intro_end(body)

    # الوسط: بعد الرابط الأول بكتلة على الأقل وقبل النهاية بكتلة على الأقل.
    mid_index = None
    if len(body) >= 3:
        total = sum(len(b) for b in body)
        running, positions = 0, []
        for index, block in enumerate(body[:-1], start=1):
            running += len(block)
            if index > intro:
                positions.append((index, running, body[index].lstrip().startswith("## ")))
        pool = [p for p in positions if p[2]] or positions
        if pool:
            mid_index = min(pool, key=lambda p: abs(p[1] - total / 2))[0]

    if mid_index is not None:                          # الأبعد أولًا كي لا تتزحزح الفهارس
        body = body[:mid_index] + [mid] + body[mid_index:]
    if intro < len(body):                              # إن كانت القصة كلها مقدمة يكفي رابط النهاية
        body = body[:intro] + [top] + body[intro:]

    merged = blocks[:start] + body + [end] + blocks[stop:]
    return "\n\n".join(merged).strip() + "\n"


def patch_outputs(out_dir: Path, url: str, extra: dict[str, Any]) -> None:
    md_path = out_dir / "rewritten_article.md"
    md_path.write_text(insert_video_links(md_path.read_text(encoding="utf-8"), url),
                       encoding="utf-8")
    result_path = out_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["youtube_url"] = url
    result["video"] = extra
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


# ---------------------------------------------------------------------------
# التشغيل
# ---------------------------------------------------------------------------

YOUTUBE_DESCRIPTION_BYTES = 4900     # حد يوتيوب 5000 بايت (والحرف العربي بايتان)
MIN_CHAPTER_SECONDS = 10             # أقل طول لفصل في فصول يوتيوب
MAX_HASHTAGS = 12                    # أكثر من 15 يجعل يوتيوب يتجاهلها كلها


def _fit_bytes(text: str, limit: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", errors="ignore").rstrip()


def _hashtag(text: str) -> str:
    """وسم يوتيوب: بلا مسافات ولا رموز (المسافة تصبح _)، ولا يكون أرقامًا فقط."""
    word = re.sub(r"[^\w]+", "_", _plain(str(text or ""))).strip("_")
    return f"#{word}" if word and not word.isdigit() else ""


def build_hashtags(result: dict[str, Any], limit: int = MAX_HASHTAGS) -> list[str]:
    """أول ثلاثة وسوم تظهر فوق عنوان الفيديو، لذا تبدأ بالكلمة المفتاحية ثم التصنيف."""
    seo = result.get("seo") if isinstance(result.get("seo"), dict) else {}
    candidates = [seo.get("focus_keyword"), result.get("category"), "قصص مسموعة",
                  story_categories.PARENT_LABEL, *(seo.get("keywords") or []),
                  *[str(h).lstrip("#") for h in (result.get("hashtags") or [])]]
    tags: list[str] = []
    for item in candidates:
        tag = _hashtag(item)
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:limit]


def _stamp(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def build_chapters(markdown: str, title: str, info: dict[str, Any],
                   audio_seconds: float) -> list[str]:
    """
    فصول يوتيوب من العناوين الفرعية (##) في القصة، بتوقيتها الفعلي في الصوت المروي.
    تبدأ بـ «المقدمة» عند 00:00. يوتيوب لا يفعّل الفصول إلا بثلاثة فصول فأكثر وبطول
    10 ثوانٍ لكل فصل على الأقل، وإلا تعيد قائمة فارغة فلا يُكتب شيء.
    """
    raw_blocks = [b.strip() for b in re.split(r"\n{2,}", markdown or "")]
    names = {re.sub(r"^#{1,6}\s*", "", b).replace("**", "").replace("*", "").strip()
             for b in raw_blocks if re.match(r"##+\s", b)}
    if not names:
        return []
    paragraphs = story_paragraphs(markdown)
    starts = story_scenes.paragraph_start_times(
        paragraphs, title, info["chunk_texts"], info["chunk_seconds"], PAUSE_SECONDS)

    chapters: list[tuple[float, str]] = [(0.0, "المقدمة")]
    for paragraph, moment in zip(paragraphs, starts):
        if paragraph not in names:
            continue
        if (moment - chapters[-1][0] >= MIN_CHAPTER_SECONDS
                and audio_seconds - moment >= MIN_CHAPTER_SECONDS):
            chapters.append((moment, paragraph[:70]))
    if len(chapters) < 3:
        return []
    return [f"{_stamp(moment)} {name}" for moment, name in chapters]


def build_description(result: dict[str, Any], chapters: list[str] | None = None) -> str:
    """
    وصف يوتيوب خاص بكل قصة: عنوانها ووصفها، فصولها بتوقيتها، روابط القسم والتصنيف،
    الكلمات المفتاحية، ثم الوسوم في الآخر. يُقتطع بالبايت لا بالحروف كي لا يرفضه يوتيوب.
    """
    seo = result.get("seo") if isinstance(result.get("seo"), dict) else {}
    title = str(result.get("title") or "").strip()
    category = str(result.get("category") or "").strip()
    meta = str(seo.get("meta_description") or "").strip()
    keywords = [str(k).strip() for k in (seo.get("keywords") or []) if str(k).strip()]

    parts = [f"🎧 {title} | قصة مسموعة" + (f" — {category}" if category else "")]
    if meta:
        parts.append(meta)
    summary = str(seo.get("video_summary") or "").strip()
    if summary and summary != meta:                     # ملخص خاص بالفيديو يكتبه كاتب القصة
        parts.append(summary)
    if chapters:
        parts.append("⏱ فصول القصة:\n" + "\n".join(chapters))

    blog = os.getenv("STORY_BLOG_URL", "").strip().rstrip("/")
    if blog:
        links = ["📖 اقرأ القصص مكتوبة على مدونتنا:"]
        links.append(f"{story_categories.PARENT_LABEL}: "
                     f"{blog}/search/label/{quote(story_categories.PARENT_LABEL)}")
        if category:
            links.append(f"{category}: {blog}/search/label/{quote(category)}")
        parts.append("\n".join(links))

    parts.append("🔔 إن أعجبتك القصة فاشترك في القناة وفعّل الجرس ليصلك كل جديد، "
                 "وشاركنا رأيك في التعليقات.")
    if keywords:
        parts.append("🔎 الكلمات المفتاحية: " + "، ".join(keywords))
    parts.append("⚠ القصة من وحي الخيال، وأي تشابه مع أشخاص أو وقائع حقيقية غير مقصود.")

    tail = " ".join(build_hashtags(result))
    body = "\n\n".join(p for p in parts if p).replace("<", "").replace(">", "")
    budget = YOUTUBE_DESCRIPTION_BYTES - len(tail.encode("utf-8")) - 2
    return (_fit_bytes(body, budget) + ("\n\n" + tail if tail else "")).strip()


def build_tags(result: dict[str, Any]) -> list[str]:
    """وسوم الفيديو المخفية (حدها 500 حرف؛ الوسم متعدد الكلمات يُحسب بعلامتي اقتباس)."""
    seo = result.get("seo") if isinstance(result.get("seo"), dict) else {}
    tags, total = [], 0
    for tag in [seo.get("focus_keyword"), *(seo.get("keywords") or []), "قصص", "قصة مسموعة",
                story_categories.PARENT_LABEL, str(result.get("category") or "")]:
        tag = str(tag or "").strip().replace("<", "").replace(">", "")
        cost = len(tag) + (2 if " " in tag else 0) + 1
        if tag and tag not in tags and total + cost < 450:
            tags.append(tag)
            total += cost
    return tags


def build_visuals(result: dict[str, Any], markdown: str, info: dict[str, Any],
                  work_dir: Path, audio_seconds: float) -> list[dict[str, Any]]:
    """
    مقاطع الصور بحسب لوحة المشاهد: [{"image", "seconds"}]. تعيد قائمة فارغة إن لم توجد
    لوحة أو فشل توليد صور كافية (فيرجع الفيديو إلى إطار الغلاف الواحد).
    """
    board = result.get("storyboard")
    if not story_scenes.scenes_enabled() or not isinstance(board, dict) or not board.get("scenes"):
        print("لا لوحة مشاهد؛ يُستعمل إطار الغلاف وحده.")
        return []
    print(f"توليد صور المشاهد ({len(board['scenes'])})...")
    try:
        rows = story_scenes.generate_scene_images(board, str(result.get("category")), work_dir)
    except Exception as exc:  # noqa: BLE001 - الصور إضافة: نرجع إلى الإطار الواحد
        print(f"::warning title=صور المشاهد::{type(exc).__name__}: {exc}")
        return []
    generated = sum(1 for r in rows if r["generated"])
    if generated < 2:
        print("::warning title=صور المشاهد::لم تنجح إلا صورة أو أقل؛ يُستعمل إطار الغلاف.")
        return []

    paragraphs = story_paragraphs(markdown)
    starts = story_scenes.paragraph_start_times(
        paragraphs, str(result.get("title") or ""), info["chunk_texts"],
        info["chunk_seconds"], PAUSE_SECONDS)
    segments = story_scenes.scene_segments(board["scenes"], starts, audio_seconds)
    visuals = [{"image": rows[s["scene"]]["path"], "seconds": s["seconds"]} for s in segments]
    print(f"مشاهد الفيديو: {len(visuals)} (مولّدة {generated} من {len(rows)}) | "
          "مدد (ث): " + ", ".join(f"{v['seconds']:.0f}" for v in visuals))
    info["scenes"] = {"total": len(rows), "generated": generated, "shown": len(visuals)}
    return visuals


def run(out_dir: Path, work_dir: Path) -> dict[str, Any]:
    mode = video_mode()
    if mode == "off":
        print("STORY_VIDEO=off؛ يُتخطى الفيديو.")
        return {"mode": mode}

    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    if result.get("youtube_url"):
        print(f"للقصة فيديو سابق ({result['youtube_url']})؛ لا حاجة لإعادة الإنتاج.")
        return {"mode": mode, "url": result["youtube_url"]}

    frame = work_dir / "video_frame.jpg"
    if not frame.is_file():
        raise VideoError(f"إطار الفيديو غير موجود: {frame}")
    if mode == "full" and not os.getenv("MAKE_VIDEO_WEBHOOK_URL", "").strip():
        raise VideoError("STORY_VIDEO=full يحتاج السر MAKE_VIDEO_WEBHOOK_URL.")

    markdown = (out_dir / "rewritten_article.md").read_text(encoding="utf-8")
    text = clean_story_text(markdown)
    video_dir = work_dir / "video_out"
    video_dir.mkdir(parents=True, exist_ok=True)
    wav, mp4 = video_dir / "narration.wav", video_dir / f"story-{out_dir.name}.mp4"

    info = synthesize(text, wav, title=str(result.get("title") or ""))
    visuals = build_visuals(result, markdown, info, work_dir, info["seconds"])
    music = pick_music(out_dir.parent)
    if visuals:
        report = make_slideshow(visuals, wav, mp4, music)
    else:
        report = make_video(frame, wav, mp4,
                            clip_seconds=_env_int("STORY_CLIP_SECONDS", CLIP_SECONDS, 4, 600),
                            music=music)
    video_seconds, audio_seconds = media_seconds(mp4), info["seconds"]
    print(f"الفيديو: {mp4.name} ({mp4.stat().st_size / 1e6:.1f} ميجا، "
          f"{video_seconds / 60:.1f} دقيقة) | المسارات: " + " | ".join(report["streams"]))
    if abs(video_seconds - audio_seconds) > 5:
        raise VideoError(f"مدة الفيديو ({video_seconds:.0f}ث) لا تطابق الصوت ({audio_seconds:.0f}ث).")

    # ملف الصوت المضغوط للفحص والأرشفة
    _run([ffmpeg_bin(), "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "96k",
          str(video_dir / "narration.m4a")], "ضغط الصوت")
    wav.unlink(missing_ok=True)

    details = {"mode": mode, "minutes": round(video_seconds / 60, 1),
               "mb": round(mp4.stat().st_size / 1e6, 1), "audio_mean_db": report["mean"],
               "music": report.get("music"), "scenes": info.get("scenes")}
    if mode == "local":
        print(f"الوضع local: الفيديو في {video_dir} (يُرفع كـ artifact للفحص فقط).")
        return details

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    cleanup_old_releases(repo)
    title = f"{result.get('title', '')} | قصة مسموعة"[:100]
    tag = f"story-video-{out_dir.name}"
    video_url = upload_release(mp4, repo, tag, title)
    print(f"رابط الفيديو للتنزيل: {video_url}")

    try:
        chapters = build_chapters(markdown, str(result.get("title") or ""), info,
                                  info["seconds"])
    except Exception as exc:  # noqa: BLE001 - الفصول إضافة: الوصف يعمل بدونها
        print(f"::warning title=فصول الفيديو::{type(exc).__name__}: {exc}")
        chapters = []
    if chapters:
        print("فصول الفيديو:\n  " + "\n  ".join(chapters))

    started = datetime.now(timezone.utc)
    payload = {
        "title": title,
        "description": build_description(result, chapters),
        "tags": build_tags(result),
        "tags_csv": ", ".join(build_tags(result)),
        "video_url": video_url,
        "category": result.get("category"),
        "language": "ar",
        "privacy": "public",
        "made_for_kids": False,
        "story_title": result.get("title"),
    }
    reply = post_video_webhook(payload)
    youtube_url = url_from_reply(reply)

    if not youtube_url:
        channel = os.getenv("YOUTUBE_CHANNEL_ID", "").strip()
        if not channel:
            raise VideoError("لم يردّ Make برابط يوتيوب، ولا YOUTUBE_CHANNEL_ID لمراقبة القناة.")
        minutes = _env_int("STORY_VIDEO_WAIT_MINUTES", 20, 1, 60)
        print(f"ننتظر ظهور الفيديو في قناة يوتيوب (حتى {minutes} دقيقة)...")
        youtube_url = wait_for_video(channel, title, started, minutes)
    if not youtube_url:
        raise VideoError("لم يظهر الفيديو على يوتيوب خلال المهلة (تأكد أنه عام لا خاص).")

    details["youtube_url"] = youtube_url
    patch_outputs(out_dir, youtube_url, details)
    print(f"رابط يوتيوب: {youtube_url}\nوُضع رابط الفيديو في القصة: بعد المقدمة ووسطها وآخرها.")
    return details


def main() -> int:
    out = os.getenv("OUT_DIR", "").strip()
    work = os.getenv("WORK_DIR", "").strip()
    if not out or not work:
        print("::error::OUT_DIR و WORK_DIR مطلوبان.", file=sys.stderr)
        return 1
    try:
        run(Path(out), Path(work))
    except VideoError as exc:
        print(f"::warning title=تعذر إنتاج فيديو القصة::{exc}")
    except Exception as exc:  # noqa: BLE001 - الفيديو إضافة: لا يمنع نشر القصة
        print(f"::warning title=عطل غير متوقع في فيديو القصة::{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
