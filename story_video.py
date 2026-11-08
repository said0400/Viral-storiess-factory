"""
فيديو القصة: يحوّل نص القصة إلى صوت بـ Gemini TTS، ويركّبه على صورة 16:9 بلا نص مع
تحريك بسيط (تكبير وانسياب بطيئان)، ثم ينشره على يوتيوب عبر Make ويضع رابطه في القصة.

المراحل:
  1) تنظيف النص وتقسيمه إلى مقاطع (حد المخرج الصوتي في الطلب الواحد) ثم تحويل كل مقطع.
  2) ffmpeg: مقطع تحريك دوري (ping-pong) يُكرَّر بلا إعادة ترميز + الصوت = story.mp4
  3) رفع story.mp4 إلى GitHub Releases (رابط عام) فيُرسَل الرابط إلى Make لا الملف.
  4) Make يرفع إلى يوتيوب. نقرأ رابط الفيديو من رد Make (youtube_url أو video_id)،
     وإلا نراقب خلاصة RSS لقناتك (YOUTUBE_CHANNEL_ID) حتى يظهر الفيديو بعنوانه.
  5) نضع رابط الفيديو في القصة: أعلاها ووسطها وآخرها، ونحفظه في result.json.

STORY_TTS_STYLE / STORY_TTS_OUTRO: النص الافتراضي إن تُركا فارغين، و none يعطّلهما.
STORY_VIDEO: off = تعطيل | local = صوت وفيديو للفحص فقط (بلا رفع) | full = كل المراحل.
الافتراضي (فارغ): full إن وُجد السر MAKE_VIDEO_WEBHOOK_URL وإلا local.

الفيديو إضافة اختيارية: أي فشل هنا يُسجَّل تحذيرًا ولا يمنع نشر القصة.

التشغيل: OUT_DIR=output/<hash> WORK_DIR=story_work/<hash> python story_video.py
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


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
CLIP_SECONDS = 60            # طول التحريك الدوري (يُكرَّر حتى نهاية الصوت)
CLIP_FPS = 15
VIDEO_SIZE = (1280, 720)

RELATED_MARK = "اقرأ أيضا"   # بلا تشكيل، للمقارنة
VIDEO_MARK = "🎧"

_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


class VideoError(Exception):
    """خطأ في إنتاج الفيديو أو نشره."""


def _plain(text: str) -> str:
    return _DIACRITICS.sub("", text or "")


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
# 1) النص: تنظيف وتقسيم
# ---------------------------------------------------------------------------

def clean_story_text(markdown: str) -> str:
    """نص صالح للقراءة بصوت عالٍ: بلا Markdown ولا روابط ولا سطور «اقرأ أيضًا» أو الروابط الصوتية."""
    lines: list[str] = []
    for raw in (markdown or "").splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith("# "):                       # العنوان الرئيسي (يُقرأ منفصلًا)
            continue
        plain = _plain(line)
        if RELATED_MARK in plain or VIDEO_MARK in line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = _URL.sub("", line).replace("**", "").replace("*", "")
        lines.append(line.strip())
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!؟?…؛])\s+", paragraph)
    return [p for p in parts if p.strip()]


def split_chunks(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """مقاطع بحدود الفقرات (وبحدود الجمل للفقرة الطويلة جدًا)، كل منها حتى max_words كلمة."""
    units: list[str] = []
    for para in re.split(r"\n{2,}", text.strip()):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) <= max_words:
            units.append(para)
        else:
            buf: list[str] = []
            for sentence in _split_sentences(para):
                if buf and len(" ".join(buf + [sentence]).split()) > max_words:
                    units.append(" ".join(buf))
                    buf = []
                buf.append(sentence)
            if buf:
                units.append(" ".join(buf))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and len("\n\n".join(current + [unit]).split()) > max_words:
            chunks.append("\n\n".join(current))
            current = []
        current.append(unit)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# 2) الصوت: Gemini TTS
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
    """يحوّل النص كله إلى WAV واحد (مقاطع مفصولة بصمت قصير) ويعيد معلومات عنه."""
    outro = _text_setting("STORY_TTS_OUTRO", DEFAULT_OUTRO)
    spoken = (f"{title.strip()}.\n\n" if title.strip() else "") + text
    if outro:
        spoken += "\n\n" + outro
    chunks = split_chunks(spoken, _env_int("STORY_TTS_CHUNK_WORDS", CHUNK_WORDS, 100, 700))
    if not chunks:
        raise VideoError("لا نص للتحويل إلى صوت.")
    print(f"الصوت: {len(chunks)} مقاطع.")

    pcm_parts: list[bytes] = []
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
    return {"chunks": len(chunks), "seconds": seconds, "rate": rate}


# ---------------------------------------------------------------------------
# 3) الفيديو: ffmpeg
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


def make_clip(frame: Path, clip_path: Path, seconds: int = CLIP_SECONDS,
              fps: int = CLIP_FPS) -> None:
    """تحريك دوري: تكبير بطيء ثم عودة (يبدأ وينتهي بالإطار نفسه) مع انسياب خفيف جدًا."""
    n = seconds * fps
    w, h = VIDEO_SIZE
    zoom = f"1+0.06*(1-cos(2*PI*on/{n}))"
    x = f"iw/2-(iw/zoom/2)+sin(2*PI*on/{n})*iw*0.010"
    y = f"ih/2-(ih/zoom/2)+(cos(2*PI*on/{n})-1)*ih*0.008"
    vf = (f"scale={w * 3}:{h * 3}:flags=lanczos,"
          f"zoompan=z='{zoom}':x='{x}':y='{y}':d=1:s={w}x{h}:fps={fps},"
          "format=yuv420p")
    _run([ffmpeg_bin(), "-y", "-loop", "1", "-framerate", str(fps), "-i", str(frame),
          "-vf", vf, "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast",
          "-crf", "23", "-g", str(fps * 2), "-r", str(fps), str(clip_path)], "التحريك")


def make_video(frame: Path, wav: Path, out_mp4: Path,
               clip_seconds: int = CLIP_SECONDS) -> None:
    """يكرّر مقطع التحريك (نسخًا بلا إعادة ترميز) ويركّب عليه الصوت حتى نهاية الصوت."""
    with wave.open(str(wav)) as handle:
        audio_seconds = handle.getnframes() / handle.getframerate()
    clip = out_mp4.parent / "clip.mp4"
    make_clip(frame, clip, seconds=clip_seconds)
    # مهم: -t صريح بمدة الصوت. مع التكرار بالنسخ (-stream_loop + -c copy) لا يوقف
    # -shortest الفيديو فيستمر بلا نهاية (اختبرنا ذلك: ملف بعشرات الجيجابايت).
    _run([ffmpeg_bin(), "-y", "-stream_loop", "-1", "-i", str(clip), "-i", str(wav),
          "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
          "-c:a", "aac", "-b:a", "96k", "-ar", "44100", "-ac", "1",
          "-t", f"{audio_seconds:.3f}", "-movflags", "+faststart", str(out_mp4)],
         "التركيب")
    clip.unlink(missing_ok=True)


def media_seconds(path: Path) -> float:
    """مدة ملف وسائط بالثواني (من مخرج ffmpeg -i)."""
    result = subprocess.run([ffmpeg_bin(), "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise VideoError(f"تعذر قراءة مدة {path.name}.")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


# ---------------------------------------------------------------------------
# 4) الاستضافة والنشر
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
# 5) روابط الفيديو داخل القصة
# ---------------------------------------------------------------------------

def insert_video_links(markdown: str, url: str) -> str:
    """يضيف رابط «القصة مسموعة» أول القصة ووسطها وآخرها (قبل سطور «اقرأ أيضًا»)."""
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

    if len(body) >= 3:
        total = sum(len(b) for b in body)
        running, positions = 0, []
        for index, block in enumerate(body[:-1], start=1):
            running += len(block)
            positions.append((index, running, body[index].lstrip().startswith("## ")))
        headings = [p for p in positions if p[2]]
        pool = headings or positions
        index = min(pool, key=lambda p: abs(p[1] - total / 2))[0]
        body = body[:index] + [mid] + body[index:]

    merged = blocks[:start] + [top] + body + [end] + blocks[stop:]
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

def build_description(result: dict[str, Any]) -> str:
    seo = result.get("seo") if isinstance(result.get("seo"), dict) else {}
    parts = [str(seo.get("meta_description") or "").strip(),
             str(result.get("category") or "").strip(),
             " ".join(result.get("hashtags") or [])]
    blog = os.getenv("STORY_BLOG_URL", "").strip().rstrip("/")
    if blog:
        parts.append(f"اقرأ القصص مكتوبة على مدونتنا: {blog}/search/label/قصص")
    return "\n\n".join(p for p in parts if p)[:4500]


def build_tags(result: dict[str, Any]) -> list[str]:
    seo = result.get("seo") if isinstance(result.get("seo"), dict) else {}
    tags, total = [], 0
    for tag in [*(seo.get("keywords") or []), "قصص", "قصة مسموعة",
                str(result.get("category") or "")]:
        tag = str(tag).strip()
        if tag and tag not in tags and total + len(tag) < 450:
            tags.append(tag)
            total += len(tag)
    return tags


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
    make_video(frame, wav, mp4, clip_seconds=_env_int("STORY_CLIP_SECONDS", CLIP_SECONDS, 4, 600))
    video_seconds, audio_seconds = media_seconds(mp4), info["seconds"]
    print(f"الفيديو: {mp4.name} ({mp4.stat().st_size / 1e6:.1f} ميجا، "
          f"{video_seconds / 60:.1f} دقيقة)")
    if abs(video_seconds - audio_seconds) > 5:
        raise VideoError(f"مدة الفيديو ({video_seconds:.0f}ث) لا تطابق الصوت ({audio_seconds:.0f}ث).")

    # ملف الصوت المضغوط للفحص والأرشفة
    _run([ffmpeg_bin(), "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "96k",
          str(video_dir / "narration.m4a")], "ضغط الصوت")
    wav.unlink(missing_ok=True)

    details = {"mode": mode, "minutes": round(video_seconds / 60, 1),
               "mb": round(mp4.stat().st_size / 1e6, 1)}
    if mode == "local":
        print(f"الوضع local: الفيديو في {video_dir} (يُرفع كـ artifact للفحص فقط).")
        return details

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    cleanup_old_releases(repo)
    title = f"{result.get('title', '')} | قصة مسموعة"[:100]
    tag = f"story-video-{out_dir.name}"
    video_url = upload_release(mp4, repo, tag, title)
    print(f"رابط الفيديو للتنزيل: {video_url}")

    started = datetime.now(timezone.utc)
    payload = {
        "title": title,
        "description": build_description(result),
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
    print(f"رابط يوتيوب: {youtube_url}\nوُضع رابط الفيديو في القصة: أعلاها ووسطها وآخرها.")
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
