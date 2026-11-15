"""Shared FFmpeg/FFprobe discovery and audio probing helpers."""
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

PROBE_TIMEOUT_SECONDS = 20


class MediaToolError(RuntimeError):
    """Raised when FFmpeg itself is unavailable (a server problem, not a bad upload)."""


@dataclass(frozen=True)
class AudioProbe:
    duration: float
    has_audio: bool


def _venv_tool(name: str) -> Optional[str]:
    scripts_dir = Path(sys.executable).parent
    for candidate in (scripts_dir / f"{name}.exe", scripts_dir / name):
        if candidate.is_file():
            return str(candidate)
    return None


@lru_cache(maxsize=None)
def find_ffmpeg() -> str:
    """Locate ffmpeg: virtualenv, then PATH, then the imageio-ffmpeg bundled binary."""
    found = _venv_tool("ffmpeg") or shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # ImportError or "binary not found"
        raise MediaToolError(
            "FFmpeg was not found. Install FFmpeg or run `pip install imageio-ffmpeg`."
        ) from exc


@lru_cache(maxsize=None)
def find_ffprobe() -> Optional[str]:
    """Locate ffprobe if available. imageio-ffmpeg does not ship it, so this may be None."""
    found = _venv_tool("ffprobe") or shutil.which("ffprobe")
    if found:
        return found
    try:
        ffmpeg_dir = Path(find_ffmpeg()).parent
    except MediaToolError:
        return None
    for candidate in (ffmpeg_dir / "ffprobe.exe", ffmpeg_dir / "ffprobe"):
        if candidate.is_file():
            return str(candidate)
    return None


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_AUDIO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Audio:")


def parse_ffmpeg_info(stderr: str) -> AudioProbe:
    """Parse the banner `ffmpeg -i <file>` prints to stderr (used when ffprobe is missing)."""
    match = _DURATION_RE.search(stderr)
    duration = 0.0
    if match:
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return AudioProbe(duration=duration, has_audio=bool(_AUDIO_STREAM_RE.search(stderr)))


def probe_audio(path: Path) -> AudioProbe:
    """Return duration and whether the file has an audio stream.

    Raises MediaToolError if no FFmpeg tooling exists, ValueError if the file is unreadable.
    """
    ffprobe = find_ffprobe()
    try:
        if ffprobe:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                 "-of", "json", str(path)],
                capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                raise ValueError("The uploaded file is not a valid audio file.")
            media = json.loads(result.stdout or "{}")
            has_audio = any(s.get("codec_type") == "audio" for s in media.get("streams", []))
            raw_duration = media.get("format", {}).get("duration")
            duration = float(raw_duration) if raw_duration not in (None, "N/A") else 0.0
            return AudioProbe(duration=duration, has_audio=has_audio)

        # `ffmpeg -i` with no output exits non-zero by design; the stream info is on stderr.
        result = subprocess.run(
            [find_ffmpeg(), "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS,
            encoding="utf-8", errors="replace",
        )
        return parse_ffmpeg_info(result.stderr or "")
    except (subprocess.SubprocessError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("The uploaded file could not be validated as audio.") from exc
    except FileNotFoundError as exc:
        raise MediaToolError(f"Could not run FFmpeg tooling: {exc}") from exc
