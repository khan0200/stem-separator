import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from app.media_tools import MediaToolError, find_ffmpeg, probe_audio
from app.separator_engine import MASTER_EXT, DualModeSeparatorEngine, SeparationCancelled
from app.tempo import estimate_bpm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stem_separator")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("STEM_DATA_DIR", BASE_DIR / "data" / "tasks"))
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

TASK_RETENTION_SECONDS = int(os.environ.get("TASK_RETENTION_HOURS", "24")) * 60 * 60
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("CLEANUP_INTERVAL_MINUTES", "60")) * 60
# Task folders without metadata are uploads that never finished; drop them after this.
ORPHAN_MAX_AGE_SECONDS = 60 * 60
MAX_QUEUED_JOBS = max(0, int(os.environ.get("MAX_QUEUED_JOBS", "2")))
# A job that is still "processing" after this many server starts has most likely
# crashed the process (e.g. out of memory) and must not be retried forever.
MAX_JOB_ATTEMPTS = max(1, int(os.environ.get("MAX_JOB_ATTEMPTS", "2")))
MAX_CONCURRENT_EXPORTS = max(1, int(os.environ.get("MAX_CONCURRENT_EXPORTS", "2")))
EXPORT_TIMEOUT_SECONDS = 15 * 60
FFMPEG_TIMEOUT_SECONDS = 10 * 60

MAX_FILE_SIZE = 75 * 1024 * 1024  # 75 MB
MULTIPART_OVERHEAD = 1024 * 1024
MIN_DURATION_SECONDS = 10
MAX_DURATION_SECONDS = 10 * 60
UPLOAD_CHUNK_SIZE = 1024 * 1024
PREVIEW_BITRATE = "256k"
WAVEFORM_BARS = 120

ALLOWED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")
ALLOWED_MP3_QUALITIES = {"320k", "256k", "192k"}
ALLOWED_WAV_QUALITIES = {"wav24", "wav16"}
ALLOWED_STEM_NAMES = {"vocals", "drums", "bass", "other", "no_vocals", "instrumental"}
MODE_STEMS = {
    "2stems": ["vocals", "instrumental"],
    "4stems": ["vocals", "drums", "bass", "other"],
}
ACTIVE_STATUSES = {"pending", "processing"}
TERMINAL_STATUSES = {"completed", "error", "cancelled"}
UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


class JobCancelled(SeparationCancelled):
    """Raised when a user cancels a queued or running separation."""


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
class UploadTooLarge(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=413,
            detail=f"Audio file exceeds the maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )


class UploadSizeLimitMiddleware:
    """Reject oversized uploads before the multipart body is spooled to disk.

    FastAPI parses the whole form before the endpoint runs, so a size check inside
    the endpoint only fires after the full upload has been received and stored.
    """

    def __init__(self, app, max_body_size: int, paths: set):
        self.app = app
        self.max_body_size = max_body_size
        self.paths = paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in self.paths:
            await self.app(scope, receive, send)
            return

        content_length = dict(scope.get("headers") or []).get(b"content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > self.max_body_size
            except ValueError:
                too_large = False
            if too_large:
                error = UploadTooLarge()
                response = JSONResponse({"detail": error.detail}, status_code=error.status_code,
                                        headers={"Connection": "close"})
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_size:
                    # An HTTPException is re-raised by FastAPI's body parsing and rendered as 413.
                    raise UploadTooLarge()
            return message

        await self.app(scope, limited_receive, send)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "media-src 'self' blob:; "
            "img-src 'self' data:; "
            "object-src 'none';"
        )
        return response


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
tasks_db: Dict[str, Dict[str, Any]] = {}
task_lock = threading.RLock()  # Guards tasks_db, cancelled_tasks and meta.json writes.
cancelled_tasks = set()
job_queue: Queue = Queue()
worker_stop = threading.Event()
worker_thread: Optional[threading.Thread] = None
cleanup_thread: Optional[threading.Thread] = None
separator_engine: Optional[DualModeSeparatorEngine] = None

_export_locks: Dict[str, threading.Lock] = {}
_export_locks_guard = threading.Lock()
_export_slots = threading.BoundedSemaphore(MAX_CONCURRENT_EXPORTS)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def task_dir_for(task_id: str) -> Path:
    return DATA_DIR / task_id


def task_meta_path(task_id: str) -> Path:
    return task_dir_for(task_id) / "meta.json"


def stems_dir_for(task_id: str) -> Path:
    return task_dir_for(task_id) / "output" / "stems"


def preview_dir_for(task_id: str) -> Path:
    return task_dir_for(task_id) / "output" / "preview"


def find_input_file(task_id: str) -> Optional[Path]:
    files = sorted(p for p in task_dir_for(task_id).glob("input.*") if p.is_file())
    return files[0] if files else None


def _legacy_stem_dirs(task_id: str) -> List[Path]:
    """Older versions stored MP3 stems in output/stems or output/htdemucs/<name>/."""
    output = task_dir_for(task_id) / "output"
    dirs = [output / "stems"]
    htdemucs = output / "htdemucs"
    if htdemucs.is_dir():
        dirs.extend(sorted(p for p in htdemucs.iterdir() if p.is_dir()))
    return dirs


def master_stem_path(task_id: str, stem: str) -> Optional[Path]:
    """Best-quality source for a stem: lossless master, or a legacy MP3."""
    lossless = stems_dir_for(task_id) / f"{stem}{MASTER_EXT}"
    if lossless.is_file():
        return lossless
    for folder in _legacy_stem_dirs(task_id):
        legacy = folder / f"{stem}.mp3"
        if legacy.is_file():
            return legacy
    return None


def preview_stem_path(task_id: str, stem: str) -> Optional[Path]:
    """Compact MP3 used for in-browser playback."""
    preview = preview_dir_for(task_id) / f"{stem}.mp3"
    if preview.is_file():
        return preview
    master = master_stem_path(task_id, stem)
    return master if master is not None and master.suffix == ".mp3" else None


def is_within_data_dir(path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(DATA_DIR.resolve())
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON so a crash mid-write never leaves a truncated file behind."""
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                # Windows refuses to replace a file another thread has open for reading.
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        tmp_path.unlink(missing_ok=True)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def persist_task(task_id: str) -> None:
    """Save task state to meta.json. Writes are serialized so an older snapshot never wins."""
    with task_lock:
        task_info = tasks_db.get(task_id)
        if task_info is None:
            return
        try:
            write_json_atomic(task_meta_path(task_id), dict(task_info))
        except OSError:
            logger.exception("Could not persist task %s", task_id)


def update_task(task_id: str, persist: bool = False, **changes: Any) -> None:
    with task_lock:
        task_info = tasks_db.get(task_id)
        if task_info is None:
            return
        task_info.update(changes)
        task_info["updated_at"] = time.time()
        if persist:
            persist_task(task_id)


def load_task_from_disk(task_id: str) -> Optional[Dict[str, Any]]:
    """Read a task's saved state. Pure: never writes to disk or memory."""
    meta = read_json(task_meta_path(task_id))
    if meta is None:
        return None
    info = dict(meta)
    info["task_id"] = task_id
    status = info.get("status")

    if status is None:
        # Pre-status metadata from old versions: the ZIP was only written once a
        # job finished, so it's the one reliable sign the separation succeeded.
        zips = list((task_dir_for(task_id) / "output").glob("*.zip"))
        if zips:
            status = "completed"
            info.setdefault("progress", 100)
            info.setdefault("message", "Separation completed successfully!")
        else:
            status = "error"
            info["error"] = "Task data from an older version is incomplete."
        info["status"] = status

    if status == "completed":
        stems = info.get("stems") or MODE_STEMS.get(info.get("mode", "4stems"), [])
        stems = [s for s in stems if master_stem_path(task_id, s) is not None]
        if not stems:
            info.update(status="error", progress=0, message="Separation failed.",
                        error="The separated stem files are missing.")
        info["stems"] = stems
    return info


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Return a snapshot of a task, loading it from disk into the cache if needed."""
    if not UUID_REGEX.match(task_id):
        return None
    with task_lock:
        cached = tasks_db.get(task_id)
        if cached is not None:
            return dict(cached)

    info = load_task_from_disk(task_id)
    if info is None:
        return None
    if info.get("status") in ACTIVE_STATUSES:
        # Every active job is re-queued at startup, so one that isn't in memory
        # is not going to make progress. Report it instead of polling forever.
        info.update(status="error", progress=0, message="Separation failed.",
                    error="Processing was interrupted. Please upload the file again.")
    with task_lock:
        return dict(tasks_db.setdefault(task_id, info))


# ---------------------------------------------------------------------------
# Cleanup & recovery
# ---------------------------------------------------------------------------
def _last_activity(task_dir: Path, meta: Optional[Dict[str, Any]]) -> float:
    if meta:
        for key in ("updated_at", "created_at"):
            value = meta.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    try:
        return task_dir.stat().st_mtime
    except OSError:
        return time.time()


def cleanup_old_tasks(now: Optional[float] = None) -> int:
    """Delete finished tasks past retention and abandoned uploads. Returns the count removed."""
    now = time.time() if now is None else now
    removed = 0
    try:
        task_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and UUID_REGEX.match(d.name)]
    except OSError:
        logger.exception("Could not list %s", DATA_DIR)
        return 0

    for task_dir in task_dirs:
        task_id = task_dir.name
        with task_lock:
            cached = tasks_db.get(task_id)
            meta = dict(cached) if cached is not None else None
        if meta is None:
            meta = read_json(task_dir / "meta.json")

        if meta is None:
            expired = now - _last_activity(task_dir, None) > ORPHAN_MAX_AGE_SECONDS
        elif meta.get("status") in ACTIVE_STATUSES and cached is not None:
            continue
        else:
            expired = now - _last_activity(task_dir, meta) > TASK_RETENTION_SECONDS
        if not expired:
            continue

        with task_lock:
            current = tasks_db.get(task_id)
            if current is not None and current.get("status") in ACTIVE_STATUSES:
                continue
            tasks_db.pop(task_id, None)
            cancelled_tasks.discard(task_id)
        shutil.rmtree(task_dir, ignore_errors=True)
        if task_dir.exists():
            logger.warning("Could not fully remove expired task %s", task_id)
        else:
            removed += 1
            logger.info("Removed expired task %s", task_id)
    return removed


def recover_tasks() -> None:
    """Re-queue jobs interrupted by a restart, oldest first, with a retry limit."""
    recovered = []
    for task_dir in DATA_DIR.iterdir():
        if not task_dir.is_dir() or not UUID_REGEX.match(task_dir.name):
            continue
        meta = read_json(task_dir / "meta.json")
        if meta is None:
            continue  # Orphan; cleanup_old_tasks removes it once it's old enough.
        if meta.get("status") in ACTIVE_STATUSES:
            recovered.append((meta.get("created_at") or 0, task_dir.name, meta))

    for _, task_id, meta in sorted(recovered):
        input_path = find_input_file(task_id)
        attempts = int(meta.get("attempts") or 0)
        meta["task_id"] = task_id
        with task_lock:
            tasks_db[task_id] = meta
        if input_path is None:
            update_task(task_id, persist=True, status="error", progress=0,
                        message="Separation failed.", error="The uploaded file is missing.")
        elif attempts >= MAX_JOB_ATTEMPTS:
            logger.warning("Task %s was interrupted %d times; giving up.", task_id, attempts)
            update_task(
                task_id, persist=True, status="error", progress=0, message="Separation failed.",
                error=("Processing stopped unexpectedly several times (the server may have run "
                       "out of memory). Try a shorter file or 2-stem mode."),
            )
        else:
            update_task(task_id, persist=True, status="pending", progress=0,
                        message="Recovered after server restart.")
            enqueue_job(task_id, input_path, meta.get("mode", "4stems"), allow_when_full=True)
            logger.info("Re-queued interrupted task %s", task_id)


def cleanup_loop() -> None:
    while not worker_stop.wait(CLEANUP_INTERVAL_SECONDS):
        try:
            cleanup_old_tasks()
        except Exception:
            logger.exception("Periodic cleanup failed")


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------
def task_is_cancelled(task_id: str) -> bool:
    with task_lock:
        return task_id in cancelled_tasks


def mark_cancelled(task_id: str) -> None:
    """Record a cancellation and delete everything except the metadata."""
    task_dir = task_dir_for(task_id)
    for name in ("output", "exports"):
        shutil.rmtree(task_dir / name, ignore_errors=True)
    input_path = find_input_file(task_id)
    if input_path is not None:
        try:
            input_path.unlink()
        except OSError:
            logger.warning("Could not delete input of cancelled task %s", task_id)
    update_task(task_id, persist=True, status="cancelled", progress=0,
                message="Separation cancelled by user.", error=None)


def enqueue_job(task_id: str, input_path: Path, mode: str, allow_when_full: bool = False) -> None:
    if not allow_when_full and MAX_QUEUED_JOBS and job_queue.qsize() >= MAX_QUEUED_JOBS:
        raise HTTPException(status_code=429, detail="The separation queue is full. Try again later.")
    job_queue.put((task_id, input_path, mode))


def job_worker() -> None:
    global separator_engine
    separator_engine = DualModeSeparatorEngine()
    while not worker_stop.is_set():
        try:
            job = job_queue.get(timeout=0.5)
        except Empty:
            continue
        try:
            if job is None:
                break
            task_id = job[0]
            if task_is_cancelled(task_id):
                mark_cancelled(task_id)
            else:
                run_separation_job(*job)
        except Exception:
            # Never let one bad job kill the only worker thread.
            logger.exception("Worker failed while handling %s", job)
        finally:
            if job is not None:
                with task_lock:
                    cancelled_tasks.discard(job[0])
            job_queue.task_done()


def start_background_threads() -> None:
    global worker_thread, cleanup_thread
    worker_stop.clear()
    cleanup_old_tasks()
    recover_tasks()
    worker_thread = threading.Thread(target=job_worker, name="separation-worker", daemon=True)
    worker_thread.start()
    cleanup_thread = threading.Thread(target=cleanup_loop, name="task-cleanup", daemon=True)
    cleanup_thread.start()


def stop_background_threads() -> None:
    worker_stop.set()
    job_queue.put(None)
    if worker_thread:
        worker_thread.join(timeout=5)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_background_threads()
    try:
        yield
    finally:
        stop_background_threads()


app = FastAPI(title="Demucs Audio Stem Separator", version="1.1.0", lifespan=lifespan)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(UploadSizeLimitMiddleware, max_body_size=MAX_FILE_SIZE + MULTIPART_OVERHEAD,
                   paths={"/api/separate"})
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", Path(name).stem)
    return cleaned[:40] if cleaned else "track"


def validate_audio_file(audio_path: Path) -> float:
    """Validate the media container and enforce the supported duration range."""
    probe = probe_audio(audio_path)
    if not probe.has_audio:
        raise ValueError("The uploaded file does not contain an audio stream.")
    if probe.duration <= 0:
        raise ValueError("Could not determine the audio duration.")
    if not MIN_DURATION_SECONDS <= probe.duration <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"Audio duration must be between {MIN_DURATION_SECONDS} seconds and "
            f"{MAX_DURATION_SECONDS // 60} minutes."
        )
    return probe.duration


def peaks_from_samples(samples: np.ndarray, num_bars: int = WAVEFORM_BARS) -> List[float]:
    """Reduce mono samples to normalised bar heights for the waveform display."""
    if not len(samples):
        return []
    peaks = []
    for bucket in np.array_split(np.abs(samples), num_bars):
        if not len(bucket):
            peaks.append(0.0)
            continue
        rms = float(np.sqrt(np.mean(bucket ** 2)))
        peaks.append(0.65 * float(np.max(bucket)) + 0.35 * rms)
    max_val = max(peaks) or 1.0
    return [round(float((p / max_val) ** 0.8), 3) for p in peaks]


def compute_stem_waveforms(stem_files: Dict[str, Path], num_bars: int = WAVEFORM_BARS) -> Dict[str, list]:
    """Decode all stems in one ffmpeg run and extract waveform peaks."""
    if not stem_files:
        return {}
    waveforms: Dict[str, list] = {}
    with tempfile.TemporaryDirectory(prefix="waveforms_") as tmp:
        names = list(stem_files)
        temp_paths = [Path(tmp) / f"{i}.f32" for i in range(len(names))]
        cmd = [find_ffmpeg(), "-v", "error", "-y"]
        for name in names:
            cmd += ["-i", str(stem_files[name])]
        for index, temp_path in enumerate(temp_paths):
            cmd += ["-map", f"{index}:a:0", "-ac", "1", "-ar", "1000", "-f", "f32le", str(temp_path)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  text=True, timeout=FFMPEG_TIMEOUT_SECONDS)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[-500:])
            for name, temp_path in zip(names, temp_paths):
                peaks = peaks_from_samples(np.fromfile(temp_path, dtype=np.float32), num_bars)
                if peaks:
                    waveforms[name] = peaks
        except (OSError, subprocess.SubprocessError, RuntimeError, MediaToolError) as exc:
            logger.warning("Failed to extract waveforms: %s", exc)
    return waveforms


def create_previews(stem_files: Dict[str, Path], preview_dir: Path) -> None:
    """Encode compact MP3 previews of the lossless stems for browser playback."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = {name: preview_dir / f".{name}.{uuid.uuid4().hex}.tmp.mp3" for name in stem_files}
    cmd = [find_ffmpeg(), "-v", "error", "-y"]
    for name in stem_files:
        cmd += ["-i", str(stem_files[name])]
    for index, name in enumerate(stem_files):
        cmd += ["-map", f"{index}:a:0", "-c:a", "libmp3lame", "-b:a", PREVIEW_BITRATE,
                str(temp_paths[name])]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, timeout=FFMPEG_TIMEOUT_SECONDS)
        if proc.returncode != 0:
            raise RuntimeError(f"Preview encoding failed: {proc.stderr[-500:]}")
        for name, temp_path in temp_paths.items():
            os.replace(temp_path, preview_dir / f"{name}.mp3")
    finally:
        for temp_path in temp_paths.values():
            temp_path.unlink(missing_ok=True)


def run_separation_job(task_id: str, input_path: Path, mode: str) -> None:
    """Run one separation: models -> previews -> waveforms -> tempo."""
    last_message = {"text": None}

    def update_progress(pct: float, msg: str) -> None:
        # Called for every model chunk, so this is also the cancellation checkpoint.
        if task_is_cancelled(task_id):
            raise JobCancelled()
        update_task(task_id, progress=int(pct), message=msg)
        if msg != last_message["text"]:
            last_message["text"] = msg
            logger.info("Task %s [%d%%]: %s", task_id, int(pct), msg)

    try:
        if task_is_cancelled(task_id):
            raise JobCancelled()
        with task_lock:
            attempts = int((tasks_db.get(task_id) or {}).get("attempts") or 0) + 1
        update_task(task_id, persist=True, status="processing", progress=1, attempts=attempts,
                    message="Initializing AI model engine...", error=None)

        if separator_engine is None:
            raise RuntimeError("Separation worker is not initialized.")
        output_dir = task_dir_for(task_id) / "output"
        # A retried job must not pick up half-written files from the interrupted run.
        shutil.rmtree(output_dir, ignore_errors=True)
        stems_dir = stems_dir_for(task_id)
        stems_dir.mkdir(parents=True, exist_ok=True)

        if mode == "2stems":
            stem_files = separator_engine.separate_2stems(input_path, stems_dir, progress_cb=update_progress)
        else:
            mode = "4stems"
            stem_files = separator_engine.separate_4stems(input_path, stems_dir, progress_cb=update_progress)
        stems_list = [s for s in MODE_STEMS[mode] if s in stem_files]

        update_progress(91, "Encoding preview tracks...")
        create_previews({s: stem_files[s] for s in stems_list}, preview_dir_for(task_id))

        update_progress(95, "Computing audio waveforms...")
        waveforms = compute_stem_waveforms({s: stem_files[s] for s in stems_list})

        update_progress(97, "Estimating tempo...")
        bpm = estimate_bpm(stem_files.get("drums") or input_path)
        update_progress(99, "Finalizing...")

        update_task(task_id, persist=True, status="completed", progress=100,
                    message="Separation completed successfully!", stems=stems_list,
                    waveforms=waveforms, bpm=bpm, mode=mode)
        logger.info("Task %s completed with mode %s. Stems: %s", task_id, mode, stems_list)

    except SeparationCancelled:
        logger.info("Task %s cancelled.", task_id)
        mark_cancelled(task_id)
    except Exception as exc:
        logger.exception("Unexpected error in task %s: %s", task_id, exc)
        update_task(task_id, persist=True, status="error", progress=0,
                    message="Separation failed.", error=str(exc))


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _export_locks_guard:
        lock = _export_locks.get(key)
        if lock is None:
            lock = _export_locks[key] = threading.Lock()
        return lock


def _codec_args(format_type: str, quality: str) -> List[str]:
    if format_type == "wav":
        codec = "pcm_s24le" if quality == "wav24" else "pcm_s16le"
        return ["-c:a", codec, "-ar", "44100"]
    bitrate = quality if quality in ALLOWED_MP3_QUALITIES else "256k"
    return ["-c:a", "libmp3lame", "-b:a", bitrate, "-ar", "44100"]


def export_audio_file(src_path: Path, dest_path: Path, format_type: str, quality: str) -> Path:
    """Convert a stem with ffmpeg and cache it. Only complete files ever reach dest_path."""
    with _lock_for(dest_path):
        if dest_path.is_file() and dest_path.stat().st_size > 0:
            return dest_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the real extension last so ffmpeg picks the right container.
        temp_path = dest_path.with_name(f".{dest_path.stem}.{uuid.uuid4().hex}.tmp{dest_path.suffix}")
        cmd = [find_ffmpeg(), "-v", "error", "-y", "-i", str(src_path),
               *_codec_args(format_type, quality), str(temp_path)]
        try:
            with _export_slots:
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     text=True, timeout=EXPORT_TIMEOUT_SECONDS)
            if res.returncode != 0 or not temp_path.is_file() or temp_path.stat().st_size == 0:
                logger.error("FFmpeg conversion failed for %s: %s", src_path, (res.stderr or "")[-500:])
                raise RuntimeError("Audio conversion failed.")
            os.replace(temp_path, dest_path)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Audio conversion timed out.") from exc
        finally:
            temp_path.unlink(missing_ok=True)
    return dest_path


def export_cached_archive(
    task_id: str,
    stems_list: List[str],
    zip_export_path: Path,
    format_type: str,
    quality: str,
) -> Path:
    with _lock_for(zip_export_path):
        if zip_export_path.is_file() and zip_export_path.stat().st_size > 0:
            return zip_export_path
        exports_dir = zip_export_path.parent
        exports_dir.mkdir(parents=True, exist_ok=True)
        temp_zip = zip_export_path.with_name(f".{zip_export_path.stem}.{uuid.uuid4().hex}.tmp.zip")
        # MP3 doesn't compress further; deflate only helps raw PCM.
        compression = zipfile.ZIP_DEFLATED if format_type == "wav" else zipfile.ZIP_STORED
        try:
            with zipfile.ZipFile(temp_zip, "w", compression) as zipf:
                for stem in stems_list:
                    src = master_stem_path(task_id, stem)
                    if src is None:
                        continue
                    dest = exports_dir / f"{stem}_{quality}.{format_type}"
                    export_audio_file(src, dest, format_type, quality)
                    zipf.write(dest, arcname=f"{stem}.{format_type}")
            os.replace(temp_zip, zip_export_path)
        finally:
            temp_zip.unlink(missing_ok=True)
    return zip_export_path


def _require_completed(task_id: str, detail: str) -> Dict[str, Any]:
    task_info = get_task(task_id)
    if not task_info or task_info.get("status") != "completed":
        raise HTTPException(status_code=404, detail=detail)
    return task_info


def _export_format(format_value: str, quality: str) -> str:
    format_type = "wav" if format_value.lower() == "wav" else "mp3"
    allowed = ALLOWED_WAV_QUALITIES if format_type == "wav" else ALLOWED_MP3_QUALITIES
    if quality not in allowed:
        raise HTTPException(status_code=400, detail="Invalid export quality.")
    return format_type


def _song_title(task_info: Dict[str, Any]) -> str:
    title = task_info.get("original_name") or "Track"
    return Path(title).stem if Path(title).suffix.lower() in ALLOWED_EXTENSIONS else title


def _download_name(requested: Optional[str], default: str, extension: str) -> str:
    clean = re.sub(r"[^\w\s\-\.]", "", requested or default).strip() or default
    if not clean.lower().endswith(f".{extension}"):
        clean += f".{extension}"
    return clean


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=FileResponse)
async def index_page():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.post("/api/separate")
async def start_separation(file: UploadFile = File(...), mode: str = Form("4stems")):
    original_name = file.filename or "audio.mp3"
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail=f"Only the following audio formats are accepted: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    if mode not in MODE_STEMS:
        mode = "4stems"

    task_id = str(uuid.uuid4())
    task_dir = task_dir_for(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    input_path = task_dir / f"input{ext}"

    try:
        total_size = 0
        with open(input_path, "wb") as f_out:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise UploadTooLarge()
                await run_in_threadpool(f_out.write, chunk)
        duration = await run_in_threadpool(validate_audio_file, input_path)
    except ValueError as exc:
        await run_in_threadpool(shutil.rmtree, task_dir, True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MediaToolError as exc:
        await run_in_threadpool(shutil.rmtree, task_dir, True)
        logger.error("Upload rejected because FFmpeg is unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BaseException:
        await run_in_threadpool(shutil.rmtree, task_dir, True)
        raise
    finally:
        await file.close()

    now = time.time()
    task_info = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "File uploaded and separation queued.",
        "error": None,
        "original_name": original_name,
        "base_stem_name": sanitize_filename(original_name),
        "mode": mode,
        "duration": duration,
        "stems": [],
        "waveforms": {},
        "bpm": None,
        "attempts": 0,
        "created_at": now,
        "updated_at": now,
    }
    with task_lock:
        tasks_db[task_id] = task_info
        persist_task(task_id)

    try:
        enqueue_job(task_id, input_path, mode)
    except HTTPException:
        with task_lock:
            tasks_db.pop(task_id, None)
        await run_in_threadpool(shutil.rmtree, task_dir, True)
        raise

    return JSONResponse({
        "success": True,
        "task_id": task_id,
        "status": "pending",
        "message": "File uploaded and separation queued.",
    })


@app.post("/api/cancel/{task_id}")
async def cancel_separation(task_id: str):
    task_info = get_task(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task_info.get("status") in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail="Task is no longer cancellable.")

    with task_lock:
        cancelled_tasks.add(task_id)
        status = (tasks_db.get(task_id) or {}).get("status")
        if status == "processing":
            update_task(task_id, message="Cancelling...")
    if status == "pending":
        # The worker will see the marker and skip it; free the files right away.
        await run_in_threadpool(mark_cancelled, task_id)
    return JSONResponse({"task_id": task_id, "status": "cancelled", "message": "Cancellation requested."})


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    task_info = get_task(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found.")
    return JSONResponse({
        "task_id": task_id,
        "status": task_info.get("status"),
        "progress": task_info.get("progress", 0),
        "message": task_info.get("message", ""),
        "error": task_info.get("error"),
        "stems": task_info.get("stems", []),
        "waveforms": task_info.get("waveforms", {}),
        "original_name": task_info.get("original_name"),
        "duration": task_info.get("duration"),
        "bpm": task_info.get("bpm"),
        "zip_ready": task_info.get("status") == "completed",
    })


@app.get("/api/waveforms/{task_id}")
async def get_waveforms(task_id: str):
    task_info = _require_completed(task_id, "Waveforms not available.")
    waveforms = task_info.get("waveforms")
    if not waveforms:
        sources = {s: master_stem_path(task_id, s) for s in task_info.get("stems", [])}
        sources = {s: p for s, p in sources.items() if p is not None}
        if not sources:
            raise HTTPException(status_code=404, detail="Waveform source files not found.")
        waveforms = await run_in_threadpool(compute_stem_waveforms, sources)
        if waveforms:
            update_task(task_id, persist=True, waveforms=waveforms)
    return JSONResponse(waveforms)


@app.get("/api/stream/{task_id}/{stem_name}")
async def stream_stem(task_id: str, stem_name: str):
    stem_name = stem_name.lower()
    if stem_name not in ALLOWED_STEM_NAMES:
        raise HTTPException(status_code=400, detail="Invalid stem name requested.")
    _require_completed(task_id, "Audio stem not available.")
    stem_path = preview_stem_path(task_id, stem_name)
    if stem_path is None or not is_within_data_dir(stem_path):
        raise HTTPException(status_code=404, detail="Requested stem file not found.")
    return FileResponse(path=stem_path, media_type="audio/mpeg", filename=f"{stem_name}.mp3")


@app.get("/api/export/{task_id}/{stem_name}")
async def export_single_stem(
    task_id: str,
    stem_name: str,
    format: str = Query("mp3"),
    quality: str = Query("256k"),
    filename: Optional[str] = Query(None),
    prepare: bool = Query(False),
):
    stem_name = stem_name.lower()
    if stem_name not in ALLOWED_STEM_NAMES:
        raise HTTPException(status_code=400, detail="Invalid stem name requested.")
    task_info = _require_completed(task_id, "Audio stem not available.")
    src_stem = master_stem_path(task_id, stem_name)
    if src_stem is None:
        raise HTTPException(status_code=404, detail="Requested stem file not found.")

    format_type = _export_format(format, quality)
    cached_dest = task_dir_for(task_id) / "exports" / f"{stem_name}_{quality}.{format_type}"
    try:
        await run_in_threadpool(export_audio_file, src_stem, cached_dest, format_type, quality)
    except (RuntimeError, MediaToolError, OSError) as exc:
        logger.error("Export error for %s / %s: %s", task_id, stem_name, exc)
        raise HTTPException(status_code=500, detail="Failed to process audio export.") from exc

    if prepare:
        return JSONResponse({"ready": True})
    default_name = f"{_song_title(task_info)} - {stem_name.capitalize()}"
    return FileResponse(
        path=cached_dest,
        media_type="audio/wav" if format_type == "wav" else "audio/mpeg",
        filename=_download_name(filename, default_name, format_type),
    )


@app.get("/api/export-all/{task_id}")
async def export_all_stems(
    task_id: str,
    format: str = Query("mp3"),
    quality: str = Query("256k"),
    filename: Optional[str] = Query(None),
    prepare: bool = Query(False),
):
    task_info = _require_completed(task_id, "Stems not ready or task not found.")
    format_type = _export_format(format, quality)
    stems_list = task_info.get("stems") or MODE_STEMS["4stems"]
    zip_export_path = task_dir_for(task_id) / "exports" / f"stems_{format_type}_{quality}.zip"
    try:
        await run_in_threadpool(export_cached_archive, task_id, stems_list, zip_export_path,
                                format_type, quality)
    except (RuntimeError, MediaToolError, OSError) as exc:
        logger.error("Archive export error for %s: %s", task_id, exc)
        raise HTTPException(status_code=500, detail="Failed to create the stems archive.") from exc

    if prepare:
        return JSONResponse({"ready": True})
    return FileResponse(
        path=zip_export_path,
        media_type="application/zip",
        filename=_download_name(filename, f"{_song_title(task_info)}_stems", "zip"),
    )


@app.get("/api/download/{task_id}")
async def download_zip(task_id: str):
    """Kept for compatibility: all stems as 320 kbps MP3 in a ZIP."""
    return await export_all_stems(task_id, format="mp3", quality="320k", filename=None, prepare=False)
