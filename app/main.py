import os
import sys
import uuid
import re
import shutil
import zipfile
import json
import logging
import subprocess
import threading
import time
from queue import Empty, Full, Queue
from pathlib import Path
from typing import Dict, Any, Optional
from app.separator_engine import DualModeSeparatorEngine

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stem_separator")

# Papkalar
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "tasks"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TASK_RETENTION_SECONDS = int(os.environ.get("TASK_RETENTION_HOURS", "24")) * 60 * 60
MAX_QUEUED_JOBS = max(0, int(os.environ.get("MAX_QUEUED_JOBS", "2")))

# Demucs yo'lini aniqlash
VENV_SCRIPTS = Path(sys.executable).parent
DEMUCS_EXE = VENV_SCRIPTS / "demucs.exe"
if not DEMUCS_EXE.exists():
    DEMUCS_EXE = "demucs"

# Xavfsizlik qoidalari
MAX_FILE_SIZE = 75 * 1024 * 1024  # 75 MB
MIN_DURATION_SECONDS = 10
MAX_DURATION_SECONDS = 10 * 60
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
ALLOWED_MP3_QUALITIES = {"320k", "256k", "192k"}
ALLOWED_WAV_QUALITIES = {"wav24", "wav16"}
ALLOWED_STEM_NAMES = {"vocals", "drums", "bass", "other", "no_vocals", "instrumental"}
UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

app = FastAPI(title="Demucs Audio Stem Separator", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Xotiradagi vazifalar holati
tasks_db: Dict[str, Dict[str, Any]] = {}
task_lock = threading.RLock()
job_queue: Queue = Queue()
worker_stop = threading.Event()
worker_thread: Optional[threading.Thread] = None
separator_engine: Optional[DualModeSeparatorEngine] = None
export_lock = threading.Lock()
cancelled_tasks = set()


class JobCancelled(Exception):
    """Raised when a user cancels a queued or running separation."""


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


app.add_middleware(SecurityHeadersMiddleware)


def task_meta_path(task_id: str) -> Path:
    return DATA_DIR / task_id / "meta.json"


def persist_task(task_id: str) -> None:
    """Persist task state so Colab/server restarts can recover it."""
    with task_lock:
        task_info = tasks_db.get(task_id)
        if not task_info:
            return
        meta = dict(task_info)
    try:
        with open(task_meta_path(task_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except OSError:
        logger.exception("Could not persist task %s", task_id)


def update_task(task_id: str, persist: bool = False, **changes: Any) -> None:
    with task_lock:
        if task_id not in tasks_db:
            return
        tasks_db[task_id].update(changes)
    if persist:
        persist_task(task_id)


def task_is_cancelled(task_id: str) -> bool:
    with task_lock:
        return task_id in cancelled_tasks


def mark_cancelled(task_id: str, clear_marker: bool = True) -> None:
    if clear_marker:
        with task_lock:
            cancelled_tasks.discard(task_id)
    task_dir = DATA_DIR / task_id
    shutil.rmtree(task_dir / "output", ignore_errors=True)
    update_task(
        task_id,
        persist=True,
        status="cancelled",
        progress=0,
        message="Separation cancelled by user.",
        error=None
    )


def cleanup_old_tasks() -> None:
    cutoff = time.time() - TASK_RETENTION_SECONDS
    for task_dir in DATA_DIR.iterdir():
        if not task_dir.is_dir() or not UUID_REGEX.match(task_dir.name):
            continue
        try:
            meta_file = task_dir / "meta.json"
            if not meta_file.exists():
                continue
            meta = {}
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            if meta.get("status") not in {"completed", "error"}:
                continue
            if task_dir.stat().st_mtime < cutoff:
                shutil.rmtree(task_dir)
                logger.info("Removed expired task %s", task_dir.name)
        except (OSError, ValueError, json.JSONDecodeError):
            logger.exception("Could not clean task directory %s", task_dir)


def recover_tasks() -> None:
    """Requeue interrupted jobs after a process restart."""
    for task_dir in DATA_DIR.iterdir():
        if not task_dir.is_dir() or not UUID_REGEX.match(task_dir.name):
            continue
        try:
            meta_file = task_dir / "meta.json"
            if not meta_file.exists():
                logger.warning("Skipping task %s without metadata.", task_dir.name)
                continue
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("status") not in {"pending", "processing"}:
                continue
            input_files = list(task_dir.glob("input.*"))
            if not input_files:
                continue
            task_id = task_dir.name
            tasks_db[task_id] = meta
            tasks_db[task_id]["status"] = "pending"
            tasks_db[task_id]["message"] = "Recovered after server restart."
            output_dir = task_dir / "output"
            try:
                enqueue_job(
                    task_id,
                    input_files[0],
                    output_dir,
                    meta.get("mode", "4stems"),
                    meta.get("base_stem_name", "track"),
                    allow_when_full=True
                )
                persist_task(task_id)
            except HTTPException:
                logger.warning("Recovery queue is full; leaving task %s pending.", task_id)
        except (OSError, ValueError, json.JSONDecodeError):
            logger.exception("Could not recover task %s", task_dir.name)


def job_worker() -> None:
    global separator_engine
    separator_engine = DualModeSeparatorEngine()
    while not worker_stop.is_set():
        try:
            job = job_queue.get(timeout=0.5)
        except Empty:
            continue
        if job is None:
            job_queue.task_done()
            break
        try:
            if task_is_cancelled(job[0]):
                mark_cancelled(job[0])
            else:
                run_separation_job(*job)
        finally:
            job_queue.task_done()


def enqueue_job(
    task_id: str,
    input_path: Path,
    output_dir: Path,
    mode: str,
    base_stem_name: str,
    allow_when_full: bool = False
) -> None:
    if not allow_when_full and MAX_QUEUED_JOBS and job_queue.qsize() >= MAX_QUEUED_JOBS:
        raise HTTPException(status_code=429, detail="The separation queue is full. Try again later.")
    try:
        job_queue.put_nowait((task_id, input_path, output_dir, mode, base_stem_name))
    except Full:
        raise HTTPException(status_code=429, detail="The separation queue is full. Try again later.")


@app.on_event("startup")
def start_worker() -> None:
    global worker_thread
    cleanup_old_tasks()
    worker_stop.clear()
    worker_thread = threading.Thread(target=job_worker, name="separation-worker", daemon=True)
    worker_thread.start()
    recover_tasks()


@app.on_event("shutdown")
def stop_worker() -> None:
    worker_stop.set()
    try:
        job_queue.put_nowait(None)
    except Full:
        pass
    if worker_thread:
        worker_thread.join(timeout=5)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', Path(name).stem)
    return cleaned[:40] if cleaned else "track"


import numpy as np


def validate_audio_file(audio_path: Path) -> float:
    """Validate the media container and enforce the supported duration range."""
    ffprobe_exe = Path(sys.executable).parent / "ffprobe.exe"
    ffprobe_cmd = str(ffprobe_exe) if ffprobe_exe.exists() else "ffprobe"
    cmd = [
        ffprobe_cmd,
        "-v", "error",
        "-show_entries", "format=duration:stream=codec_type",
        "-of", "json",
        str(audio_path)
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20
        )
        if result.returncode != 0:
            raise ValueError("The uploaded file is not a valid audio file.")
        media = json.loads(result.stdout)
        if not any(s.get("codec_type") == "audio" for s in media.get("streams", [])):
            raise ValueError("The uploaded file does not contain an audio stream.")
        duration = float(media.get("format", {}).get("duration", 0))
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("The uploaded file could not be validated as audio.") from exc

    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"Audio duration must be between {MIN_DURATION_SECONDS} seconds and "
            f"{MAX_DURATION_SECONDS // 60} minutes."
        )
    return duration


def compute_stem_waveforms(stems_dir: Path, output_dir: Path, num_bars: int = 120) -> Dict[str, list]:
    """Extract real audio waveform peak amplitudes for all generated stems"""
    waveforms_file = output_dir / "waveforms.json"
    if waveforms_file.exists():
        try:
            with open(waveforms_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    waveforms = {}
    ffmpeg_exe = Path(sys.executable).parent / "ffmpeg.exe"
    ffmpeg_cmd = str(ffmpeg_exe) if ffmpeg_exe.exists() else "ffmpeg"

    stem_paths = sorted(stems_dir.glob("*.mp3"))
    if not stem_paths:
        return waveforms

    temp_paths = [
        output_dir / f".waveform_{index}.f32"
        for index in range(len(stem_paths))
    ]
    cmd = [ffmpeg_cmd, "-v", "error", "-y"]
    for stem_path in stem_paths:
        cmd.extend(["-i", str(stem_path)])
    for index, temp_path in enumerate(temp_paths):
        cmd.extend([
            "-map", f"{index}:a:0",
            "-ac", "1",
            "-ar", "1000",
            "-f", "f32le",
            str(temp_path)
        ])

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=MAX_DURATION_SECONDS
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-500:])

        for stem_path, temp_path in zip(stem_paths, temp_paths):
            samples = np.fromfile(temp_path, dtype=np.float32)
            if not len(samples):
                continue
            bins = np.array_split(np.abs(samples), num_bars)
            peaks = []
            for bucket in bins:
                rms = float(np.sqrt(np.mean(bucket ** 2))) if len(bucket) else 0.0
                peak = float(np.max(bucket)) if len(bucket) else 0.0
                peaks.append(0.65 * peak + 0.35 * rms)
            max_val = max(peaks) if max(peaks) > 0 else 1.0
            waveforms[stem_path.stem.lower()] = [
                round(float((peak / max_val) ** 0.8), 3)
                for peak in peaks
            ]
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        logger.warning("Failed to extract waveforms: %s", exc)
    finally:
        for temp_path in temp_paths:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove waveform temporary file %s", temp_path)

    if waveforms:
        try:
            with open(waveforms_file, "w") as f:
                json.dump(waveforms, f)
        except Exception:
            pass

    return waveforms


def estimate_bpm(audio_path: Path) -> int:
    """Estimate tempo (BPM) from audio file using energy autocorrelation."""
    if not audio_path.exists():
        return 87
    try:
        ffmpeg_exe = Path(sys.executable).parent / "ffmpeg.exe"
        ffmpeg_cmd = str(ffmpeg_exe) if ffmpeg_exe.exists() else "ffmpeg"

        cmd = [
            ffmpeg_cmd,
            "-ss", "10",
            "-t", "90",
            "-i", str(audio_path),
            "-ac", "1",
            "-ar", "200",
            "-f", "f32le",
            "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if proc.returncode != 0 or not proc.stdout:
            cmd = [
                ffmpeg_cmd,
                "-t", "90",
                "-i", str(audio_path),
                "-ac", "1",
                "-ar", "200",
                "-f", "f32le",
                "-"
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        if proc.returncode == 0 and proc.stdout:
            samples = np.frombuffer(proc.stdout, dtype=np.float32)
            if len(samples) > 200 * 5:
                energy = samples ** 2
                diff = np.diff(energy)
                diff[diff < 0] = 0
                diff -= np.mean(diff)

                autocorr = np.correlate(diff, diff, mode="full")
                autocorr = autocorr[len(autocorr) // 2:]

                sr = 200
                min_lag = int(sr * 60 / 200)  # 200 BPM
                max_lag = int(sr * 60 / 60)   # 60 BPM

                peak_lag = min_lag + np.argmax(autocorr[min_lag:max_lag])
                bpm = int(round(60.0 * sr / peak_lag))
                if 50 <= bpm <= 220:
                    return bpm
    except Exception as e:
        logger.warning(f"BPM estimation error: {e}")

    return 87


def get_or_restore_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task info from memory or restore from persistent storage"""
    if not UUID_REGEX.match(task_id):
        return None
    task_info = tasks_db.get(task_id)
    if task_info and "original_name" in task_info and "bpm" in task_info:
        return task_info

    task_dir = DATA_DIR / task_id
    if not task_dir.exists():
        return None

    # Load meta.json if present
    meta_file = task_dir / "meta.json"
    meta_data = {}
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        except Exception:
            pass

    output_dir = task_dir / "output"
    stem_files = []
    actual_stem_dir = None

    # Check dedicated stems directory first
    if (output_dir / "stems").exists():
        found = list((output_dir / "stems").glob("*.mp3"))
        if found:
            stem_files = found
            actual_stem_dir = output_dir / "stems"

    # Fallback to legacy htdemucs directory
    if not stem_files:
        stem_dirs = list((output_dir / "htdemucs").glob("*")) if (output_dir / "htdemucs").exists() else []
        if stem_dirs:
            for sd in stem_dirs:
                if sd.is_dir():
                    found = list(sd.glob("*.mp3"))
                    if found:
                        stem_files = found
                        actual_stem_dir = sd
                        break

    zips = list(output_dir.glob("*.zip"))

    if meta_data.get("status") == "completed" or stem_files:
        stems_dir = Path(meta_data.get("stems_dir", "")) if meta_data.get("stems_dir") else actual_stem_dir
        if not stems_dir or not stems_dir.exists():
            stems_dir = actual_stem_dir
        stems_list = meta_data.get("stems") or [sf.stem.lower() for sf in stem_files]
        zip_path_value = meta_data.get("zip_path")
        zip_path = Path(zip_path_value) if zip_path_value else (zips[0] if zips else None)
        tasks_db[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Separation completed successfully!",
            "stems": stems_list,
            "waveforms": meta_data.get("waveforms", {}),
            "zip_filename": meta_data.get("zip_filename", zip_path.name if zip_path else ""),
            "stems_dir": str(stems_dir) if stems_dir else "",
            "zip_path": str(zip_path) if zip_path else "",
            "original_name": meta_data.get("original_name", "Music Track.mp3"),
            "bpm": meta_data.get("bpm")
        }
        persist_task(task_id)
        return tasks_db[task_id]

    input_files = list(task_dir.glob("input.*"))
    if input_files:
        tasks_db[task_id] = {
            "status": meta_data.get("status", "pending"),
            "progress": 50,
            "message": "Analyzing audio...",
            "stems": [],
            "original_name": meta_data.get("original_name", "Music Track.mp3"),
            "bpm": meta_data.get("bpm", 87)
        }
        return tasks_db[task_id]

    return None


def run_separation_job(task_id: str, input_path: Path, output_dir: Path, mode: str, base_stem_name: str):
    """
    Dual-Mode AI Stem Separation Background Task:
    - Mode '2stems': BS-RoFormer AI separates into clean Vocals & Instrumental.
    - Mode '4stems': Two-stage AI pipeline:
        Stage 1: BS-RoFormer isolates clean Vocals & Instrumental.
        Stage 2: Demucs operates ONLY on the isolated Instrumental -> Drums, Bass, Other.
        Zero vocal leakage into instruments.
    """
    try:
        if task_is_cancelled(task_id):
            raise JobCancelled()

        update_task(
            task_id,
            persist=True,
            status="processing",
            progress=5,
            message="Initializing AI model engine..."
        )

        def update_progress(pct: int, msg: str):
            if task_is_cancelled(task_id):
                raise JobCancelled()
            update_task(task_id, progress=pct, message=msg)
            logger.info(f"Task {task_id} [{pct}%]: {msg}")

        if separator_engine is None:
            raise RuntimeError("Separation worker is not initialized.")
        engine = separator_engine
        stems_dir = output_dir / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)

        if mode == "2stems":
            stems_dict = engine.separate_2stems(input_path, stems_dir, progress_cb=update_progress)
            stems_list = ["vocals", "instrumental"]
        else:
            stems_dict = engine.separate_4stems(input_path, stems_dir, progress_cb=update_progress)
            stems_list = ["vocals", "drums", "bass", "other"]

        update_progress(94, "Computing audio waveforms & peak data...")
        waveforms = compute_stem_waveforms(stems_dir, output_dir)
        if task_is_cancelled(task_id):
            raise JobCancelled()

        # Estimate BPM from drums or input
        drums_file = stems_dir / "drums.mp3"
        bpm_source = drums_file if drums_file.exists() else input_path
        bpm = estimate_bpm(bpm_source)
        if task_is_cancelled(task_id):
            raise JobCancelled()

        update_progress(97, "Packaging output stems archive (ZIP)...")
        zip_filename = f"{base_stem_name}_stems.zip"
        zip_path = output_dir / zip_filename
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for s_name, s_path in stems_dict.items():
                if s_path.exists():
                    zipf.write(s_path, arcname=s_path.name)

        # Persist to meta.json
        meta_file = output_dir.parent / "meta.json"
        meta_data = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                pass
        meta_data["bpm"] = bpm
        meta_data["mode"] = mode
        meta_data["stems"] = stems_list
        meta_data["original_name"] = tasks_db[task_id].get("original_name", base_stem_name)
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        update_task(
            task_id,
            persist=True,
            status="completed",
            progress=100,
            message="Separation completed successfully!",
            stems=stems_list,
            waveforms=waveforms,
            bpm=bpm,
            zip_filename=zip_filename,
            stems_dir=str(stems_dir),
            zip_path=str(zip_path)
        )
        logger.info(f"Task {task_id} completed successfully with mode {mode}. Stems: {stems_list}")

    except JobCancelled:
        logger.info("Task %s cancelled.", task_id)
        mark_cancelled(task_id)
    except Exception as exc:
        logger.exception(f"Unexpected error in Task {task_id}: {exc}")
        update_task(
            task_id,
            persist=True,
            status="error",
            progress=0,
            message="Separation failed.",
            error=str(exc)
        )


@app.post("/api/cancel/{task_id}")
async def cancel_separation(task_id: str):
    task_info = get_or_restore_task(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task_info.get("status") in {"completed", "error", "cancelled"}:
        raise HTTPException(status_code=409, detail="Task is no longer cancellable.")

    with task_lock:
        cancelled_tasks.add(task_id)
    if task_info.get("status") == "pending":
        mark_cancelled(task_id, clear_marker=False)
    return JSONResponse({
        "task_id": task_id,
        "status": "cancelled",
        "message": "Cancellation requested."
    })


@app.get("/", response_class=FileResponse)
async def index_page():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.post("/api/separate")
async def start_separation(
    file: UploadFile = File(...),
    mode: str = Form("4stems")
):
    original_name = file.filename or "audio.mp3"
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only the following audio formats are accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    if mode not in {"4stems", "2stems"}:
        mode = "4stems"

    task_id = str(uuid.uuid4())
    task_dir = DATA_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    input_path = task_dir / f"input{ext}"
    output_dir = task_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stream file to disk with strict size limit
    total_size = 0
    with open(input_path, "wb") as f_out:
        while chunk := await file.read(1024 * 64):
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                f_out.close()
                shutil.rmtree(task_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail="Audio file exceeds the maximum allowed size of 75 MB."
                )
            f_out.write(chunk)
    await file.close()

    try:
        duration = validate_audio_file(input_path)
    except ValueError as exc:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    safe_base_name = sanitize_filename(original_name)

    # Save initial meta.json
    try:
        with open(task_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "task_id": task_id,
                "original_name": original_name,
                "base_stem_name": safe_base_name,
                "mode": mode,
                "duration": duration
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    tasks_db[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "File uploaded and separation queued.",
        "original_name": original_name,
        "base_stem_name": safe_base_name,
        "mode": mode,
        "duration": duration,
        "stems": [],
        "bpm": None
    }
    persist_task(task_id)

    try:
        enqueue_job(task_id, input_path, output_dir, mode, safe_base_name)
    except HTTPException:
        shutil.rmtree(task_dir, ignore_errors=True)
        tasks_db.pop(task_id, None)
        raise

    return JSONResponse({
        "success": True,
        "task_id": task_id,
        "status": "pending",
        "message": "File uploaded and separation queued."
    })


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    task_info = get_or_restore_task(task_id)
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
        "bpm": task_info.get("bpm"),
        "zip_ready": task_info.get("status") == "completed"
    })


@app.get("/api/waveforms/{task_id}")
async def get_waveforms(task_id: str):
    task_info = get_or_restore_task(task_id)
    if not task_info or task_info.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Waveforms not available.")

    waveforms = task_info.get("waveforms")
    if not waveforms:
        zip_path = task_info.get("zip_path")
        stems_dir_value = task_info.get("stems_dir")
        if not zip_path or not stems_dir_value:
            raise HTTPException(status_code=404, detail="Waveform source files not found.")
        output_dir = Path(zip_path).parent
        stems_dir = Path(stems_dir_value)
        waveforms = await run_in_threadpool(compute_stem_waveforms, stems_dir, output_dir)
        task_info["waveforms"] = waveforms
        persist_task(task_id)

    return JSONResponse(waveforms)


@app.get("/api/download/{task_id}")
async def download_zip(task_id: str):
    task_info = get_or_restore_task(task_id)
    if not task_info or task_info.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Stems not ready or task not found.")

    zip_path_str = task_info.get("zip_path")
    if not zip_path_str or not Path(zip_path_str).exists():
        raise HTTPException(status_code=404, detail="ZIP archive not found.")

    zip_path = Path(zip_path_str)
    if not str(zip_path.resolve()).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Unauthorized file path access.")

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=task_info.get("zip_filename", "stems.zip")
    )


@app.get("/api/stream/{task_id}/{stem_name}")
async def stream_stem(task_id: str, stem_name: str):
    stem_name = stem_name.lower()
    if stem_name not in ALLOWED_STEM_NAMES:
        raise HTTPException(status_code=400, detail="Invalid stem name requested.")

    task_info = get_or_restore_task(task_id)
    if not task_info or task_info.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Audio stem not available.")

    stems_dir_str = task_info.get("stems_dir")
    if not stems_dir_str:
        raise HTTPException(status_code=404, detail="Stem directory not found.")

    stem_path = Path(stems_dir_str) / f"{stem_name}.mp3"
    if not stem_path.exists():
        raise HTTPException(status_code=404, detail="Requested stem file not found.")

    if not str(stem_path.resolve()).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Unauthorized file path access.")

    return FileResponse(
        path=stem_path,
        media_type="audio/mpeg",
        filename=f"{stem_name}.mp3"
    )


def export_audio_file(src_path: Path, dest_path: Path, format_type: str, quality: str):
    """
    Converts audio using ffmpeg to requested format (MP3/WAV) and quality.
    Caches the result in dest_path.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    venv_scripts = Path(sys.executable).parent
    ffmpeg_bin = venv_scripts / "ffmpeg.exe"
    if not ffmpeg_bin.exists():
        ffmpeg_bin = Path("ffmpeg")

    cmd = [str(ffmpeg_bin), "-y", "-i", str(src_path)]
    if format_type == "wav":
        if quality == "wav24":
            cmd.extend(["-c:a", "pcm_s24le", "-ar", "48000"])
        else:  # wav16 default
            cmd.extend(["-c:a", "pcm_s16le", "-ar", "44100"])
    else:  # mp3
        bitrate = "320k" if quality == "320k" else ("192k" if quality == "192k" else "256k")
        cmd.extend(["-c:a", "libmp3lame", "-b:a", bitrate, "-ar", "44100"])

    cmd.append(str(dest_path))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        logger.error(f"FFmpeg conversion failed: {res.stderr}")
        raise RuntimeError("Audio conversion failed.")
    return dest_path


def export_cached_stem(
    src_stem: Path,
    cached_dest: Path,
    format_type: str,
    quality: str
) -> Path:
    with export_lock:
        return export_audio_file(src_stem, cached_dest, format_type, quality)


def export_cached_archive(
    stems_dir: Path,
    stems_list: list,
    exports_dir: Path,
    zip_export_path: Path,
    format_type: str,
    quality: str
) -> Path:
    with export_lock:
        if not zip_export_path.exists() or zip_export_path.stat().st_size == 0:
            temp_zip_path = zip_export_path.with_suffix(".tmp.zip")
            try:
                with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for s_name in stems_list:
                        src_stem = stems_dir / f"{s_name}.mp3"
                        if src_stem.exists():
                            stem_dest = exports_dir / f"{s_name}_{quality}.{format_type}"
                            export_audio_file(src_stem, stem_dest, format_type, quality)
                            zipf.write(stem_dest, arcname=f"{s_name}.{format_type}")
                temp_zip_path.replace(zip_export_path)
            finally:
                if temp_zip_path.exists():
                    temp_zip_path.unlink()
    return zip_export_path


@app.get("/api/export/{task_id}/{stem_name}")
async def export_single_stem(
    task_id: str,
    stem_name: str,
    format: str = Query("mp3"),
    quality: str = Query("256k"),
    filename: Optional[str] = Query(None)
):
    stem_name = stem_name.lower()
    if stem_name not in ALLOWED_STEM_NAMES:
        raise HTTPException(status_code=400, detail="Invalid stem name requested.")

    task_info = get_or_restore_task(task_id)
    if not task_info or task_info.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Audio stem not available.")

    stems_dir_str = task_info.get("stems_dir")
    if not stems_dir_str:
        raise HTTPException(status_code=404, detail="Stem directory not found.")

    src_stem = Path(stems_dir_str) / f"{stem_name}.mp3"
    if not src_stem.exists():
        raise HTTPException(status_code=404, detail="Requested stem file not found.")

    format_type = "wav" if format.lower() == "wav" else "mp3"
    allowed_qualities = ALLOWED_WAV_QUALITIES if format_type == "wav" else ALLOWED_MP3_QUALITIES
    if quality not in allowed_qualities:
        raise HTTPException(status_code=400, detail="Invalid export quality.")
    task_dir = DATA_DIR / task_id
    exports_dir = task_dir / "exports"
    cached_dest = exports_dir / f"{stem_name}_{quality}.{format_type}"

    try:
        await run_in_threadpool(
            export_cached_stem,
            src_stem,
            cached_dest,
            format_type,
            quality
        )
    except Exception as e:
        logger.error(f"Export error for {task_id} / {stem_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process audio export.")

    if not filename:
        song_title = task_info.get("original_name", "Track")
        if song_title.lower().endswith(".mp3") or song_title.lower().endswith(".wav"):
            song_title = Path(song_title).stem
        filename = f"{song_title} - {stem_name.capitalize()}"

    clean_name = re.sub(r'[^\w\s\-\.]', '', filename).strip() or stem_name
    if not clean_name.lower().endswith(f".{format_type}"):
        clean_name += f".{format_type}"

    return FileResponse(
        path=cached_dest,
        media_type="audio/wav" if format_type == "wav" else "audio/mpeg",
        filename=clean_name
    )


@app.get("/api/export-all/{task_id}")
async def export_all_stems(
    task_id: str,
    format: str = Query("mp3"),
    quality: str = Query("256k"),
    filename: Optional[str] = Query(None)
):
    task_info = get_or_restore_task(task_id)
    if not task_info or task_info.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Stems not ready or task not found.")

    stems_dir_str = task_info.get("stems_dir")
    if not stems_dir_str:
        raise HTTPException(status_code=404, detail="Stem directory not found.")

    stems_list = task_info.get("stems", ["vocals", "drums", "bass", "other"])
    format_type = "wav" if format.lower() == "wav" else "mp3"
    allowed_qualities = ALLOWED_WAV_QUALITIES if format_type == "wav" else ALLOWED_MP3_QUALITIES
    if quality not in allowed_qualities:
        raise HTTPException(status_code=400, detail="Invalid export quality.")

    task_dir = DATA_DIR / task_id
    exports_dir = task_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    zip_export_path = exports_dir / f"stems_{format_type}_{quality}.zip"

    await run_in_threadpool(
        export_cached_archive,
        Path(stems_dir_str),
        stems_list,
        exports_dir,
        zip_export_path,
        format_type,
        quality
    )

    song_title = task_info.get("original_name", "Track")
    if song_title.lower().endswith(".mp3") or song_title.lower().endswith(".wav"):
        song_title = Path(song_title).stem

    if not filename:
        filename = f"{song_title}_stems.zip"

    clean_name = re.sub(r'[^\w\s\-\.]', '', filename).strip() or "stems.zip"
    if not clean_name.lower().endswith(".zip"):
        clean_name += ".zip"

    return FileResponse(
        path=zip_export_path,
        media_type="application/zip",
        filename=clean_name
    )
