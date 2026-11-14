import os
import sys
import uuid
import re
import shutil
import zipfile
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from app.separator_engine import DualModeSeparatorEngine

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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

# Demucs yo'lini aniqlash
VENV_SCRIPTS = Path(sys.executable).parent
DEMUCS_EXE = VENV_SCRIPTS / "demucs.exe"
if not DEMUCS_EXE.exists():
    DEMUCS_EXE = "demucs"

# Xavfsizlik qoidalari
MAX_FILE_SIZE = 75 * 1024 * 1024  # 75 MB
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
ALLOWED_STEM_NAMES = {"vocals", "drums", "bass", "other", "no_vocals", "instrumental"}
UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

app = FastAPI(title="Demucs Audio Stem Separator", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Xotiradagi vazifalar holati
tasks_db: Dict[str, Dict[str, Any]] = {}


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


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', Path(name).stem)
    return cleaned[:40] if cleaned else "track"


import json
import numpy as np


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

    for stem_path in stems_dir.glob("*.mp3"):
        stem_name = stem_path.stem.lower()
        try:
            cmd = [
                ffmpeg_cmd,
                "-i", str(stem_path),
                "-ac", "1",
                "-ar", "1000",
                "-f", "f32le",
                "-"
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if proc.returncode == 0 and proc.stdout:
                samples = np.frombuffer(proc.stdout, dtype=np.float32)
                bins = np.array_split(np.abs(samples), num_bars)
                peaks = []
                for b in bins:
                    if len(b) > 0:
                        rms = float(np.sqrt(np.mean(b**2)))
                        pk = float(np.max(b))
                        peaks.append(0.65 * pk + 0.35 * rms)
                    else:
                        peaks.append(0.0)
                max_val = max(peaks) if max(peaks) > 0 else 1.0
                norm = [round(float((p / max_val) ** 0.8), 3) for p in peaks]
                waveforms[stem_name] = norm
        except Exception as e:
            logger.warning(f"Failed to extract waveform for {stem_name}: {e}")

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

    if stem_files:
        stems_list = [sf.stem.lower() for sf in stem_files]
        zip_path = zips[0] if zips else None
        if not zip_path:
            zip_path = output_dir / "audio_stems.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for sf in stem_files:
                    zipf.write(sf, arcname=sf.name)

        waveforms = compute_stem_waveforms(actual_stem_dir, output_dir)

        bpm = meta_data.get("bpm")
        if not bpm:
            drums_file = actual_stem_dir / "drums.mp3"
            target_audio = drums_file if drums_file.exists() else (task_dir / "input.mp3")
            bpm = estimate_bpm(target_audio)
            meta_data["bpm"] = bpm

        original_name = meta_data.get("original_name")
        if not original_name:
            input_files = list(task_dir.glob("input.*"))
            original_name = input_files[0].name if input_files else "Music Track.mp3"
            meta_data["original_name"] = original_name

        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        tasks_db[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Separation completed successfully!",
            "stems": stems_list,
            "waveforms": waveforms,
            "zip_filename": zip_path.name,
            "stems_dir": str(actual_stem_dir),
            "zip_path": str(zip_path),
            "original_name": meta_data.get("original_name", "Music Track.mp3"),
            "bpm": meta_data.get("bpm", 87)
        }
        return tasks_db[task_id]

    input_files = list(task_dir.glob("input.*"))
    if input_files:
        tasks_db[task_id] = {
            "status": "processing",
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
        tasks_db[task_id]["status"] = "processing"
        tasks_db[task_id]["progress"] = 5
        tasks_db[task_id]["message"] = "Initializing AI model engine..."

        def update_progress(pct: int, msg: str):
            tasks_db[task_id]["progress"] = pct
            tasks_db[task_id]["message"] = msg
            logger.info(f"Task {task_id} [{pct}%]: {msg}")

        engine = DualModeSeparatorEngine()
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

        # Estimate BPM from drums or input
        drums_file = stems_dir / "drums.mp3"
        bpm_source = drums_file if drums_file.exists() else input_path
        bpm = estimate_bpm(bpm_source)

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

        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["progress"] = 100
        tasks_db[task_id]["message"] = "Separation completed successfully!"
        tasks_db[task_id]["stems"] = stems_list
        tasks_db[task_id]["waveforms"] = waveforms
        tasks_db[task_id]["bpm"] = bpm
        tasks_db[task_id]["zip_filename"] = zip_filename
        tasks_db[task_id]["stems_dir"] = str(stems_dir)
        tasks_db[task_id]["zip_path"] = str(zip_path)
        logger.info(f"Task {task_id} completed successfully with mode {mode}. Stems: {stems_list}")

    except Exception as exc:
        logger.exception(f"Unexpected error in Task {task_id}: {exc}")
        tasks_db[task_id]["status"] = "error"
        tasks_db[task_id]["error"] = str(exc)


@app.get("/", response_class=FileResponse)
async def index_page():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.post("/api/separate")
async def start_separation(
    background_tasks: BackgroundTasks,
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

    safe_base_name = sanitize_filename(original_name)

    # Save initial meta.json
    try:
        with open(task_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "task_id": task_id,
                "original_name": original_name,
                "base_stem_name": safe_base_name,
                "mode": mode
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    tasks_db[task_id] = {
        "status": "pending",
        "original_name": original_name,
        "base_stem_name": safe_base_name,
        "mode": mode,
        "stems": [],
        "bpm": None
    }

    background_tasks.add_task(run_separation_job, task_id, input_path, output_dir, mode, safe_base_name)

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
        output_dir = Path(task_info.get("zip_path")).parent
        stems_dir = Path(task_info.get("stems_dir"))
        waveforms = compute_stem_waveforms(stems_dir, output_dir)
        task_info["waveforms"] = waveforms

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
    task_dir = DATA_DIR / task_id
    exports_dir = task_dir / "exports"
    cached_dest = exports_dir / f"{stem_name}_{quality}.{format_type}"

    try:
        export_audio_file(src_stem, cached_dest, format_type, quality)
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

    task_dir = DATA_DIR / task_id
    exports_dir = task_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    zip_export_path = exports_dir / f"stems_{format_type}_{quality}.zip"

    if not zip_export_path.exists() or zip_export_path.stat().st_size == 0:
        with zipfile.ZipFile(zip_export_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for s_name in stems_list:
                src_stem = Path(stems_dir_str) / f"{s_name}.mp3"
                if src_stem.exists():
                    stem_dest = exports_dir / f"{s_name}_{quality}.{format_type}"
                    export_audio_file(src_stem, stem_dest, format_type, quality)
                    zipf.write(stem_dest, arcname=f"{s_name}.{format_type}")

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
