import sys
from pathlib import Path

import modal


PROJECT_ROOT = "/root/stem-separator"
APP_DIR = Path(__file__).resolve().parent / "app"

app = modal.App("stem-separator")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "libsndfile1", "clang", "build-essential")
    .pip_install(
        "audio-separator[gpu]>=0.48.0",
        "demucs>=4.1.0",
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "python-multipart>=0.0.9",
        "jinja2>=3.1.4",
        "soundfile>=0.12.1",
        "imageio-ffmpeg>=0.5.1",
        "numpy",
    )
    .env(
        {
            "XDG_CACHE_HOME": f"{PROJECT_ROOT}/data/cache",
            "TORCH_HOME": f"{PROJECT_ROOT}/data/cache/torch",
            "MAX_QUEUED_JOBS": "2",
            "TASK_RETENTION_HOURS": "24",
        }
    )
    .add_local_dir(APP_DIR, remote_path=f"{PROJECT_ROOT}/app")
)

storage = modal.Volume.from_name("stem-separator-storage", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    scaledown_window=60,
    min_containers=0,
    max_containers=1,
    volumes={f"{PROJECT_ROOT}/data": storage},
)
@modal.asgi_app()
def web():
    sys.path.insert(0, PROJECT_ROOT)
    from app.main import app as fastapi_app

    return fastapi_app
