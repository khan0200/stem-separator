import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Point the app at a throwaway data directory *before* app.main is imported.
_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="stem_tests_"))
os.environ["STEM_DATA_DIR"] = str(_TEST_DATA_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state():
    """Each test starts with an empty data directory, task cache and queue."""
    def reset():
        with main.task_lock:
            main.tasks_db.clear()
            main.cancelled_tasks.clear()
        while not main.job_queue.empty():
            main.job_queue.get_nowait()
            main.job_queue.task_done()
        for child in main.DATA_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()

    reset()
    yield
    reset()


def write_tone(path: Path, seconds: float = 12.0, sr: int = 44100) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), np.stack([tone, tone], axis=1), sr)
    return path


@pytest.fixture
def tone_file(tmp_path):
    return write_tone(tmp_path / "tone.wav")


def make_completed_task(task_id: str, stems=("vocals", "drums", "bass", "other"), with_previews=True):
    """Create the on-disk layout of a finished 4-stem job."""
    stems_dir = main.stems_dir_for(task_id)
    stems_dir.mkdir(parents=True)
    for stem in stems:
        write_tone(stems_dir / f"{stem}{main.MASTER_EXT}", seconds=11)
    if with_previews:
        main.create_previews({s: stems_dir / f"{s}{main.MASTER_EXT}" for s in stems},
                             main.preview_dir_for(task_id))
    main.write_json_atomic(main.task_meta_path(task_id), {
        "task_id": task_id, "status": "completed", "progress": 100, "stems": list(stems),
        "original_name": "Song.mp3", "mode": "4stems", "duration": 11.0, "bpm": 120,
    })
