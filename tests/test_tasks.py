import json
import os
import time
import uuid

from app import main
from tests.conftest import make_completed_task, write_tone


def new_id() -> str:
    return str(uuid.uuid4())


def write_meta(task_id: str, **meta) -> None:
    main.task_dir_for(task_id).mkdir(parents=True, exist_ok=True)
    main.write_json_atomic(main.task_meta_path(task_id), {"task_id": task_id, **meta})


def test_write_json_atomic_replaces_without_leftovers():
    task_id = new_id()
    write_meta(task_id, status="pending")
    write_meta(task_id, status="completed")
    folder = main.task_dir_for(task_id)
    assert json.loads((folder / "meta.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert [p.name for p in folder.iterdir()] == ["meta.json"]


def test_failed_job_with_partial_stems_stays_failed_after_restart():
    """Regression: stems on disk used to make an errored task look completed."""
    task_id = new_id()
    stems_dir = main.stems_dir_for(task_id)
    stems_dir.mkdir(parents=True)
    write_tone(stems_dir / "vocals.flac")
    write_tone(stems_dir / "instrumental.flac")
    write_meta(task_id, status="error", error="Demucs crashed", mode="4stems")

    info = main.get_task(task_id)
    assert info["status"] == "error"
    assert info["error"] == "Demucs crashed"
    assert json.loads(main.task_meta_path(task_id).read_text(encoding="utf-8"))["status"] == "error"


def test_completed_task_with_missing_files_is_reported_as_error():
    task_id = new_id()
    write_meta(task_id, status="completed", stems=["vocals", "drums"])
    assert main.get_task(task_id)["status"] == "error"


def test_legacy_meta_without_status_uses_zip_as_completion_marker():
    done, unfinished = new_id(), new_id()
    for task_id in (done, unfinished):
        legacy_dir = main.task_dir_for(task_id) / "output" / "htdemucs" / "input"
        legacy_dir.mkdir(parents=True)
        for stem in ("vocals", "drums", "bass", "other"):
            (legacy_dir / f"{stem}.mp3").write_bytes(b"mp3")
        write_meta(task_id, original_name="Old.mp3", mode="4stems")
    (main.task_dir_for(done) / "output" / "audio_stems.zip").write_bytes(b"zip")

    assert main.get_task(done)["status"] == "completed"
    assert main.get_task(done)["stems"] == ["vocals", "drums", "bass", "other"]
    assert main.get_task(unfinished)["status"] == "error"


def test_active_task_unknown_to_this_process_is_not_left_polling():
    task_id = new_id()
    write_meta(task_id, status="processing")
    info = main.get_task(task_id)
    assert info["status"] == "error"
    assert "interrupted" in info["error"]


def test_get_task_does_not_write_to_disk():
    task_id = new_id()
    make_completed_task(task_id, with_previews=False)
    meta_path = main.task_meta_path(task_id)
    before = meta_path.read_bytes(), meta_path.stat().st_mtime_ns
    main.get_task(task_id)
    main.get_task(task_id)
    assert (meta_path.read_bytes(), meta_path.stat().st_mtime_ns) == before


def test_get_task_returns_a_copy():
    task_id = new_id()
    make_completed_task(task_id, with_previews=False)
    main.get_task(task_id)["status"] = "mutated"
    assert main.get_task(task_id)["status"] == "completed"


def test_cleanup_removes_expired_cancelled_and_orphan_tasks():
    now = time.time()
    old = now - main.TASK_RETENTION_SECONDS - 60
    cancelled, completed_recent, orphan_old, orphan_new = new_id(), new_id(), new_id(), new_id()

    write_meta(cancelled, status="cancelled", updated_at=old)
    write_meta(completed_recent, status="completed", updated_at=now)
    for orphan in (orphan_old, orphan_new):
        main.task_dir_for(orphan).mkdir(parents=True)
        (main.task_dir_for(orphan) / "input.mp3").write_bytes(b"partial upload")
    stale = now - main.ORPHAN_MAX_AGE_SECONDS - 60
    os.utime(main.task_dir_for(orphan_old), (stale, stale))

    assert main.cleanup_old_tasks(now=now) == 2
    assert not main.task_dir_for(cancelled).exists()
    assert not main.task_dir_for(orphan_old).exists()
    assert main.task_dir_for(completed_recent).exists()
    assert main.task_dir_for(orphan_new).exists()


def test_cleanup_never_removes_running_task():
    task_id = new_id()
    old = time.time() - main.TASK_RETENTION_SECONDS - 60
    write_meta(task_id, status="processing", updated_at=old)
    main.tasks_db[task_id] = {"status": "processing", "updated_at": old}
    assert main.cleanup_old_tasks() == 0
    assert main.task_dir_for(task_id).exists()


def test_recovery_requeues_then_gives_up_after_max_attempts():
    retry, give_up, no_input = new_id(), new_id(), new_id()
    for task_id, attempts in ((retry, 1), (give_up, main.MAX_JOB_ATTEMPTS), (no_input, 0)):
        write_meta(task_id, status="processing", attempts=attempts, mode="2stems", created_at=1)
    for task_id in (retry, give_up):
        (main.task_dir_for(task_id) / "input.wav").write_bytes(b"audio")

    main.recover_tasks()

    assert main.get_task(retry)["status"] == "pending"
    assert main.job_queue.get_nowait()[0] == retry
    main.job_queue.task_done()
    assert main.job_queue.empty()
    assert main.get_task(give_up)["status"] == "error"
    assert main.get_task(no_input)["status"] == "error"
    saved = json.loads(main.task_meta_path(give_up).read_text(encoding="utf-8"))
    assert saved["status"] == "error"


def test_mark_cancelled_frees_input_and_outputs():
    task_id = new_id()
    write_meta(task_id, status="pending")
    main.tasks_db[task_id] = {"status": "pending"}
    (main.task_dir_for(task_id) / "input.mp3").write_bytes(b"x" * 1000)
    main.stems_dir_for(task_id).mkdir(parents=True)

    main.mark_cancelled(task_id)

    assert [p.name for p in main.task_dir_for(task_id).iterdir()] == ["meta.json"]
    assert main.get_task(task_id)["status"] == "cancelled"


def test_cancelled_job_is_skipped_by_worker_logic():
    """A job cancelled while queued must not reach the separation engine."""
    task_id = new_id()
    write_meta(task_id, status="pending")
    main.tasks_db[task_id] = {"status": "pending"}
    main.cancelled_tasks.add(task_id)

    called = []
    original = main.separator_engine
    main.separator_engine = type("Engine", (), {
        "separate_4stems": lambda *a, **k: called.append(True),
    })()
    try:
        main.run_separation_job(task_id, main.task_dir_for(task_id) / "input.mp3", "4stems")
    finally:
        main.separator_engine = original
    assert not called
    assert main.get_task(task_id)["status"] == "cancelled"


def test_running_job_cancels_at_next_progress_checkpoint(tone_file):
    task_id = new_id()
    write_meta(task_id, status="pending")
    main.tasks_db[task_id] = {"status": "pending"}
    chunks_run = []

    class FakeEngine:
        def separate_4stems(self, input_path, out_dir, progress_cb):
            for i in range(100):
                if i == 3:
                    main.cancelled_tasks.add(task_id)  # User presses Cancel mid-stage.
                progress_cb(5 + i * 0.5, "Separating...")
                chunks_run.append(i)
            raise AssertionError("cancellation was ignored")

    original = main.separator_engine
    main.separator_engine = FakeEngine()
    try:
        main.run_separation_job(task_id, tone_file, "4stems")
    finally:
        main.separator_engine = original
    assert chunks_run == [0, 1, 2]
    assert main.get_task(task_id)["status"] == "cancelled"


def test_failed_conversion_never_leaves_a_cached_file(monkeypatch, tmp_path, tone_file):
    dest = tmp_path / "exports" / "vocals_wav16.wav"

    def fake_run(cmd, **kwargs):
        # Simulate ffmpeg dying halfway: partial output written, non-zero exit.
        with open(cmd[-1], "wb") as f:
            f.write(b"RIFF partial")
        return type("Result", (), {"returncode": 1, "stderr": "killed"})()

    monkeypatch.setattr(main.subprocess, "run", fake_run)
    try:
        main.export_audio_file(tone_file, dest, "wav", "wav16")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected failure")
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []


def test_export_conversion_and_cache(tmp_path, tone_file):
    dest = tmp_path / "vocals_320k.mp3"
    main.export_audio_file(tone_file, dest, "mp3", "320k")
    assert dest.stat().st_size > 0
    mtime = dest.stat().st_mtime_ns
    main.export_audio_file(tone_file, dest, "mp3", "320k")
    assert dest.stat().st_mtime_ns == mtime  # Served from cache.


def test_waveform_peaks_are_normalised():
    import numpy as np

    peaks = main.peaks_from_samples(np.linspace(-1, 1, 12000, dtype=np.float32), num_bars=10)
    assert len(peaks) == 10
    assert max(peaks) == 1.0
    assert all(0 <= p <= 1 for p in peaks)
