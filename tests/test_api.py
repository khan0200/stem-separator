import uuid

import pytest
from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient

from app import main
from tests.conftest import make_completed_task

# No `with` block: the lifespan (worker thread, recovery) does not run in these tests.
client = TestClient(main.app)


def test_status_includes_duration_and_bpm():
    task_id = str(uuid.uuid4())
    make_completed_task(task_id)
    body = client.get(f"/api/status/{task_id}").json()
    assert body["status"] == "completed"
    assert body["duration"] == 11.0
    assert body["bpm"] == 120
    assert body["stems"] == ["vocals", "drums", "bass", "other"]


def test_unknown_task_is_404():
    assert client.get(f"/api/status/{uuid.uuid4()}").status_code == 404
    assert client.get("/api/status/not-a-uuid").status_code == 404


def test_upload_is_validated_saved_and_queued(tone_file):
    with open(tone_file, "rb") as f:
        res = client.post("/api/separate", files={"file": ("My Song.wav", f, "audio/wav")},
                          data={"mode": "2stems"})
    assert res.status_code == 200, res.text
    task_id = res.json()["task_id"]
    queued = main.job_queue.get_nowait()
    main.job_queue.task_done()
    assert queued[0] == task_id and queued[2] == "2stems"
    status = client.get(f"/api/status/{task_id}").json()
    assert status["status"] == "pending"
    assert status["duration"] == pytest.approx(12.0, abs=0.1)


def test_invalid_upload_leaves_nothing_behind(tmp_path):
    bogus = tmp_path / "bogus.mp3"
    bogus.write_bytes(b"nope" * 1000)
    with open(bogus, "rb") as f:
        res = client.post("/api/separate", files={"file": ("bogus.mp3", f, "audio/mpeg")})
    assert res.status_code == 400
    assert list(main.DATA_DIR.iterdir()) == []


def test_cancel_pending_task_removes_upload(tone_file):
    with open(tone_file, "rb") as f:
        task_id = client.post("/api/separate", files={"file": ("a.wav", f, "audio/wav")}).json()["task_id"]
    assert client.post(f"/api/cancel/{task_id}").status_code == 200
    assert client.get(f"/api/status/{task_id}").json()["status"] == "cancelled"
    assert main.find_input_file(task_id) is None
    assert client.post(f"/api/cancel/{task_id}").status_code == 409


def test_stream_and_export_prepare_then_download():
    task_id = str(uuid.uuid4())
    make_completed_task(task_id)

    stream = client.get(f"/api/stream/{task_id}/vocals")
    assert stream.status_code == 200
    assert stream.headers["content-type"] == "audio/mpeg"

    prepared = client.get(f"/api/export/{task_id}/drums", params={"format": "wav", "quality": "wav24", "prepare": True})
    assert prepared.json() == {"ready": True}
    download = client.get(f"/api/export/{task_id}/drums", params={"format": "wav", "quality": "wav24"})
    assert download.status_code == 200
    assert download.content[:4] == b"RIFF"

    archive = client.get(f"/api/export-all/{task_id}", params={"format": "mp3", "quality": "192k", "prepare": True})
    assert archive.json() == {"ready": True}


def test_export_errors_are_json():
    task_id = str(uuid.uuid4())
    make_completed_task(task_id, with_previews=False)
    res = client.get(f"/api/export/{task_id}/vocals", params={"format": "mp3", "quality": "999k"})
    assert res.status_code == 400
    assert res.json()["detail"] == "Invalid export quality."


def _tiny_upload_app(limit: int) -> TestClient:
    tiny = FastAPI()

    @tiny.post("/api/separate")
    async def upload(file: UploadFile = File(...)):
        return {"size": len(await file.read())}

    tiny.add_middleware(main.UploadSizeLimitMiddleware, max_body_size=limit, paths={"/api/separate"})
    return TestClient(tiny)


def test_size_limit_rejects_by_content_length_before_reading_body():
    tiny = _tiny_upload_app(limit=2000)
    assert tiny.post("/api/separate", files={"file": ("a.mp3", b"x" * 500)}).status_code == 200
    assert tiny.post("/api/separate", files={"file": ("a.mp3", b"x" * 5000)}).status_code == 413


def test_size_limit_applies_to_streamed_bodies_without_content_length():
    tiny = _tiny_upload_app(limit=2000)
    boundary = "testboundary"
    payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\n"
               f"Content-Type: audio/mpeg\r\n\r\n").encode() + b"x" * 10000 + f"\r\n--{boundary}--\r\n".encode()

    def chunks():
        for i in range(0, len(payload), 1000):
            yield payload[i:i + 1000]

    res = tiny.post("/api/separate", content=chunks(),
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert res.status_code == 413
