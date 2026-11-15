import pytest

from app import main, media_tools
from app.media_tools import parse_ffmpeg_info, probe_audio

FFMPEG_BANNER = """Input #0, mp3, from 'song.mp3':
  Duration: 00:03:25.47, start: 0.025057, bitrate: 320 kb/s
  Stream #0:0: Audio: mp3 (mp3float), 44100 Hz, stereo, fltp, 320 kb/s
  Stream #0:1: Video: mjpeg (Baseline), yuvj420p(pc), 500x500, 90k tbr (attached pic)
At least one output file must be specified"""


def test_parse_ffmpeg_info_reads_duration_and_audio_stream():
    probe = parse_ffmpeg_info(FFMPEG_BANNER)
    assert probe.duration == pytest.approx(205.47)
    assert probe.has_audio


def test_parse_ffmpeg_info_without_audio_stream():
    probe = parse_ffmpeg_info("Input #0, image2\n  Duration: 00:00:00.04\n  Stream #0:0: Video: png")
    assert not probe.has_audio


@pytest.fixture
def without_ffprobe(monkeypatch):
    """Simulate a machine that only has imageio-ffmpeg (no ffprobe binary)."""
    monkeypatch.setattr(media_tools, "find_ffprobe", lambda: None)


def test_probe_falls_back_to_ffmpeg_when_ffprobe_missing(without_ffprobe, tone_file):
    probe = probe_audio(tone_file)
    assert probe.has_audio
    assert probe.duration == pytest.approx(12.0, abs=0.1)


def test_validate_accepts_audio_without_ffprobe(without_ffprobe, tone_file):
    assert main.validate_audio_file(tone_file) == pytest.approx(12.0, abs=0.1)


def test_validate_rejects_non_audio(without_ffprobe, tmp_path):
    bogus = tmp_path / "fake.mp3"
    bogus.write_bytes(b"definitely not audio" * 100)
    with pytest.raises(ValueError):
        main.validate_audio_file(bogus)


def test_validate_rejects_too_short(tmp_path):
    from tests.conftest import write_tone

    short = write_tone(tmp_path / "short.wav", seconds=3)
    with pytest.raises(ValueError, match="between"):
        main.validate_audio_file(short)
