import numpy as np
import pytest

from app.tempo import ANALYSIS_SR, estimate_tempo


def click_track(bpm: float, seconds: float = 60.0, accent_every: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = np.zeros(int(ANALYSIS_SR * seconds), dtype=np.float32)
    burst_len = int(0.01 * ANALYSIS_SR)
    burst = (rng.standard_normal(burst_len) * np.exp(-np.linspace(0, 8, burst_len))).astype(np.float32)
    period, t, beat = 60.0 / bpm, 0.0, 0
    while t < seconds - 0.05:
        start = int(t * ANALYSIS_SR)
        signal[start:start + burst_len] += burst * (1.0 if beat % accent_every == 0 else 0.6)
        t += period
        beat += 1
    return signal + 0.01 * rng.standard_normal(len(signal)).astype(np.float32)


@pytest.mark.parametrize("bpm", [62, 75, 87, 100, 120, 128, 140, 150])
@pytest.mark.parametrize("accent_every", [1, 4])
def test_estimates_click_track_tempo(bpm, accent_every):
    assert estimate_tempo(click_track(bpm, accent_every=accent_every)) == pytest.approx(bpm, abs=1.5)


def test_no_tempo_for_noise_or_silence():
    rng = np.random.default_rng(3)
    assert estimate_tempo(0.1 * rng.standard_normal(ANALYSIS_SR * 30).astype(np.float32)) is None
    assert estimate_tempo(np.zeros(ANALYSIS_SR * 30, dtype=np.float32)) is None
