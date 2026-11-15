"""Tempo (BPM) estimation from a spectral-flux onset envelope."""
import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from app.media_tools import MediaToolError, find_ffmpeg

logger = logging.getLogger("stem_separator.tempo")

ANALYSIS_SR = 11025
FRAME_SIZE = 512
HOP_SIZE = 128
MIN_BPM = 60.0
MAX_BPM = 200.0
# Log-normal tempo prior centred on 120 BPM with a one-octave spread. This is the
# usual way to resolve half/double-tempo ambiguity: both peaks exist in the
# autocorrelation and the prior picks the musically more common one.
PRIOR_CENTER_BPM = 120.0
PRIOR_OCTAVE_STD = 1.0


def onset_envelope(samples: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Half-wave rectified spectral flux of log-compressed STFT magnitudes."""
    if len(samples) < FRAME_SIZE * 4:
        return np.zeros(0, dtype=np.float32)
    n_frames = 1 + (len(samples) - FRAME_SIZE) // HOP_SIZE
    frames = np.lib.stride_tricks.sliding_window_view(samples, FRAME_SIZE)[::HOP_SIZE][:n_frames]
    window = np.hanning(FRAME_SIZE).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
    log_mag = np.log1p(100.0 * magnitude)
    flux = np.maximum(np.diff(log_mag, axis=0), 0.0).sum(axis=1)
    # Remove the slowly varying loudness trend so sustained passages don't dominate.
    kernel = np.ones(16) / 16.0
    flux = np.maximum(flux - np.convolve(flux, kernel, mode="same"), 0.0)
    return flux.astype(np.float32)


def _autocorrelation(envelope: np.ndarray) -> np.ndarray:
    centred = envelope - envelope.mean()
    size = 1 << int(np.ceil(np.log2(2 * len(centred) - 1)))
    spectrum = np.fft.rfft(centred, size)
    return np.fft.irfft(spectrum * np.conj(spectrum), size)[: len(centred)]


def estimate_tempo(samples: np.ndarray, sr: int = ANALYSIS_SR) -> Optional[float]:
    """Estimate tempo in BPM from mono samples, or None when there is no clear pulse."""
    envelope = onset_envelope(np.asarray(samples, dtype=np.float32), sr)
    if len(envelope) < 64 or not np.any(envelope > 0):
        return None
    frame_rate = sr / HOP_SIZE
    # Light smoothing so a beat period that falls between two integer lags still
    # registers its full peak height (otherwise double-period peaks can win).
    smoothing = np.hanning(7)
    envelope = np.convolve(envelope, smoothing / smoothing.sum(), mode="same")
    acf = _autocorrelation(envelope)
    if acf[0] <= 0:
        return None

    min_lag = int(np.floor(60.0 * frame_rate / MAX_BPM))
    max_lag = min(int(np.ceil(60.0 * frame_rate / MIN_BPM)), len(acf) - 2)
    if max_lag <= min_lag + 1:
        return None
    lags = np.arange(min_lag, max_lag + 1)
    bpms = 60.0 * frame_rate / lags
    prior = np.exp(-0.5 * (np.log2(bpms / PRIOR_CENTER_BPM) / PRIOR_OCTAVE_STD) ** 2)
    strength = np.maximum(acf[lags], 0.0)
    scores = strength * prior
    best = int(np.argmax(scores))
    if strength[best] < 0.1 * acf[0]:
        return None  # No periodicity worth reporting.

    # Parabolic interpolation around the peak for sub-frame lag precision.
    lag = float(lags[best])
    if 0 < best < len(lags) - 1:
        y0, y1, y2 = acf[lags[best] - 1], acf[lags[best]], acf[lags[best] + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            lag += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
    return 60.0 * frame_rate / lag


def _decode_mono(audio_path: Path, offset: float, duration: float) -> np.ndarray:
    cmd = [find_ffmpeg(), "-v", "error"]
    if offset:
        cmd += ["-ss", str(offset)]
    cmd += ["-t", str(duration), "-i", str(audio_path), "-ac", "1", "-ar", str(ANALYSIS_SR),
            "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
    if proc.returncode != 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def estimate_bpm(audio_path: Path) -> Optional[int]:
    """Estimate the tempo of an audio file. Returns None if it can't be determined."""
    if not audio_path.exists():
        return None
    try:
        samples = _decode_mono(audio_path, offset=10, duration=90)
        if len(samples) < ANALYSIS_SR * 8:  # Short track: analyse from the start.
            samples = _decode_mono(audio_path, offset=0, duration=90)
        tempo = estimate_tempo(samples)
        return int(round(tempo)) if tempo else None
    except (OSError, subprocess.SubprocessError, MediaToolError, ValueError) as exc:
        logger.warning("BPM estimation failed for %s: %s", audio_path, exc)
        return None
