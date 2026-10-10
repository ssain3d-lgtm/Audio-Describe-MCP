"""Synthetic sounds for the tests: no audio files are shipped with the repository."""
import wave
from pathlib import Path

import numpy as np

SR = 48000


def tone(hz, seconds=1.0, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * hz * t)


def write_wav(path, x, sr=SR, channels=1):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    pcm = np.repeat(pcm[:, None], channels, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return Path(path)


def lowpassed_noise(cutoff_hz=400, seconds=2.0, seed=0):
    from scipy import signal as sps

    x = np.random.default_rng(seed).normal(0, 0.1, int(SR * seconds))
    x = sps.sosfilt(sps.butter(4, cutoff_hz, "low", fs=SR, output="sos"), x)
    return x / np.max(np.abs(x)) * 0.5
