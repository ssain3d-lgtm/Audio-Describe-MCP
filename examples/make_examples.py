"""Make two small synthetic comparisons and the pictures used in the README (no third-party audio is involved).

    python examples/make_examples.py            # writes docs/img/*.png and prints the text a model would receive

Needs only the base install and ffmpeg.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy import signal as sps

from audio_describe_mcp.core import Describer, format_compare, format_describe

SR = 48000
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "img"


def write_wav(path: Path, x: np.ndarray) -> str:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return str(path)


def rumble(seconds: float = 4.0) -> np.ndarray:
    """Thunder-like: low-passed noise that swells for 0.15 s and then dies away over a few seconds."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = np.random.default_rng(1).normal(0, 1, n)
    x = sps.sosfilt(sps.butter(4, 350, "low", fs=SR, output="sos"), x)
    env = np.minimum(t / 0.15, 1.0) * np.exp(-t / 1.1)
    return 0.8 * x / np.max(np.abs(x)) * env


def ping(seconds: float = 2.0) -> np.ndarray:
    """Bell-like: three decaying partials."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum(a * np.sin(2 * np.pi * f * t) for a, f in ((1.0, 1200), (0.5, 2400), (0.25, 3600)))
    return 0.5 * x / 1.75 * np.exp(-t / 0.4)


def tone(hz: float, seconds: float = 2.0) -> np.ndarray:
    return 0.4 * np.sin(2 * np.pi * hz * np.arange(int(seconds * SR)) / SR)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="audio-describe-examples-"))
    os.environ["AUDIO_DESCRIBE_CACHE"] = str(work / "cache")
    try:
        files = {name: write_wav(work / f"{name}.wav", x) for name, x in
                 (("rumble", rumble()), ("ping", ping()), ("tone-6k", tone(6000)), ("tone-12k", tone(12000)))}
        d = Describer()
        OUT.mkdir(parents=True, exist_ok=True)
        print("=== describe rumble.wav ===")
        print(format_describe(d.describe(files["rumble"])))
        for a, b, name in (("rumble", "ping", "compare-rumble-vs-ping"), ("tone-6k", "tone-12k", "compare-6k-vs-12k")):
            res = d.compare(files[a], files[b])
            shutil.copy(res["stacked_png"], OUT / f"{name}.png")
            print(f"\n=== compare {a}.wav {b}.wav ===")
            print(format_compare(res).replace(str(work), "<tmp>"))
        print(f"\npictures written to {OUT}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
