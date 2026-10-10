"""Decode audio with ffmpeg and turn it into numbers: a stable ID, a numeric signature, a comparison table.  numpy / scipy only."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess

import numpy as np
from scipy import signal as sps

from . import config as C

BANDS = (("sub500", 0.0, 500.0), ("500_2k", 500.0, 2000.0), ("2k_8k", 2000.0, 8000.0), ("8k_plus", 8000.0, None))
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class AudioError(ValueError):
    """Something about the file (no ffmpeg, no audio stream, cannot be decoded, empty window) the caller should hear about."""


def find_ffmpeg() -> str | None:
    env = os.environ.get("AUDIO_DESCRIBE_FFMPEG")
    if env and os.path.isfile(env):
        return env
    return shutil.which("ffmpeg")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)


_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_STREAM = re.compile(r"Audio:.*?,\s*(\d+)\s*Hz,\s*([^,\n]+)")


def probe(path) -> dict:
    """{duration_s, sr_hz, channels} of the first audio stream, parsed from `ffmpeg -i` (so ffprobe is not needed)."""
    ff = find_ffmpeg()
    if not ff:
        raise AudioError("ffmpeg not found (put it on PATH or set AUDIO_DESCRIBE_FFMPEG)")
    err = _run([ff, "-hide_banner", "-i", str(path)]).stderr.decode("utf-8", "replace")
    m = _STREAM.search(err)
    if not m:
        raise AudioError(f"no audio stream found in {path}")
    d = _DURATION.search(err)
    dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else 0.0
    layout = m.group(2).strip()
    channels = {"mono": 1, "stereo": 2}.get(layout)
    if channels is None:
        n = re.match(r"(\d+) channels", layout)
        channels = int(n.group(1)) if n else {"2.1": 3, "quad": 4, "5.1": 6, "7.1": 8}.get(layout.split("(")[0], 2)
    return {"duration_s": round(dur, 3), "sr_hz": int(m.group(1)), "channels": channels}


def decode(path, start_s: float, dur_s: float, sr: int = C.ANALYSIS_SR) -> np.ndarray:
    """Mono float32 samples of [start_s, start_s + dur_s) resampled to `sr`.  Arguments go to ffmpeg as a list (no shell),
    so spaces and non-ASCII characters in the path are fine."""
    ff = find_ffmpeg()
    if not ff:
        raise AudioError("ffmpeg not found (put it on PATH or set AUDIO_DESCRIBE_FFMPEG)")
    cmd = [ff, "-v", "error", "-ss", f"{max(0.0, float(start_s)):.3f}", "-i", str(path), "-t", f"{float(dur_s):.3f}",
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    res = _run(cmd)
    if res.returncode != 0 or not res.stdout:
        lines = res.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioError(f"could not decode {path}: {lines[-1] if lines else 'no audio was produced'}")
    return np.frombuffer(res.stdout, dtype="<f4").astype(np.float32, copy=True)


def audio_id(path) -> str:
    """`a-` + the first 10 hex digits of the SHA-1 of the file's bytes: identical copies share it."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return "a-" + h.hexdigest()[:10]


def window(duration_s: float, start_s, end_s, cap_s: float) -> tuple[float, float, bool]:
    """Clamp a requested part of a file to what exists and to the per-call cap.  Returns (start, end, truncated)."""
    start = max(0.0, float(start_s or 0.0))
    end = float(end_s) if end_s is not None else (duration_s or start + cap_s)
    if duration_s:
        end = min(end, duration_s)
    if end <= start:
        raise AudioError(f"empty window: start {start:.2f} s, end {end:.2f} s, file length {duration_s:.2f} s")
    over = end - start > cap_s
    truncated = over or (not duration_s and end_s is None)  # an unknown length may well be longer than the cap: say so
    return start, (start + cap_s if over else end), truncated


def _db(x: float) -> float:
    return round(20.0 * float(np.log10(max(x, 1e-6))), 1)  # 1e-6 -> -120 dBFS


def signature(x: np.ndarray, sr: int) -> dict:
    """Numbers that tell two sounds apart.  `x`: mono samples in [-1, 1] at sample rate `sr`."""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    peak = float(np.max(np.abs(x))) if n else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if n else 0.0
    sig = {"duration_s": round(n / sr, 3), "rms_dbfs": _db(rms), "peak_dbfs": _db(peak)}
    if peak < 1e-6:  # silence: defined numbers, no division by zero
        sig.update(dom_hz=0.0, centroid_hz=0.0, rolloff95_hz=0.0, flatness=0.0, attack_ms=0, decay_ms=0, decay_found=False)
        sig.update({f"band_{name}": 0.0 for name, _, _ in BANDS})
        return sig
    pad = max(0, 256 - n)
    f, p = sps.welch(np.pad(x, (0, pad)), fs=sr, window="hann", nperseg=min(4096, n + pad))
    e = p * (f[1] - f[0])  # energy per bin
    total = float(e.sum()) or 1e-30
    for name, lo, hi in BANDS:
        m = (f >= lo) & ((f < hi) if hi is not None else True)
        sig[f"band_{name}"] = round(float(e[m].sum()) / total, 3)
    hz = f >= 20.0
    sig["dom_hz"] = round(float(f[hz][int(np.argmax(p[hz]))]), 1)
    sig["centroid_hz"] = round(float((f * e).sum() / total), 1)
    sig["rolloff95_hz"] = round(float(f[min(int(np.searchsorted(np.cumsum(e) / total, 0.95)), len(f) - 1)]), 1)
    # flatness (0 = tone, 1 = white noise) over the band the sound actually occupies.  Over the whole 0-24 kHz it reads ~0 for
    # every sound whose energy sits low, so a thunder rumble looked "tonal" (measured on a sound-effect library).
    ph = p[hz & (f <= max(sig["rolloff95_hz"] * 1.5, 200.0))] + 1e-30
    sig["flatness"] = round(float(np.exp(np.mean(np.log(ph))) / np.mean(ph)), 3)
    frame = max(1, int(sr * 0.005))  # 5 ms envelope frames
    k = n // frame
    env = np.sqrt(np.mean(x[: k * frame].reshape(k, frame) ** 2, axis=1)) if k else np.array([rms])
    top_env = float(np.max(env))
    # the attack ends at the FIRST frame within 3 dB of the loudest, not at the loudest frame itself: in a steady tone or buzz the
    # loudest 5 ms frame can be anywhere (measured: attack 1995 ms for a 2 s sine), which read as a slow fade-in
    pk = int(np.argmax(env >= top_env * 10 ** (-3 / 20)))
    below = np.nonzero(env[pk:] < top_env * 10 ** (-30 / 20))[0]
    sig["attack_ms"] = int(round(pk * 5))
    sig["decay_found"] = bool(below.size)
    sig["decay_ms"] = int(round((below[0] if below.size else env.size - pk) * 5))
    return sig


# ---- comparison table ------------------------------------------------------------------------------------------------
def _ratio_ge(a: float, b: float, r: float) -> bool:
    lo, hi = sorted((abs(a), abs(b)))
    return hi > 0 and (lo == 0 or hi / lo >= r)


_ROWS = (
    ("duration", "duration_s", "{:.2f} s", lambda a, b: _ratio_ge(a, b, 1.25)),
    ("rms", "rms_dbfs", "{:.1f} dBFS", lambda a, b: abs(a - b) >= 3.0),
    ("peak", "peak_dbfs", "{:.1f} dBFS", lambda a, b: abs(a - b) >= 3.0),
    ("dominant", "dom_hz", "{:.0f} Hz", lambda a, b: _ratio_ge(a, b, 1.12)),
    ("centroid", "centroid_hz", "{:.0f} Hz", lambda a, b: _ratio_ge(a, b, 1.2)),
    ("95% rolloff", "rolloff95_hz", "{:.0f} Hz", lambda a, b: _ratio_ge(a, b, 1.2)),
    ("flatness", "flatness", "{:.2f}", lambda a, b: abs(a - b) >= 0.15),
    ("<500 Hz", "band_sub500", "{:.0%}", lambda a, b: abs(a - b) >= 0.10),
    ("500-2k", "band_500_2k", "{:.0%}", lambda a, b: abs(a - b) >= 0.10),
    ("2k-8k", "band_2k_8k", "{:.0%}", lambda a, b: abs(a - b) >= 0.10),
    (">8k", "band_8k_plus", "{:.0%}", lambda a, b: abs(a - b) >= 0.10),
    ("attack", "attack_ms", "{:.0f} ms", lambda a, b: abs(a - b) >= 20 and _ratio_ge(a, b, 1.5)),
    ("decay", "decay_ms", "{:.0f} ms", lambda a, b: abs(a - b) >= 50 and _ratio_ge(a, b, 1.5)),
)


def compare_rows(a: dict, b: dict) -> list[dict]:
    """One row per signature field: label, A and B formatted, and whether the difference is worth mentioning."""
    rows = []
    for label, key, fmt, notable in _ROWS:
        va, vb = float(a[key]), float(b[key])
        rows.append({"label": label, "a": fmt.format(va), "b": fmt.format(vb), "notable": bool(notable(va, vb))})
    return rows
