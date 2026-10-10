"""Draw a sound as a picture a vision-capable model can read: waveform on top, a linear-frequency spectrogram below,
kHz / second ticks and a dotted line at 8 kHz (the ceiling of the optional embedding model).  numpy / scipy / Pillow only."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import signal as sps

LEFT, RIGHT, TOP, GAP, BOTTOM = 56, 8, 22, 6, 26
PLOT_W, WAVE_H, SPEC_H = 1000, 80, 320
_ANCHORS = [(0.00, (0, 0, 4)), (0.20, (40, 11, 84)), (0.40, (120, 28, 109)), (0.60, (203, 71, 71)),
            (0.80, (251, 156, 6)), (1.00, (252, 255, 164))]  # an inferno-like colour ramp, built by hand (no matplotlib)
_LUT = np.stack([np.interp(np.linspace(0, 1, 256), [a for a, _ in _ANCHORS], [c[i] for _, c in _ANCHORS]) for i in range(3)],
                axis=1).astype(np.uint8)
_DIM = (200, 200, 200)
_CYAN = (0, 255, 255)


def spec_y(freq_hz: float, fmax: float) -> int:
    """Pixel row (in the whole canvas) of a frequency on the spectrogram axis."""
    return int(round(TOP + WAVE_H + GAP + SPEC_H * (1.0 - min(max(freq_hz / fmax, 0.0), 1.0))))


def spectrogram_image(x: np.ndarray, sr: int, title: str = "", fmax: float | None = None) -> tuple[Image.Image, dict]:
    """Returns (image, meta); meta has fmax, y_8k (None if the axis stops at or below 8 kHz), plot_box and duration_s."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        x = np.zeros(1, dtype=np.float32)
    nyq = sr / 2.0
    fmax = float(min(fmax or nyq, nyq))
    dur = x.size / sr
    nper = 1024
    xx = np.pad(x, (0, nper - x.size)) if x.size < nper else x
    f, _, S = sps.spectrogram(xx, fs=sr, window="hann", nperseg=nper, noverlap=nper * 3 // 4, mode="magnitude", scaling="spectrum")
    db = 20.0 * np.log10(S[f <= fmax + 1e-6] + 1e-10)
    # Absolute floor for the brightest bin: silence stays black instead of being stretched to full brightness.  -40 dB (not lower)
    # because a 16-bit file's +-1 LSB dither sits around -116 dB per bin and must not be drawn as a wall of noise; a -60 dBFS tone
    # (-66 dB) still shows at about two thirds brightness.
    top = max(float(db.max()), -40.0)
    norm = np.clip((db - (top - 80.0)) / 80.0, 0.0, 1.0)
    spec = Image.fromarray(_LUT[(norm[::-1] * 255).astype(np.uint8)]).resize((PLOT_W, SPEC_H), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (LEFT + PLOT_W + RIGHT, TOP + WAVE_H + GAP + SPEC_H + BOTTOM), (16, 16, 20))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    y0 = TOP + WAVE_H + GAP
    canvas.paste(spec, (LEFT, y0))

    mid, half = TOP + WAVE_H // 2, WAVE_H // 2 - 2
    peak = max(float(np.max(np.abs(x))), 1e-6)
    idx = np.linspace(0, x.size, PLOT_W + 1).astype(int)
    for i in range(PLOT_W):
        lo = min(int(idx[i]), x.size - 1)
        seg = x[lo:max(int(idx[i + 1]), lo + 1)]
        ya = mid - int(float(seg.max()) / peak * half)
        yb = mid - int(float(seg.min()) / peak * half)
        d.line([(LEFT + i, ya), (LEFT + i, max(ya, yb))], fill=(120, 200, 255))

    step_hz = 2000.0 if fmax <= 16000 else 4000.0
    for k in range(int(fmax // step_hz) + 1):
        hz = k * step_hz
        y = spec_y(hz, fmax)
        d.line([(LEFT - 4, y), (LEFT, y)], fill=_DIM)
        d.text((4, min(max(y - 5, y0 - 2), y0 + SPEC_H - 10)), "0" if k == 0 else f"{hz / 1000:g}k", fill=_DIM, font=font)
    y8 = None
    if fmax > 8000:
        y8 = spec_y(8000, fmax)
        for xs in range(LEFT, LEFT + PLOT_W, 12):
            d.line([(xs, y8), (min(xs + 6, LEFT + PLOT_W), y8)], fill=_CYAN)
        d.text((LEFT + PLOT_W - 96, y8 - 12), "model limit 8k", fill=_CYAN, font=font)
    step = 120.0
    for cand in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0):
        if dur / cand <= 10:
            step = cand
            break
    for i in range(int(dur / step) + 1):
        t = i * step
        px = LEFT + int(round(t / dur * PLOT_W))
        d.line([(px, y0 + SPEC_H), (px, y0 + SPEC_H + 4)], fill=_DIM)
        d.text((px - 8, y0 + SPEC_H + 8), f"{t:g}s", fill=_DIM, font=font)
    d.text((LEFT, 4), title[:140], fill=(235, 235, 235), font=font)
    return canvas, {"fmax": fmax, "y_8k": y8, "plot_box": (LEFT, y0, LEFT + PLOT_W, y0 + SPEC_H), "duration_s": dur}


def _save_png(img: Image.Image, out_path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part")  # not *.png: prune_dir must never delete another process's file in flight
    img.save(tmp, format="PNG", optimize=True)
    os.replace(tmp, out)
    return out


def write_spectrogram(x, sr: int, out_path, title: str = "", fmax: float | None = None) -> Path:
    img, _ = spectrogram_image(x, sr, title, fmax)
    return _save_png(img, out_path)


def stacked_png(x_a, x_b, sr: int, out_path, title_a: str = "", title_b: str = "") -> Path:
    """A on top of B with the same frequency scale (each keeps its own time axis)."""
    ia, _ = spectrogram_image(x_a, sr, "A  " + title_a)
    ib, _ = spectrogram_image(x_b, sr, "B  " + title_b)
    both = Image.new("RGB", (ia.width, ia.height + ib.height), (16, 16, 20))
    both.paste(ia, (0, 0))
    both.paste(ib, (0, ia.height))
    return _save_png(both, out_path)


def prune_dir(folder, max_bytes: int) -> int:
    """Delete the oldest PNGs until the folder holds at most `max_bytes`.  Returns how many were removed."""
    files = sorted((p for p in Path(folder).glob("*.png") if p.is_file()), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    removed = 0
    for p in files:
        if total <= max_bytes:
            break
        total -= p.stat().st_size
        p.unlink(missing_ok=True)
        removed += 1
    return removed
