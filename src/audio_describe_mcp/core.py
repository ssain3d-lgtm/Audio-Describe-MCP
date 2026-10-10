"""describe() and compare(): the work behind the two tools, and the text a model reads."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import config as C
from . import signature as S
from . import spectrogram as G


def _norm_path(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()


class Describer:
    """`semantic` is an optional embedding encoder (see semantic.SemanticEncoder); it is created on first use of
    compare(..., semantic=True), so the base install never imports torch."""

    def __init__(self, semantic=None):
        self._semantic = semantic

    # ---------------------------------------------------------------- analysis
    @staticmethod
    def _audio_file(path: str) -> Path:
        p = _norm_path(path)
        if not p.is_file():
            raise S.AudioError(f"audio file not found: {p}")
        if p.suffix.lower() not in C.AUDIO_EXTS:
            raise S.AudioError(f"not an audio file ({p.suffix or 'no extension'}); supported: {' '.join(C.AUDIO_EXTS)}")
        return p

    def _analyse(self, path: str, start_s: float = 0.0, end_s: float | None = None) -> tuple[dict, np.ndarray]:
        p = self._audio_file(path)
        info = S.probe(p)
        start, end, truncated = S.window(info["duration_s"], start_s, end_s, C.MAX_AUDIO_SECONDS)
        x = S.decode(p, start, end - start, C.ANALYSIS_SR)
        d = dict(id=S.audio_id(p), path=str(p), info=info, start=start, end=end, truncated=truncated,
                 sig=S.signature(x, C.ANALYSIS_SR), png=None)
        return d, x

    @staticmethod
    def _png_path(d: dict) -> Path:
        return C.cache_dir() / f"{d['id']}_{round(d['start'] * 1000)}_{round(d['end'] * 1000)}.png"

    @staticmethod
    def _title(d: dict) -> str:
        return f"{d['id']}  {Path(d['path']).name}  {d['start']:.2f}-{d['end']:.2f}s  source {d['info']['sr_hz']} Hz"

    def describe(self, path: str, start_s: float = 0.0, end_s: float | None = None, spectrogram: bool = True) -> dict:
        d, x = self._analyse(path, start_s, end_s)
        if spectrogram:
            png = self._png_path(d)
            if not png.exists():
                G.write_spectrogram(x, C.ANALYSIS_SR, png, self._title(d))
                G.prune_dir(C.cache_dir(), C.SPECTRO_MAX_BYTES)
            d["png"] = str(png)
        return d

    # ---------------------------------------------------------------- optional embedding comparison
    @staticmethod
    def _model_clip(x: np.ndarray) -> np.ndarray:
        """What the audio encoder gets: 16 kHz mono, at most SEMANTIC_MAX_SECONDS of it."""
        from scipy.signal import resample_poly

        return np.asarray(resample_poly(x[: int(C.SEMANTIC_MAX_SECONDS * C.ANALYSIS_SR)], 1, C.ANALYSIS_SR // C.MODEL_AUDIO_SR),
                          dtype=np.float32)

    def _encoder(self):
        if self._semantic is None:
            from .semantic import SemanticEncoder

            self._semantic = SemanticEncoder()
            self._semantic.start_idle_watch()
        return self._semantic

    @staticmethod
    def _blocker(enc) -> str | None:
        """Why the embedding model should not be loaded right now (little free RAM), else None."""
        if enc.loaded:
            return None
        try:
            import psutil

            free = psutil.virtual_memory().available / 2**30
        except Exception:  # noqa: BLE001
            return None
        if free < C.AUDIO_MODEL_MIN_FREE_GB:
            return (f"only {free:.1f} GB RAM free; the embedding model needs about {C.AUDIO_MODEL_MIN_FREE_GB:g} GB. "
                    f"Free some memory or call without semantic=true.")
        return None

    def compare(self, a: str, b, semantic: bool = False, spectrogram: bool = True) -> dict:
        bs = [b] if isinstance(b, str) else [str(x) for x in b]
        if not bs:
            raise ValueError("b is empty")
        if len(bs) > C.MAX_COMPARE_CANDIDATES:
            raise ValueError(f"at most {C.MAX_COMPARE_CANDIDATES} files can be compared at once")
        da, xa = self._analyse(a)
        clips = [self._model_clip(xa)] if semantic else []
        items, xb_only = [], None
        for one in bs:
            db, xb = self._analyse(one)
            items.append(dict(b=db, same_id=db["id"] == da["id"], rows=S.compare_rows(da["sig"], db["sig"]), cos=None))
            if semantic:
                clips.append(self._model_clip(xb))
            if len(bs) == 1:
                xb_only = xb
        res = dict(a=da, items=items, stacked_png=None, semantic=bool(semantic), semantic_error=None)
        if spectrogram and xb_only is not None:
            out = C.cache_dir() / f"{da['id']}_vs_{items[0]['b']['id']}.png"
            if not out.exists():
                G.stacked_png(xa, xb_only, C.ANALYSIS_SR, out, self._title(da), self._title(items[0]["b"]))
                G.prune_dir(C.cache_dir(), C.SPECTRO_MAX_BYTES)
            res["stacked_png"] = str(out)
        if semantic:
            try:
                enc = self._encoder()
                blocked = self._blocker(enc)
                if blocked:
                    res["semantic_error"] = blocked
                else:
                    vecs = enc.embed(clips)
                    for i, it in enumerate(items):
                        it["cos"] = round(float(vecs[0] @ vecs[i + 1]), 4)
                    if len(items) > 1:
                        items.sort(key=lambda it: -it["cos"])
            except Exception as exc:  # noqa: BLE001 - the signature table is still a useful answer
                res["semantic_error"] = f"{type(exc).__name__}: {exc}"
        return res


# ---- the text a model reads --------------------------------------------------------------------------------------------
def format_describe(d: dict) -> str:
    i, s = d["info"], d["sig"]
    part = ""
    if d["truncated"] or d["start"] > 0 or d["end"] < i["duration_s"] - 0.01:
        part = f'  [analysed {d["start"]:.2f}-{d["end"]:.2f} s]'
    shape = "noise-like" if s["flatness"] >= 0.2 else "tonal" if s["flatness"] < 0.05 else "mixed"
    decay = f'{s["decay_ms"]} ms' if s["decay_found"] else f'{s["decay_ms"]} ms+ (still audible at the end)'
    lines = [
        f'{d["id"]}  {Path(d["path"]).name}  {i["duration_s"]:.2f} s, {i["sr_hz"]} Hz, {i["channels"]} ch{part}',
        f'level: rms {s["rms_dbfs"]} dBFS, peak {s["peak_dbfs"]} dBFS | attack {s["attack_ms"]} ms, decay {decay} | '
        f'flatness {s["flatness"]} ({shape})',
        f'spectrum: dominant {s["dom_hz"]:.0f} Hz, centroid {s["centroid_hz"]:.0f} Hz, 95% of the energy below {s["rolloff95_hz"]:.0f} Hz',
        'bands: <500 Hz {:.0%} | 500-2k {:.0%} | 2k-8k {:.0%} | >8k {:.0%}   (the optional embedding model only hears up to 8 kHz)'.format(
            s["band_sub500"], s["band_500_2k"], s["band_2k_8k"], s["band_8k_plus"]),
    ]
    if i["sr_hz"] // 2 < 16000:  # a normal 44.1 / 48 kHz file needs no remark; a telephone-quality one does
        lines.append(f'note: the source is {i["sr_hz"]} Hz, so it holds nothing above {i["sr_hz"] // 2} Hz')
    elif i["sr_hz"] > C.ANALYSIS_SR:
        lines.append(f'note: the source is {i["sr_hz"]} Hz; only the part below {C.ANALYSIS_SR // 2} Hz is analysed')
    if d["png"]:
        lines.append(f'spectrogram: {d["png"]}   (open it as an image: waveform on top, 0-{C.ANALYSIS_SR // 2000} kHz below, '
                     f'dotted line = 8 kHz)')
    if d["truncated"]:
        lines.append(f'NOTE: only {C.MAX_AUDIO_SECONDS:g} s were analysed; pass start_s/end_s for other parts.')
    return "\n".join(lines)


def format_compare(res: dict) -> str:
    lines = ["A   " + format_describe(res["a"]).splitlines()[0]]
    for k, it in enumerate(res["items"], 1):
        lines.append(f"B{k}  " + format_describe(it["b"]).splitlines()[0])
        if it["same_id"]:
            lines.append("    identical file contents (same ID)")
        diffs = [r for r in it["rows"] if r["notable"]]
        lines.append("    " + ("notable differences: " + "; ".join(f'{r["label"]} {r["a"]} -> {r["b"]}' for r in diffs)
                               if diffs else "no notable differences in the signature"))
        if it["cos"] is not None:
            lines.append(f'    semantic similarity {it["cos"]:.3f}  (in a 30-clip sound-effect test the same kind of sound '
                         f'scored about 0.94, different kinds about 0.88; only the first {C.SEMANTIC_MAX_SECONDS} s of each file is embedded)')
    if len(res["items"]) == 1:
        lines.append("    full table: " + " | ".join(f'{r["label"]} {r["a"]} vs {r["b"]}' for r in res["items"][0]["rows"]))
    if res.get("semantic_error"):
        lines.append(f'semantic comparison unavailable: {res["semantic_error"]}')
    if res["stacked_png"]:
        lines.append(f'spectrograms (A on top, B below, same frequency scale): {res["stacked_png"]}   '
                     f'(open it as an image; dotted line = 8 kHz)')
    return "\n".join(lines)
