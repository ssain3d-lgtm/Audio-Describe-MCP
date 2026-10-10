import os

import numpy as np
from PIL import Image

from audio_describe_mcp import spectrogram as G
from helpers import SR, tone


def _brightness(img, meta, f_lo, f_hi):
    y_hi, y_lo = G.spec_y(f_hi, meta["fmax"]), G.spec_y(f_lo, meta["fmax"])
    x0, _, x1, _ = meta["plot_box"]
    return float(np.asarray(img.convert("L"))[y_hi:y_lo, x0 + 10:x1 - 10].mean())


def test_a_12k_tone_shows_above_the_8k_line_and_a_6k_tone_below():
    hi, mhi = G.spectrogram_image(tone(12000), SR, "12k")
    lo, mlo = G.spectrogram_image(tone(6000), SR, "6k")
    assert mhi["fmax"] == 24000 and mhi["y_8k"] == G.spec_y(8000, 24000)
    assert _brightness(hi, mhi, 11500, 12500) > 5 * _brightness(lo, mlo, 11500, 12500) + 10
    assert _brightness(lo, mlo, 5500, 6500) > 5 * _brightness(hi, mhi, 5500, 6500) + 10


def test_the_picture_has_the_documented_size_and_survives_tiny_and_empty_input():
    img, _ = G.spectrogram_image(tone(1000, 0.005), SR, "tiny")
    assert img.size == (G.LEFT + G.PLOT_W + G.RIGHT, G.TOP + G.WAVE_H + G.GAP + G.SPEC_H + G.BOTTOM)
    empty, meta = G.spectrogram_image(np.zeros(0, np.float32), SR, "empty")
    assert empty.size == img.size
    assert _brightness(empty, meta, 1000, 5000) < 5, "silence is drawn dark, not bright"


def test_dithered_digital_silence_is_drawn_dark_not_as_a_wall_of_noise():
    dither = np.random.default_rng(4).integers(-1, 2, SR).astype(np.float32) / 32768  # +-1 LSB of a 16-bit file, about -92 dBFS
    img, meta = G.spectrogram_image(dither, SR, "dither")
    assert _brightness(img, meta, 1000, 5000) < 5 and _brightness(img, meta, 10000, 20000) < 5


def test_a_quiet_tone_is_still_visible():
    img, meta = G.spectrogram_image(tone(1000, amp=0.001), SR, "-60 dBFS")
    assert _brightness(img, meta, 800, 1200) > 15


def test_a_16k_source_has_no_8k_line_beyond_its_range():
    _, meta = G.spectrogram_image(tone(1000, sr=16000), 16000)
    assert meta["fmax"] == 8000 and meta["y_8k"] is None


def test_write_makes_a_png_and_prune_removes_the_oldest_first(tmp_path):
    out = G.write_spectrogram(tone(1000), SR, tmp_path / "s" / "a.png", "t")
    assert out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    for i in range(3):
        p = G.write_spectrogram(tone(1000 + i), SR, tmp_path / "s" / f"b{i}.png")
        os.utime(p, (1000 + i, 1000 + i))
    size = sum(p.stat().st_size for p in (tmp_path / "s").glob("*.png"))
    assert G.prune_dir(tmp_path / "s", size - 1) >= 1
    assert not (tmp_path / "s" / "b0.png").exists(), "the oldest file goes first"


def test_the_temporary_file_of_a_write_in_flight_survives_a_prune_by_another_session(tmp_path, monkeypatch):
    folder = tmp_path / "s"
    seen = {}
    real_replace = os.replace

    def spying_replace(src, dst):
        G.prune_dir(folder, 0)  # another session prunes the folder at exactly this moment
        seen["survived"] = os.path.exists(src)
        return real_replace(src, dst)

    monkeypatch.setattr(G.os, "replace", spying_replace)
    G.write_spectrogram(tone(1000), SR, folder / "a.png")
    assert seen["survived"] and (folder / "a.png").exists()


def test_stacked_png_is_two_spectrograms_tall(tmp_path):
    out = G.stacked_png(tone(6000), tone(12000), SR, tmp_path / "ab.png", "A", "B")
    one, _ = G.spectrogram_image(tone(6000), SR)
    with Image.open(out) as im:
        assert im.size == (one.width, one.height * 2)
