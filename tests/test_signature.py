import numpy as np
import pytest

from audio_describe_mcp import signature as S
from conftest import needs_ffmpeg
from helpers import SR, lowpassed_noise, tone, write_wav


# ---- signature -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("hz,band", [(200, "sub500"), (1000, "500_2k"), (6000, "2k_8k"), (12000, "8k_plus"), (20000, "8k_plus")])
def test_a_tone_lands_in_its_band_with_its_dominant_frequency(hz, band):
    s = S.signature(tone(hz), SR)
    assert s[f"band_{band}"] > 0.95
    assert abs(s["dom_hz"] - hz) < 25
    assert abs(sum(s[f"band_{n}"] for n, _, _ in S.BANDS) - 1.0) < 0.01
    assert s["flatness"] < 0.01, "a pure tone is not noise-like"


def test_levels_of_a_half_scale_sine():
    s = S.signature(tone(1000, amp=0.5), SR)
    assert abs(s["peak_dbfs"] - (-6.0)) < 0.2 and abs(s["rms_dbfs"] - (-9.0)) < 0.2


def test_noise_is_flat_and_puts_two_thirds_of_its_energy_above_8k():
    s = S.signature(np.random.default_rng(1).normal(0, 0.1, SR * 2), SR)
    assert s["flatness"] > 0.5
    assert s["band_8k_plus"] > 0.5 and s["band_2k_8k"] > 0.15


def test_rolloff_of_a_6k_tone_is_about_6k():
    assert 5800 < S.signature(tone(6000), SR)["rolloff95_hz"] < 6300


def test_silence_has_defined_numbers():
    s = S.signature(np.zeros(SR), SR)
    assert s["rms_dbfs"] == -120.0 and s["dom_hz"] == 0.0 and all(s[f"band_{n}"] == 0.0 for n, _, _ in S.BANDS)
    assert S.signature(np.zeros(0), SR)["peak_dbfs"] == -120.0


def test_attack_and_decay_of_a_decaying_burst():
    t = np.arange(int(1.5 * SR)) / SR
    x = np.where(t >= 0.1, np.sin(2 * np.pi * 1000 * t) * np.exp(-(t - 0.1) / 0.2), 0.0)
    s = S.signature(x, SR)
    assert abs(s["attack_ms"] - 100) <= 10
    assert s["decay_found"] and abs(s["decay_ms"] - 690) <= 40


def test_a_steady_sound_has_no_attack_even_though_its_envelope_wobbles():
    s = S.signature(tone(6000, 2.0), SR)
    assert s["attack_ms"] <= 10 and not s["decay_found"], "was: the loudest 5 ms frame, anywhere in the clip"
    n = S.signature(np.random.default_rng(7).normal(0, 0.1, SR * 2), SR)
    assert n["attack_ms"] <= 50 and not n["decay_found"]


def test_a_low_rumble_is_noise_like_even_though_everything_above_it_is_empty():
    assert S.signature(lowpassed_noise(), SR)["flatness"] > 0.3, "measured over the band the sound occupies, not 0-24 kHz"


def test_a_tone_plus_a_little_noise_is_still_tonal():
    assert S.signature(tone(1000) + np.random.default_rng(2).normal(0, 0.002, SR), SR)["flatness"] < 0.05


def test_a_clip_shorter_than_the_analysis_window_still_works():
    s = S.signature(tone(1000, seconds=0.01), SR)
    assert abs(s["dom_hz"] - 1000) < 200 and 0 < s["band_500_2k"] <= 1.0


# ---- window / id -----------------------------------------------------------------------------------------------------

def test_window_clamps_and_reports_truncation():
    assert S.window(10.0, 0, None, 300) == (0.0, 10.0, False)
    assert S.window(10.0, 2, 99, 300) == (2.0, 10.0, False)
    assert S.window(1000.0, 0, None, 300) == (0.0, 300.0, True)
    assert S.window(0.0, 0, None, 300) == (0.0, 300.0, True), "an unknown duration is capped, not refused"
    with pytest.raises(S.AudioError):
        S.window(10.0, 12, None, 300)


def test_an_explicit_end_of_zero_is_an_empty_window_not_the_whole_file():
    with pytest.raises(S.AudioError):
        S.window(10.0, 0, 0, 300)


def test_audio_id_is_stable_for_copies_and_differs_for_different_content(tmp_path):
    a = write_wav(tmp_path / "a.wav", tone(440))
    b = tmp_path / "copy.wav"
    b.write_bytes(a.read_bytes())
    c = write_wav(tmp_path / "c.wav", tone(880))
    assert S.audio_id(a) == S.audio_id(b) != S.audio_id(c)
    assert S.audio_id(a).startswith("a-") and len(S.audio_id(a)) == 12


# ---- ffmpeg ----------------------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_probe_reads_rate_channels_and_length(tmp_path):
    p = write_wav(tmp_path / "s.wav", tone(440, 1.5, sr=44100), sr=44100, channels=2)
    info = S.probe(p)
    assert info["sr_hz"] == 44100 and info["channels"] == 2 and abs(info["duration_s"] - 1.5) < 0.05


@needs_ffmpeg
def test_decode_returns_mono_48k_floats_and_honours_the_window(tmp_path):
    p = write_wav(tmp_path / "s.wav", tone(1000, 2.0, sr=44100), sr=44100)
    x = S.decode(p, 0.5, 1.0, SR)
    assert x.dtype == np.float32 and abs(len(x) - SR) < 400
    assert abs(S.signature(x, SR)["dom_hz"] - 1000) < 25


@needs_ffmpeg
def test_something_that_is_not_audio_raises_a_readable_error(tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"this is not audio")
    with pytest.raises(S.AudioError):
        S.probe(p)
    with pytest.raises(S.AudioError):
        S.decode(p, 0, 1)


def test_a_missing_ffmpeg_is_a_readable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "find_ffmpeg", lambda: None)
    with pytest.raises(S.AudioError, match="ffmpeg not found"):
        S.probe(tmp_path / "x.wav")
    with pytest.raises(S.AudioError, match="ffmpeg not found"):
        S.decode(tmp_path / "x.wav", 0, 1)


def test_the_ffmpeg_environment_variable_wins_over_path(tmp_path, monkeypatch):
    fake = tmp_path / "my-ffmpeg"
    fake.write_text("")
    monkeypatch.setenv("AUDIO_DESCRIBE_FFMPEG", str(fake))
    assert S.find_ffmpeg() == str(fake)
    monkeypatch.setenv("AUDIO_DESCRIBE_FFMPEG", str(tmp_path / "does-not-exist"))
    assert S.find_ffmpeg() != str(tmp_path / "does-not-exist")


# ---- comparison table ------------------------------------------------------------------------------------------------

def test_compare_rows_flag_what_differs():
    a, b = S.signature(tone(6000), SR), S.signature(tone(12000), SR)
    rows = {r["label"]: r for r in S.compare_rows(a, b)}
    assert rows[">8k"]["notable"] and rows["2k-8k"]["notable"] and rows["dominant"]["notable"]
    assert not rows["duration"]["notable"] and not rows["rms"]["notable"]
    assert rows[">8k"]["a"] == "0%" and rows[">8k"]["b"] == "100%"
    assert not any(r["notable"] for r in S.compare_rows(a, a))
