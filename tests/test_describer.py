from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audio_describe_mcp import config as C
from audio_describe_mcp import signature as S
from audio_describe_mcp.core import Describer, format_compare, format_describe
from conftest import needs_ffmpeg
from helpers import lowpassed_noise, tone, write_wav

pytestmark = needs_ffmpeg


class FakeEncoder:
    """embed() returns unit vectors whose cosine with the first clip is cos[i]."""

    def __init__(self, cos=None, fail=None, loaded=False):
        self.cos, self.fail, self.calls, self.loaded = cos or [], fail, [], loaded

    def embed(self, clips):
        self.calls.append(len(clips))
        if self.fail:
            raise self.fail
        out = np.zeros((len(clips), C.EMBED_DIM), np.float32)
        out[0, 0] = 1.0
        for i in range(1, len(clips)):
            c = self.cos[i - 1]
            out[i, 0], out[i, 1] = c, float(np.sqrt(1 - c * c))
        return out


@pytest.fixture(autouse=True)
def plenty_of_ram(monkeypatch):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=16 * 2**30))  # the guard must not depend on this PC


@pytest.fixture()
def wavs(tmp_path):
    return SimpleNamespace(k6=write_wav(tmp_path / "six k.wav", tone(6000, 0.5)),
                           k12=write_wav(tmp_path / "twelve.wav", tone(12000, 0.5)), dir=tmp_path)


# ---- describe --------------------------------------------------------------------------------------------------------

def test_describe_gives_id_signature_and_a_png(wavs):
    d = Describer().describe(str(wavs.k6))
    assert d["id"] == S.audio_id(wavs.k6) and d["sig"]["band_2k_8k"] > 0.9
    assert Path(d["png"]).exists() and Path(d["png"]).name.startswith(d["id"])
    text = format_describe(d)
    assert d["id"] in text and "bands:" in text and "spectrogram:" in text and "up to 8 kHz" in text


def test_the_png_is_cached_by_id_and_window(wavs):
    desc = Describer()
    a = desc.describe(str(wavs.k6))
    mtime = Path(a["png"]).stat().st_mtime_ns
    b = desc.describe(str(wavs.k6))
    assert a["png"] == b["png"] and Path(b["png"]).stat().st_mtime_ns == mtime
    c = desc.describe(str(wavs.k6), start_s=0.1, end_s=0.3)
    assert c["png"] != a["png"] and abs(c["end"] - c["start"] - 0.2) < 1e-6


def test_the_picture_can_be_skipped(wavs):
    d = Describer().describe(str(wavs.k6), spectrogram=False)
    assert d["png"] is None and "spectrogram:" not in format_describe(d)


def test_a_missing_or_non_audio_path_is_a_readable_error(tmp_path):
    with pytest.raises(S.AudioError, match="not found"):
        Describer().describe(str(tmp_path / "nope.wav"))
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    with pytest.raises(S.AudioError, match="not an audio file"):
        Describer().describe(str(txt))


def test_a_telephone_quality_source_says_it_has_nothing_above_its_own_ceiling(tmp_path):
    p = write_wav(tmp_path / "phone.wav", tone(1000, 0.5, sr=16000), sr=16000)
    d = Describer().describe(str(p))
    text = format_describe(d)
    assert "16000 Hz" in text and "nothing above 8000 Hz" in text
    assert d["sig"]["band_8k_plus"] < 0.02


def test_an_ordinary_44k_source_gets_no_remark_and_a_96k_source_says_what_is_dropped(tmp_path):
    p44 = write_wav(tmp_path / "cd.wav", tone(1000, 0.5, sr=44100), sr=44100)
    assert "note:" not in format_describe(Describer().describe(str(p44)))
    p96 = write_wav(tmp_path / "hires.wav", tone(1000, 0.5, sr=96000), sr=96000)
    assert "only the part below 24000 Hz is analysed" in format_describe(Describer().describe(str(p96)))


def test_paths_with_spaces_and_korean_work(tmp_path):
    folder = tmp_path / "소리 test"
    folder.mkdir()
    p = write_wav(folder / "천둥 소리.wav", tone(300, 0.4))
    d = Describer().describe(str(p))
    assert d["sig"]["band_sub500"] > 0.9 and Path(d["png"]).exists()


def test_a_long_request_is_cut_to_the_cap_and_says_so(wavs, monkeypatch):
    monkeypatch.setattr(C, "MAX_AUDIO_SECONDS", 0.25)
    d = Describer().describe(str(wavs.k6))
    assert d["truncated"] and abs(d["end"] - 0.25) < 1e-6 and "only 0.25 s" in format_describe(d)


def test_a_rumble_is_described_as_noise_like_and_a_tone_as_tonal(tmp_path):
    rumble = write_wav(tmp_path / "rumble.wav", lowpassed_noise())
    assert "(noise-like)" in format_describe(Describer().describe(str(rumble), spectrogram=False))
    ping = write_wav(tmp_path / "ping.wav", tone(1000, 0.5))
    assert "(tonal)" in format_describe(Describer().describe(str(ping), spectrogram=False))


# ---- compare ---------------------------------------------------------------------------------------------------------

def test_compare_reports_the_difference_and_one_stacked_picture(wavs):
    res = Describer().compare(str(wavs.k6), str(wavs.k12))
    assert res["items"][0]["same_id"] is False and Path(res["stacked_png"]).exists()
    text = format_compare(res)
    assert "notable differences" in text and ">8k 0% -> 100%" in text and "full table" in text


def test_compare_with_an_identical_copy_says_so(wavs):
    copy = wavs.dir / "copy.wav"
    copy.write_bytes(Path(wavs.k6).read_bytes())
    res = Describer().compare(str(wavs.k6), str(copy))
    text = format_compare(res)
    assert res["items"][0]["same_id"] and "identical file contents" in text and "no notable differences" in text


def test_several_candidates_get_no_stacked_picture_and_are_capped(wavs):
    res = Describer().compare(str(wavs.k6), [str(wavs.k12), str(wavs.k6)])
    assert res["stacked_png"] is None and len(res["items"]) == 2
    with pytest.raises(ValueError):
        Describer().compare(str(wavs.k6), [str(wavs.k12)] * (C.MAX_COMPARE_CANDIDATES + 1))
    with pytest.raises(ValueError):
        Describer().compare(str(wavs.k6), [])


def test_semantic_ranks_candidates_by_cosine(wavs):
    enc = FakeEncoder(cos=[0.5, 0.9, 0.7])
    res = Describer(semantic=enc).compare(str(wavs.k6), [str(wavs.k12)] * 3, semantic=True)
    assert [it["cos"] for it in res["items"]] == [0.9, 0.7, 0.5] and enc.calls == [4]
    assert "semantic similarity 0.900" in format_compare(res)


def test_semantic_is_not_touched_unless_asked_for(wavs):
    enc = FakeEncoder(cos=[0.5])
    res = Describer(semantic=enc).compare(str(wavs.k6), str(wavs.k12))
    assert enc.calls == [] and res["items"][0]["cos"] is None and res["semantic_error"] is None


def test_a_failing_embedding_keeps_the_signature_answer(wavs):
    enc = FakeEncoder(fail=RuntimeError("model exploded"))
    res = Describer(semantic=enc).compare(str(wavs.k6), str(wavs.k12), semantic=True)
    assert "model exploded" in res["semantic_error"] and res["items"][0]["cos"] is None and res["items"][0]["rows"]
    assert "semantic comparison unavailable" in format_compare(res)


def test_semantic_is_refused_when_free_ram_is_short(wavs, monkeypatch):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=1 * 2**30))
    enc = FakeEncoder(cos=[0.5])
    res = Describer(semantic=enc).compare(str(wavs.k6), str(wavs.k12), semantic=True)
    assert "GB RAM free" in res["semantic_error"] and enc.calls == []
    enc.loaded = True  # an already loaded model needs no extra RAM
    assert Describer(semantic=enc).compare(str(wavs.k6), str(wavs.k12), semantic=True)["items"][0]["cos"] == 0.5
