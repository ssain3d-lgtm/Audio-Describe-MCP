import sys

import numpy as np
import pytest

from audio_describe_mcp import config as C
from audio_describe_mcp import semantic as M


class FakeST:
    """Stands in for sentence_transformers.SentenceTransformer: records how it was configured, never loads weights."""
    made: list = []

    def __init__(self, model_id, device=None, model_kwargs=None, config_kwargs=None):
        import torch

        self.model_id, self.device, self.config_kwargs = model_id, device, dict(config_kwargs or {})
        self.encoded = []
        names = ["text.layer.weight", "audio_tower.conv.weight"]
        self._params = [(n, torch.nn.Parameter(torch.zeros(2), requires_grad=False)) for n in names]
        FakeST.made.append(self)

    def eval(self):
        return self

    def named_parameters(self):
        return iter(self._params)

    def parameters(self):
        return iter(p for _, p in self._params)

    def encode(self, x, **kw):
        self.encoded.append((x, kw))
        v = np.zeros(C.EMBED_DIM, np.float32)
        v[0] = 1.0
        return v


@pytest.fixture()
def enc(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    FakeST.made = []
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeST)
    monkeypatch.setattr(M, "release_memory", lambda: released.append(1))
    released.clear()
    return M.SemanticEncoder(threads=1, idle_seconds=100)


released: list = []
CLIP = np.zeros(1600, np.float32)


def test_the_model_is_built_on_the_cpu_without_the_vision_tower(enc):
    enc.embed([CLIP])
    made = FakeST.made[0]
    assert made.device == "cpu" and made.config_kwargs == {"vision_config": None} and made.model_id == "google/embeddinggemma-2"


def test_clips_are_sent_as_float32_16k_audio_and_come_back_unit_length(enc):
    v = enc.embed([CLIP, CLIP + 0.1])
    assert v.shape == (2, C.EMBED_DIM) and np.allclose(np.linalg.norm(v, axis=1), 1.0)
    x, kw = FakeST.made[0].encoded[0]
    assert x["audio"]["sampling_rate"] == C.MODEL_AUDIO_SR and x["audio"]["array"].dtype == np.float32
    assert kw["truncate_dim"] == C.EMBED_DIM and kw["normalize_embeddings"] is True


def test_the_model_is_loaded_once_and_an_empty_batch_loads_nothing(enc):
    assert enc.embed([]).shape == (0, C.EMBED_DIM) and not FakeST.made
    enc.embed([CLIP])
    enc.embed([CLIP])
    assert len(FakeST.made) == 1


def test_loading_stamps_the_use_time_so_the_idle_watcher_cannot_drop_a_model_that_is_about_to_be_used(enc):
    enc.embed([CLIP])
    assert enc.last_use > 0


def test_an_idle_model_is_released_and_a_busy_one_is_kept(enc):
    enc.embed([CLIP])
    t = enc.last_use
    assert enc.release_if_idle(now=t + 50) is False and enc.loaded
    assert enc.release_if_idle(now=t + 101) is True and not enc.loaded and released == [1]
    assert enc.release_if_idle(now=t + 500) is False, "nothing left to release"
    enc.embed([CLIP])
    assert enc.loaded and len(FakeST.made) == 2, "used again after release: it loads again"


def test_unload_hands_memory_back_only_when_there_was_a_model(enc):
    enc.unload()
    assert released == []
    enc.embed([CLIP])
    enc.unload()
    assert released == [1] and not enc.loaded


def test_a_model_without_an_audio_encoder_is_refused(enc, monkeypatch):
    class NoAudio(FakeST):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._params = self._params[:1]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", NoAudio)
    with pytest.raises(M.SemanticError, match="no audio encoder"):
        enc.embed([CLIP])


def test_without_the_extra_installed_the_error_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(M.SemanticError, match=r"audio-describe-mcp\[semantic\]"):
        M.SemanticEncoder().embed([CLIP])
