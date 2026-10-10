"""Optional: compare sounds with Google's EmbeddingGemma 2 (text+audio towers, CPU, FP32).

Needs the extra:  pip install "audio-describe-mcp[semantic]".  The first use downloads the model (about 1.5 GB) from Hugging Face.
The model reads 16 kHz audio, so it cannot hear anything above 8 kHz: that range is covered by the signature and the spectrogram.
The model is loaded on first use and released again after AUDIO_DESCRIBE_IDLE_SECONDS without use.
"""
from __future__ import annotations

import gc
import os
import threading
import time

import numpy as np

from . import config as C


class SemanticError(RuntimeError):
    """The optional embedding comparison could not be done (extra not installed, model could not be loaded, ...)."""


def release_memory() -> None:
    """Hand freed memory back to the OS.  Freed model weights stay in the process's working set (the allocator keeps them for
    reuse; measured on Windows: unloading left ~1.4 GB behind).  On Windows the working set is trimmed; pages that are still
    needed fault back in on demand.  Only an optimisation: never raises."""
    gc.collect()
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
    except Exception:  # noqa: BLE001
        pass


class SemanticEncoder:
    def __init__(self, model_id: str | None = None, threads: int | None = None, idle_seconds: float | None = None):
        self.model_id = model_id or C.SEMANTIC_MODEL_ID
        self.threads = threads or max(1, min(8, (os.cpu_count() or 4) // 2))
        self.idle_seconds = C.AUDIO_IDLE_SECONDS if idle_seconds is None else idle_seconds
        self._model = None
        self._lock = threading.Lock()
        self.last_use = 0.0
        self.load_seconds: float | None = None
        self._halt = threading.Event()
        self._watch: threading.Thread | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        """Caller holds the lock."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SemanticError("the semantic comparison needs the optional extra: pip install 'audio-describe-mcp[semantic]'") from exc
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        torch.set_num_threads(self.threads)
        t0 = time.time()
        # the vision tower is never built; FP16 is forbidden for this model (NaN), so FP32 on the CPU
        model = SentenceTransformer(self.model_id, device="cpu", model_kwargs={"torch_dtype": torch.float32},
                                    config_kwargs={"vision_config": None})
        model.eval()
        names = [n for n, _ in model.named_parameters()]
        if not any("audio" in n.lower() for n in names):
            raise SemanticError(f"{self.model_id} has no audio encoder (expected google/embeddinggemma-2)")
        if any("vision" in n.lower() for n in names):
            raise SemanticError("the vision tower was built although it was switched off")
        if {str(p.dtype) for p in model.parameters()} != {"torch.float32"}:
            raise SemanticError("the model was not loaded in float32")
        self.load_seconds = time.time() - t0
        return model

    def embed(self, clips: list[np.ndarray]) -> np.ndarray:
        """512-d unit vectors for mono float32 clips at C.MODEL_AUDIO_SR (16 kHz).  Loads the model if needed."""
        if not clips:
            return np.zeros((0, C.EMBED_DIM), dtype=np.float32)
        with self._lock:
            if self._model is None:
                self._model = self._load()  # raises SemanticError with the install hint when the extra is missing
            import torch  # only after _load() succeeded, so a missing extra never surfaces as a bare ModuleNotFoundError

            self.last_use = time.time()  # stamped before encoding: the idle watcher must not drop a model that is being used
            out = []
            with torch.inference_mode():
                for a in clips:
                    v = self._model.encode({"audio": {"array": np.asarray(a, dtype=np.float32), "sampling_rate": C.MODEL_AUDIO_SR}},
                                           truncate_dim=C.EMBED_DIM, normalize_embeddings=True, convert_to_numpy=True,
                                           show_progress_bar=False)
                    out.append(np.asarray(v, dtype=np.float32).reshape(-1))
            self.last_use = time.time()
        vecs = np.stack(out)
        if vecs.shape[1] != C.EMBED_DIM or not np.isfinite(vecs).all():
            raise SemanticError("the model returned a malformed embedding")
        if not np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3):
            raise SemanticError("the embeddings are not L2-normalised")
        return vecs

    def unload(self) -> None:
        with self._lock:
            had = self._model is not None
            self._model = None
        if had:
            release_memory()

    def release_if_idle(self, now: float | None = None) -> bool:
        """Unload the model if it has not been used for idle_seconds.  Returns True if it did."""
        now = time.time() if now is None else now
        with self._lock:
            if self._model is None or now - self.last_use <= self.idle_seconds:
                return False
            self._model = None
        release_memory()
        return True

    def start_idle_watch(self, interval: float = 15.0) -> threading.Thread:
        """A daemon thread that releases the model after idle_seconds without use."""
        if self._watch is not None and self._watch.is_alive():
            return self._watch

        def run():
            while not self._halt.wait(interval):
                try:
                    self.release_if_idle()
                except Exception:  # noqa: BLE001 - must never take the server down
                    pass

        self._watch = threading.Thread(target=run, name="audio-describe-idle", daemon=True)
        self._watch.start()
        return self._watch

    def stop_idle_watch(self) -> None:
        self._halt.set()
