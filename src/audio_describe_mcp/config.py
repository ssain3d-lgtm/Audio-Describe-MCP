"""Constants and the few environment variables.  Everything has a default; nothing is required.

Environment variables
  AUDIO_DESCRIBE_FFMPEG          path to ffmpeg (default: the one on PATH)
  AUDIO_DESCRIBE_CACHE           folder for the spectrogram PNGs (default: a per-user cache folder)
  AUDIO_DESCRIBE_MODEL           Hugging Face id of the optional embedding model (default google/embeddinggemma-2)
  AUDIO_DESCRIBE_IDLE_SECONDS    release the embedding model after this much disuse (default 120)
  AUDIO_DESCRIBE_STDERR          "inherit" keeps the server's stderr; by default it goes to server.log next to the cache
"""
from __future__ import annotations

import os
from pathlib import Path

ANALYSIS_SR = 48000  # ffmpeg resamples to this for the signature; a source below it simply has nothing above its own Nyquist
MODEL_AUDIO_SR = 16000  # what the optional audio encoder takes (mono)
MAX_AUDIO_SECONDS = 300  # one describe / compare call analyses at most this much of a file
SEMANTIC_MAX_SECONDS = 30  # the embedding model sees at most this much of a file
MAX_COMPARE_CANDIDATES = 20
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma")
SPECTRO_MAX_BYTES = 300 * 2**20  # the oldest spectrogram PNGs are deleted beyond this
AUDIO_MODEL_MIN_FREE_GB = 3.5  # refuse to load the embedding model with less free RAM than this (PyTorch + the model take about 2.9 GB)
EMBED_DIM = 512  # Matryoshka-truncated, L2-normalised
SEMANTIC_MODEL_ID = os.environ.get("AUDIO_DESCRIBE_MODEL", "google/embeddinggemma-2")
AUDIO_IDLE_SECONDS = float(os.environ.get("AUDIO_DESCRIBE_IDLE_SECONDS", "120"))


def cache_dir() -> Path:
    """Where spectrogram PNGs are written (read at call time, so tests and users can change the environment)."""
    env = os.environ.get("AUDIO_DESCRIBE_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "audio-describe-mcp" / "spectro"
