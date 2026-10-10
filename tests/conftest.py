import pytest

from audio_describe_mcp import signature as S


@pytest.fixture(autouse=True)
def _private_cache(tmp_path, monkeypatch):
    """Every test writes its spectrograms into its own temporary cache folder."""
    monkeypatch.setenv("AUDIO_DESCRIBE_CACHE", str(tmp_path / "cache"))


needs_ffmpeg = pytest.mark.skipif(S.find_ffmpeg() is None, reason="ffmpeg not available")
