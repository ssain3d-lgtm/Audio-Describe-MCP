from audio_describe_mcp import cli
from conftest import needs_ffmpeg
from helpers import tone, write_wav

pytestmark = needs_ffmpeg


def test_describe_prints_the_id_and_the_picture_path(tmp_path, capsys):
    p = write_wav(tmp_path / "a.wav", tone(6000, 0.5))
    assert cli.main(["describe", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("a-") and "bands:" in out and "spectrogram:" in out


def test_compare_two_files(tmp_path, capsys):
    a, b = write_wav(tmp_path / "a.wav", tone(6000, 0.5)), write_wav(tmp_path / "b.wav", tone(12000, 0.5))
    assert cli.main(["compare", str(a), str(b)]) == 0
    assert ">8k 0% -> 100%" in capsys.readouterr().out


def test_a_bad_path_is_an_error_message_and_exit_code_1(tmp_path, capsys):
    assert cli.main(["describe", str(tmp_path / "missing.wav")]) == 1
    assert "not found" in capsys.readouterr().err
