"""Start the real server over stdio exactly like an MCP client does, list the tools and call them."""
import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from conftest import needs_ffmpeg
from helpers import tone, write_wav

pytestmark = needs_ffmpeg


def text_of(result) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content)


async def session_run(tmp_path, body):
    env = dict(os.environ, AUDIO_DESCRIBE_CACHE=str(tmp_path / "cache"), PYTHONUTF8="1")
    params = StdioServerParameters(command=sys.executable, args=["-m", "audio_describe_mcp.server"], env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            return await body(s)


def test_the_two_tools_are_listed_and_work_over_stdio(tmp_path):
    a, b = write_wav(tmp_path / "six k.wav", tone(6000, 0.5)), write_wav(tmp_path / "twelve.wav", tone(12000, 0.5))

    async def body(s):
        names = {t.name for t in (await s.list_tools()).tools}
        d = text_of(await s.call_tool("audio_describe", {"path": str(b)}))
        c = text_of(await s.call_tool("audio_compare", {"a": str(a), "b": str(b)}))
        bad = text_of(await s.call_tool("audio_describe", {"path": str(tmp_path / "nope.wav")}))
        return names, d, c, bad

    names, d, c, bad = asyncio.run(session_run(tmp_path, body))
    assert names == {"audio_describe", "audio_compare"}
    assert ">8k 100%" in d and Path(d.split("spectrogram: ")[1].split("   (")[0]).exists()
    assert ">8k 0% -> 100%" in c
    assert bad.startswith("audio_describe: ") and "not found" in bad, "a bad path is an answer, not a crash"
    assert (tmp_path / "server.log").parent.exists()
