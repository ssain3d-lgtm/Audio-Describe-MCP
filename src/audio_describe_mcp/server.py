"""MCP stdio server with two tools: audio_describe and audio_compare.

Some MCP clients (Qwen Code, measured) never drain a server's stderr pipe, and a chatty server wedges once it fills.  So unless
AUDIO_DESCRIBE_STDERR=inherit, stderr is pointed at server.log (next to the spectrogram cache) before anything else runs.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Annotated

from pydantic import Field

from . import config as C


def _redirect_stderr() -> None:
    if os.environ.get("AUDIO_DESCRIBE_STDERR", "log") == "inherit":
        return
    try:
        folder = C.cache_dir().parent
        folder.mkdir(parents=True, exist_ok=True)
        log = open(folder / "server.log", "a", buffering=1, encoding="utf-8", errors="replace")
        os.dup2(log.fileno(), 2)
        sys.stderr = log
    except OSError:
        pass  # keep the inherited stderr rather than failing to start


def build_server():
    from mcp.server.mcpserver import MCPServer
    from mcp_types import ToolAnnotations

    from .core import Describer, format_compare, format_describe

    describer = Describer()
    mcp = MCPServer(
        "audio-describe",
        instructions=("You cannot hear. audio_describe / audio_compare turn a sound file into a stable ID, numbers and a "
                      "spectrogram image you can read. Open the PNG path they return as an image."),
        log_level="WARNING",
    )
    ro = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    @mcp.tool(annotations=ro)
    async def audio_describe(path: Annotated[str, Field(description="absolute path of an audio file (wav mp3 flac ogg m4a opus); it does not need to be indexed")],
                             start_s: Annotated[float, Field(ge=0, description="start of the part to analyse, in seconds")] = 0.0,
                             end_s: Annotated[float | None, Field(description="end of the part, in seconds (default: end of file; at most 300 s per call)")] = None,
                             spectrogram: Annotated[bool, Field(description="also draw a spectrogram PNG that you can open as an image")] = True) -> str:
        """You cannot hear; this is how you tell sounds apart. Returns a stable ID (identical copies share it), numbers
        (levels, dominant/centroid frequency, energy per band including above 8 kHz, attack, decay, noise-vs-tone) and the
        path of a spectrogram PNG (waveform on top, frequency below, dotted line = 8 kHz) to open as an image. Works on any
        audio file, no indexing."""
        t0 = time.time()

        def run():
            try:
                return format_describe(describer.describe(path, start_s, end_s, spectrogram))
            except (ValueError, OSError) as exc:  # a problem with the file is an answer, not a crash
                return f"audio_describe: {exc}"

        out = await asyncio.to_thread(run)
        print(f"audio_describe took {(time.time() - t0) * 1000:.0f} ms", file=sys.stderr)
        return out

    @mcp.tool(annotations=ro)
    async def audio_compare(a: Annotated[str, Field(description="absolute path of the reference audio file")],
                            b: Annotated[str | list[str], Field(description="another audio file, or a list of up to 20 candidates")],
                            semantic: Annotated[bool, Field(description="also compare with the embedding model: is it the same KIND of sound (needs the [semantic] extra; loads a ~1.5 GB model, about 3 GB of RAM in total, for a few minutes)")] = False,
                            spectrogram: Annotated[bool, Field(description="with exactly one b: one PNG with both spectrograms stacked")] = True) -> str:
        """Tell how two sounds differ, or which candidate is closest to a reference: same-ID check (identical file), a list
        of what differs (length, level, frequency, energy per band, attack, decay) and, with semantic=true, an embedding
        similarity (several candidates are then ranked by it). With one candidate also a stacked spectrogram PNG to open as
        an image."""
        t0 = time.time()

        def run():
            try:
                return format_compare(describer.compare(a, b, semantic, spectrogram))
            except (ValueError, OSError) as exc:
                return f"audio_compare: {exc}"

        out = await asyncio.to_thread(run)
        print(f"audio_compare took {(time.time() - t0) * 1000:.0f} ms", file=sys.stderr)
        return out

    return mcp


def main() -> None:
    _redirect_stderr()
    build_server().run("stdio")


if __name__ == "__main__":
    main()
