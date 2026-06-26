"""
rtvoice → Sonos output (proof of concept)
=========================================

Routes the assistant's voice to a Sonos speaker instead of the local
sound card.

How it works
------------
A Sonos speaker is a *pull* device: it does not accept pushed PCM frames,
it streams media from an HTTP URI that you hand it over UPnP. So this
adapter does two things:

1. It implements rtvoice's ``AudioOutputDevice`` port. Every PCM chunk the
   realtime model produces is dropped into a thread-safe queue.
2. It runs a tiny HTTP server that serves one endless ``audio/wav`` stream.
   The Sonos connects once, and we feed it the queued PCM. When the queue
   runs dry between turns we emit silence, so the stream never ends and the
   speaker keeps the connection (and the "now playing" state) alive.

On ``start()`` the adapter discovers the speaker by room name, points it at
``http://<this-machine>:<port>/rtvoice.wav`` and presses play.

Caveats (it's a PoC)
--------------------
* Sonos buffers ~1-2 s internally, so there is audible latency and barge-in
  (interruption) is not crisp: ``clear_buffer()`` drops what *we* have queued,
  but audio already inside the speaker still plays out.
* This machine and the speaker must be on the same LAN, and the firewall must
  allow inbound connections on the chosen port.

Running
-------
::

    OPENAI_API_KEY=sk-...  python examples/sonosify_output.py --room "Living Room"

Install the optional dependency first: ``uv sync --group sonos``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import socket
import struct
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dotenv import load_dotenv
from sonosify import SonosClient, SonosController

from rtvoice import RealtimeAgent
from rtvoice.audio import AudioOutputDevice

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sonos_output")

# The realtime API emits mono signed-16-bit PCM at 24 kHz.
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2
STREAM_PATH = "/rtvoice.wav"


def _streaming_wav_header() -> bytes:
    """A RIFF/WAVE header for an open-ended PCM stream.

    The chunk sizes are set to the 32-bit max: we never know the final length
    because the stream only ends when the agent stops, and players that honour
    the size field must not truncate us.
    """
    byte_rate = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH
    block_align = CHANNELS * SAMPLE_WIDTH
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,  # RIFF chunk size (unknown)
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,  # PCM
        CHANNELS,
        SAMPLE_RATE,
        byte_rate,
        block_align,
        SAMPLE_WIDTH * 8,  # bits per sample
        b"data",
        0xFFFFFFFF,  # data chunk size (unknown)
    )


def _local_ip_towards(target_ip: str) -> str:
    """The IP of the interface this machine would use to reach ``target_ip``.

    Picking the right interface matters on machines with several (VPN, WSL,
    docker): the Sonos has to be able to call us back on the address we hand it.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((target_ip, 1400))
        return sock.getsockname()[0]
    finally:
        sock.close()


class _StreamHandler(BaseHTTPRequestHandler):
    """Serves the single endless WAV stream for one connected Sonos."""

    def __init__(self, *args, output: SonosOutput, **kwargs):
        self._output = output
        super().__init__(*args, **kwargs)

    def log_message(self, *_args) -> None:  # silence default stderr logging
        pass

    def do_GET(self) -> None:
        if self.path != STREAM_PATH:
            self.send_error(404)
            return

        logger.info("Sonos connected to audio stream from %s", self.client_address[0])
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Connection", "close")
        self.end_headers()

        # Sonos opens several short-lived probe connections before settling on
        # one. Only the newest gets to stream; older pump loops bow out.
        generation = self._output.begin_stream()
        try:
            self.wfile.write(_streaming_wav_header())
            self._output.pump_to(self.wfile, generation)
        except OSError:
            # Broken pipe / connection reset / aborted (WinError 10053) - the
            # Sonos dropped this connection. Expected for probe connections.
            logger.info("Sonos disconnected from audio stream")


class SonosOutput(AudioOutputDevice):
    """An ``AudioOutputDevice`` that plays through a Sonos speaker.

    Parameters
    ----------
    room:
        Sonos room/zone name (e.g. ``"Living Room"``). ``None`` lets sonosify
        pick a coordinator automatically.
    ip:
        Skip discovery and target a speaker directly by IP.
    volume:
        Optional 0-100 volume to set once playback starts.
    http_port:
        Port for the local audio stream server. ``0`` picks a free one.
    """

    # ~20 ms of silence, used to keep the stream alive between turns.
    _SILENCE = b"\x00" * (SAMPLE_RATE // 50 * CHANNELS * SAMPLE_WIDTH)

    def __init__(
        self,
        *,
        room: str | None = None,
        ip: str | None = None,
        volume: int | None = None,
        http_port: int = 0,
    ):
        self._room = room
        self._ip = ip
        self._volume = volume
        self._http_port = http_port

        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._controller: SonosController | None = None
        self._client: SonosClient | None = None
        self._active = False

        # Identifies the connection currently allowed to stream. Sonos opens
        # multiple connections; without this they would all drain the same
        # queue and tear the audio apart.
        self._generation = 0
        self._generation_lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        return not self._queue.empty()

    async def start(self) -> None:
        if self._active:
            return

        self._start_http_server()
        assert self._server is not None
        client = await self._connect_speaker()

        host = _local_ip_towards(client.ip)
        port = self._server.server_address[1]
        stream_url = f"http://{host}:{port}{STREAM_PATH}"

        # Mark active before play_uri so the pump loop streams (silence) the
        # moment the Sonos connects, instead of closing the connection.
        self._active = True

        if self._volume is not None:
            await client.set_volume(self._volume)

        logger.info("Pointing Sonos at %s", stream_url)
        await client.play_uri(stream_url, title="rtvoice")

        self._client = client

    async def stop(self) -> None:
        if not self._active:
            return
        self._active = False

        self._queue.put(None)  # release a blocked pump loop

        if self._client is not None:
            try:
                await self._client.stop()
            finally:
                await self._client.close()
            self._client = None

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None

        self._controller = None

    async def play_chunk(self, chunk: bytes) -> None:
        if self._active:
            self._queue.put(chunk)

    async def clear_buffer(self) -> None:
        """Drop everything we have queued (barge-in).

        Note: audio already buffered inside the Sonos still plays out - this
        only clears what hasn't been streamed yet.
        """
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            logger.debug("Cleared %d queued chunks", cleared)

    def begin_stream(self) -> int:
        """Claim the active stream slot, superseding any older connection."""
        with self._generation_lock:
            self._generation += 1
            return self._generation

    def pump_to(self, wfile, generation: int) -> None:
        """Drain the PCM queue into the HTTP response, padding with silence.

        Runs on the HTTP server thread. TCP back-pressure from the Sonos paces
        real audio; when no audio is queued we emit short silence frames so the
        stream stays open between turns. The loop exits as soon as a newer
        connection claims the stream, so only one connection feeds the queue.
        """
        while self._active and generation == self._generation:
            try:
                chunk = self._queue.get(timeout=0.02)
            except queue.Empty:
                chunk = self._SILENCE
            if chunk is None:  # stop sentinel
                break
            wfile.write(chunk)

    def _start_http_server(self) -> None:
        handler = partial(_StreamHandler, output=self)
        self._server = ThreadingHTTPServer(("0.0.0.0", self._http_port), handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._server_thread.start()

    async def _connect_speaker(self) -> SonosClient:
        if self._ip is not None:
            logger.info("Using Sonos speaker at %s", self._ip)
            return SonosClient(self._ip)

        logger.info(
            "Discovering Sonos speaker%s...",
            f" '{self._room}'" if self._room else "",
        )
        self._controller = SonosController()
        client = await self._controller.client(self._room)
        logger.info("Found speaker at %s", client.ip)
        return client


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play a rtvoice agent on a Sonos speaker"
    )
    parser.add_argument("--room", help="Sonos room name, e.g. 'Living Room'")
    parser.add_argument("--ip", help="Target a Sonos speaker by IP (skips discovery)")
    parser.add_argument("--volume", type=int, help="Set speaker volume (0-100)")
    args = parser.parse_args()

    agent = RealtimeAgent(
        instructions=(
            "You are Jet, a calm and friendly voice assistant. "
            "Keep replies short and natural - you're speaking out loud."
        ),
        audio_output=SonosOutput(room=args.room, ip=args.ip, volume=args.volume),
    )

    print("🔊  Talk to your assistant - it answers on the Sonos speaker.\n")
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
