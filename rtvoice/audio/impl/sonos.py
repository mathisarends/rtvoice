import asyncio
import io
import logging
import os
import re
import secrets
import socket
import wave
from contextlib import suppress
from typing import Any

from rtvoice.audio.ports import AudioOutput

logger = logging.getLogger(__name__)

_APP_ID = "io.github.mathisarends.rtvoice"
_BYTES_PER_SAMPLE = 2
_RANGE = re.compile(rb"bytes=(\d*)-(\d*)")

type ClipPath = str
type WavData = bytes


class SonosOutput(AudioOutput):
    def __init__(
        self,
        ip_address: str | None = None,
        speaker_name: str | None = None,
        *,
        volume: int | None = None,
        sample_rate: int = 24000,
        advertised_host: str | None = None,
    ) -> None:
        if volume is not None and not 0 <= volume <= 100:
            raise ValueError("volume must be between 0 and 100")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._ip_address = ip_address
        self._speaker_name = speaker_name
        self._volume = volume
        self._sample_rate = sample_rate
        self._advertised_host = advertised_host
        self._client: Any = None
        self._server: asyncio.Server | None = None
        self._routes: dict[ClipPath, WavData] = {}
        self._pcm = bytearray()
        self._active = False
        self._clip_id: str | None = None
        self._clip_path: str | None = None
        self._playback_task: asyncio.Task[None] | None = None

    @property
    def is_playing(self) -> bool:
        return bool(self._pcm) or self._clip_id is not None

    async def start(self) -> None:
        if self._active:
            return
        try:
            from sonosify import SonosClient
        except ImportError as exc:
            raise ImportError(
                "sonosify is required for SonosOutput. "
                "Install it with: pip install rtvoice[sonos]"
            ) from exc

        self._ip_address = self._ip_address or os.getenv("SONOS_IP_ADDRESS")
        self._speaker_name = self._speaker_name or os.getenv(
            "SONOS_SPEAKER_NAME", "Sonos"
        )
        if not self._ip_address:
            raise ValueError("SONOS_IP_ADDRESS is required for SonosOutput")
        self._client = SonosClient(self._ip_address)
        self._active = True

    async def stop(self) -> None:
        if not self._active:
            return
        await self.clear_buffer()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await self._client.close()
        self._client = None
        self._active = False

    async def play_chunk(self, chunk: bytes) -> None:
        if self._active:
            self._pcm.extend(chunk)

    async def finish_response(self) -> None:
        if not self._active or not self._pcm:
            return

        pcm = bytes(self._pcm)
        self._pcm.clear()
        await self._ensure_server()
        path = f"/{secrets.token_urlsafe(18)}.wav"
        self._routes[path] = _wav(pcm, self._sample_rate)
        self._clip_path = path

        from sonosify import ClipPriority

        try:
            clip = await self._client.play_audio_clip(
                self._clip_url(path),
                app_id=_APP_ID,
                name=f"rtvoice: {self._speaker_name}"[:64],
                volume=self._volume,
                priority=ClipPriority.HIGH,
            )
        except Exception:
            self._routes.pop(path, None)
            self._clip_path = None
            raise

        self._clip_id = clip.id
        duration = len(pcm) / (_BYTES_PER_SAMPLE * self._sample_rate)
        self._playback_task = asyncio.create_task(
            self._finish_playback(clip.id, path, duration)
        )

    async def clear_buffer(self) -> None:
        self._pcm.clear()
        task = self._playback_task
        self._playback_task = None
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if self._clip_id:
            clip_id, self._clip_id = self._clip_id, None
            try:
                await self._client.cancel_audio_clip(clip_id)
            except Exception:
                logger.exception("Could not cancel Sonos audio clip %s", clip_id)
        if self._clip_path:
            self._routes.pop(self._clip_path, None)
            self._clip_path = None

    async def _finish_playback(self, clip_id: str, path: str, duration: float) -> None:
        await asyncio.sleep(duration)
        if self._clip_id == clip_id:
            self._clip_id = None
        self._routes.pop(path, None)
        if self._clip_path == path:
            self._clip_path = None
        self._playback_task = None

    async def _ensure_server(self) -> None:
        if self._server:
            return
        self._server = await asyncio.start_server(self._serve_audio, "0.0.0.0", 0)

    def _clip_url(self, path: str) -> str:
        if not self._server or not self._server.sockets or not self._ip_address:
            raise RuntimeError("Sonos audio server is not running")
        host = self._advertised_host or _local_ip_for(self._ip_address)
        port = self._server.sockets[0].getsockname()[1]
        return f"http://{host}:{port}{path}"

    async def _serve_audio(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            request_line, *headers = request.split(b"\r\n")
            method, raw_path, _ = request_line.split(b" ", 2)
            data = self._routes.get(raw_path.decode("ascii", errors="ignore"))
            if data is None or method not in {b"GET", b"HEAD"}:
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            else:
                status, start, end = _requested_range(headers, len(data))
                body = data[start : end + 1]
                response_headers = [
                    f"HTTP/1.1 {status}\r\n",
                    "Content-Type: audio/wav\r\n",
                    "Accept-Ranges: bytes\r\n",
                    f"Content-Length: {len(body)}\r\n",
                    "Connection: close\r\n",
                ]
                if status.startswith("206"):
                    response_headers.append(
                        f"Content-Range: bytes {start}-{end}/{len(data)}\r\n"
                    )
                writer.write("".join(response_headers).encode() + b"\r\n")
                if method == b"GET":
                    writer.write(body)
            await writer.drain()
        except (asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(_BYTES_PER_SAMPLE)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


def _local_ip_for(remote_ip: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.connect((remote_ip, 1400))
        return str(connection.getsockname()[0])


def _requested_range(headers: list[bytes], size: int) -> tuple[str, int, int]:
    for header in headers:
        if not header.lower().startswith(b"range:"):
            continue
        match = _RANGE.search(header)
        if not match:
            break
        start = int(match.group(1) or 0)
        end = min(int(match.group(2) or size - 1), size - 1)
        if start <= end:
            return "206 Partial Content", start, end
    return "200 OK", 0, size - 1
