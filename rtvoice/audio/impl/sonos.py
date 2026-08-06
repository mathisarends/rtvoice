import asyncio
import io
import logging
import os
import wave
from contextlib import suppress
from typing import Any

from rtvoice.audio.ports import AudioOutput

logger = logging.getLogger(__name__)

_APP_ID = "io.github.mathisarends.rtvoice"
_BYTES_PER_SAMPLE = 2


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
        self._pcm = bytearray()
        self._active = False
        self._clip: Any = None
        self._playback_task: asyncio.Task[None] | None = None

    @property
    def is_playing(self) -> bool:
        return bool(self._pcm) or self._clip is not None

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

        from sonosify import ClipPriority

        self._clip = await self._client.play_audio_clip_data(
            _wav(pcm, self._sample_rate),
            content_type="audio/wav",
            local_host=self._advertised_host,
            app_id=_APP_ID,
            name=f"rtvoice: {self._speaker_name}"[:64],
            volume=self._volume,
            priority=ClipPriority.HIGH,
        )
        self._playback_task = asyncio.create_task(self._await_finish(self._clip))

    async def clear_buffer(self) -> None:
        self._pcm.clear()
        task = self._playback_task
        self._playback_task = None
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        clip, self._clip = self._clip, None
        if clip:
            try:
                await clip.cancel()
            except Exception:
                logger.exception("Could not cancel Sonos audio clip %s", clip.id)
            clip.close()

    async def _await_finish(self, clip: Any) -> None:
        with suppress(Exception):
            await clip.wait_until_finished()
        if self._clip is clip:
            self._clip = None
        self._playback_task = None


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(_BYTES_PER_SAMPLE)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()
