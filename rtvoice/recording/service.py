import asyncio

from rtvoice.realtime.schemas import AudioFormat
from rtvoice.shared.logging import LoggingMixin


class AudioRecorder(LoggingMixin):
    def __init__(
        self,
        output_file: str,
        audio_format: AudioFormat = AudioFormat.PCM16,
        sample_rate: int = 24000,
        channels: int = 1,
    ):
        self._output_file = output_file
        self._audio_format = audio_format
        self._sample_rate = sample_rate
        self._channels = channels
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        format_params = self._get_format_params(self._audio_format)

        cmd = [
            "ffmpeg",
            "-f",
            format_params["format"],
            "-ar",
            str(self._sample_rate),
            "-ac",
            str(self._channels),
            "-i",
            "pipe:0",
        ]

        if "input_codec" in format_params:
            cmd.extend(["-acodec:0", format_params["input_codec"]])

        cmd.extend(["-acodec", "libmp3lame", "-b:a", "128k", self._output_file])

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.logger.info(
            "Started recording (format=%s, rate=%d) to %s",
            self._audio_format,
            self._sample_rate,
            self._output_file,
        )

    def _get_format_params(self, audio_format: AudioFormat) -> dict[str, str]:
        match audio_format:
            case AudioFormat.PCM16:
                return {"format": "s16le"}

            case AudioFormat.G711_ULAW:
                return {"format": "mulaw", "input_codec": "pcm_mulaw"}

            case AudioFormat.G711_ALAW:
                return {"format": "alaw", "input_codec": "pcm_alaw"}

            case _:
                raise ValueError(f"Unsupported audio format: {audio_format}")

    async def write_chunk(self, audio_data: bytes) -> None:
        if self._process and self._process.stdin:
            self._process.stdin.write(audio_data)
            await self._process.stdin.drain()

    async def stop(self) -> str:
        if self._process and self._process.stdin:
            self._process.stdin.close()
            await self._process.wait()
            self.logger.info("Recording saved to %s", self._output_file)

        return self._output_file
