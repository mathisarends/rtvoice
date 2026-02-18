import asyncio

from rtvoice.shared.logging import LoggingMixin


class AudioRecorder(LoggingMixin):
    def __init__(
        self,
        output_file: str,
        ffmpeg_format: str,
        sample_rate: int = 24000,
        channels: int = 1,
        input_codec: str | None = None,
    ):
        self._output_file = output_file
        self._ffmpeg_format = ffmpeg_format
        self._sample_rate = sample_rate
        self._channels = channels
        self._input_codec = input_codec
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        cmd = [
            "ffmpeg",
            "-f",
            self._ffmpeg_format,
            "-ar",
            str(self._sample_rate),
            "-ac",
            str(self._channels),
            "-i",
            "pipe:0",
        ]

        if self._input_codec:
            cmd.extend(["-acodec:0", self._input_codec])

        cmd.extend(["-acodec", "libmp3lame", "-b:a", "128k", self._output_file])

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.logger.info("Started recording to %s", self._output_file)

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
