import asyncio

from rtvoice.shared.logging import LoggingMixin


class ConversationRecorder(LoggingMixin):
    def __init__(self, output_file: str):
        self.output_file = output_file
        self._process: asyncio.subprocess.Process | None = None

    async def start_recording(
        self, sample_rate: int = 24000, channels: int = 1
    ) -> None:
        cmd = [
            "ffmpeg",
            "-f",
            "s16le",  # PCM signed 16-bit little-endian
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",  # Read from stdin
            "-acodec",
            "libmp3lame",
            "-b:a",
            "128k",
            self.output_file,
        ]

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.logger.info("Started conversation recording to %s", self.output_file)

    async def write_audio_chunk(self, audio_data: bytes) -> None:
        if self._process and self._process.stdin:
            self._process.stdin.write(audio_data)
            await self._process.stdin.drain()

    async def stop_recording(self) -> None:
        if self._process and self._process.stdin:
            self._process.stdin.close()
            await self._process.wait()
            self.logger.info("Conversation recording saved to %s", self.output_file)
