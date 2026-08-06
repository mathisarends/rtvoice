import asyncio
import sys
from types import SimpleNamespace
from typing import ClassVar
from urllib.parse import urlparse

import pytest

from rtvoice.audio import SonosOutput


class FakeSonosClient:
    instances: ClassVar[list["FakeSonosClient"]] = []

    def __init__(self, ip_address: str) -> None:
        self.ip_address = ip_address
        self.played: list[tuple[str, dict[str, object]]] = []
        self.cancelled: list[str] = []
        self.closed = False
        self.instances.append(self)

    async def play_audio_clip(self, url: str, **options: object) -> object:
        self.played.append((url, options))
        return SimpleNamespace(id="clip-1")

    async def cancel_audio_clip(self, clip_id: str) -> None:
        self.cancelled.append(clip_id)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_sonosify(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSonosClient.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "sonosify",
        SimpleNamespace(
            SonosClient=FakeSonosClient,
            ClipPriority=SimpleNamespace(HIGH="HIGH"),
        ),
    )


@pytest.mark.asyncio
async def test_uses_environment_and_schedules_complete_wav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SONOS_IP_ADDRESS", "192.0.2.10")
    monkeypatch.setenv("SONOS_SPEAKER_NAME", "Kitchen")
    output = SonosOutput(advertised_host="127.0.0.1")

    await output.start()
    await output.play_chunk(b"\x00\x00" * 240)
    await output.finish_response()

    client = FakeSonosClient.instances[0]
    url, options = client.played[0]
    assert client.ip_address == "192.0.2.10"
    assert options["name"] == "rtvoice: Kitchen"
    assert "clip_type" not in options
    assert url.startswith("http://127.0.0.1:")
    assert output.is_playing

    await output.clear_buffer()
    assert client.cancelled == ["clip-1"]
    assert not output.is_playing
    await output.stop()


@pytest.mark.asyncio
async def test_serves_wav_with_byte_ranges() -> None:
    output = SonosOutput("192.0.2.10", advertised_host="127.0.0.1")
    await output.start()
    await output.play_chunk(b"\x01\x02" * 240)
    await output.finish_response()
    url = FakeSonosClient.instances[0].played[0][0]
    parsed = urlparse(url)

    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    writer.write(f"GET {parsed.path} HTTP/1.1\r\nRange: bytes=0-3\r\n\r\n".encode())
    await writer.drain()
    response = await reader.read()

    assert response.startswith(b"HTTP/1.1 206 Partial Content")
    assert response.endswith(b"RIFF")
    await output.stop()


@pytest.mark.asyncio
async def test_requires_ip_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SONOS_IP_ADDRESS", raising=False)

    with pytest.raises(ValueError, match="SONOS_IP_ADDRESS"):
        await SonosOutput().start()
