import asyncio
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from rtvoice.audio import SonosOutput


class FakeHostedAudioClip:
    def __init__(self, clip_id: str, client: "FakeSonosClient") -> None:
        self.id = clip_id
        self._client = client
        self.closed = False
        self._finished = asyncio.Event()

    def finish(self) -> None:
        self._finished.set()

    async def wait_until_finished(self) -> object:
        await self._finished.wait()
        return SimpleNamespace(status="DONE")

    async def cancel(self) -> None:
        await self._client.cancel_audio_clip(self.id)

    def close(self) -> None:
        self.closed = True


class FakeSonosClient:
    instances: ClassVar[list["FakeSonosClient"]] = []

    def __init__(self, ip_address: str) -> None:
        self.ip_address = ip_address
        self.played: list[tuple[bytes, dict[str, object]]] = []
        self.clips: list[FakeHostedAudioClip] = []
        self.cancelled: list[str] = []
        self.closed = False
        self.instances.append(self)

    async def play_audio_clip_data(
        self, audio: bytes, **options: object
    ) -> FakeHostedAudioClip:
        self.played.append((audio, options))
        clip = FakeHostedAudioClip(f"clip-{len(self.played)}", self)
        self.clips.append(clip)
        return clip

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
async def test_uses_environment_and_hosts_wav_via_sonosify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SONOS_IP_ADDRESS", "192.0.2.10")
    monkeypatch.setenv("SONOS_SPEAKER_NAME", "Kitchen")
    output = SonosOutput(advertised_host="127.0.0.1")

    await output.start()
    await output.play_chunk(b"\x00\x00" * 240)
    await output.finish_response()

    client = FakeSonosClient.instances[0]
    audio, options = client.played[0]
    assert client.ip_address == "192.0.2.10"
    assert options["name"] == "rtvoice: Kitchen"
    assert options["content_type"] == "audio/wav"
    assert options["local_host"] == "127.0.0.1"
    assert audio.startswith(b"RIFF")
    assert output.is_playing

    await output.clear_buffer()
    assert client.cancelled == ["clip-1"]
    assert not output.is_playing
    await output.stop()


@pytest.mark.asyncio
async def test_stops_playing_once_sonosify_reports_finished() -> None:
    output = SonosOutput("192.0.2.10")
    await output.start()
    await output.play_chunk(b"\x01\x02" * 240)
    await output.finish_response()
    assert output.is_playing

    client = FakeSonosClient.instances[0]
    client.clips[0].finish()
    await output._playback_task

    assert not output.is_playing
    await output.stop()


@pytest.mark.asyncio
async def test_requires_ip_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SONOS_IP_ADDRESS", raising=False)

    with pytest.raises(ValueError, match="SONOS_IP_ADDRESS"):
        await SonosOutput().start()
