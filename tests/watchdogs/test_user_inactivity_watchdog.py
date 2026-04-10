import asyncio

import pytest

from rtvoice.events.bus import EventBus
from rtvoice.events.views import (
    AudioPlaybackCompletedEvent,
    SubAgentFinishedEvent,
    SubAgentStartedEvent,
    UserInactivityCountdownEvent,
    UserInactivityTimeoutEvent,
)
from rtvoice.realtime.schemas import (
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeResponseObject,
    RealtimeServerEvent,
    ResponseCreatedEvent,
)
from rtvoice.watchdogs import UserInactivityTimeoutWatchdog


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


def make_speech_stopped() -> InputAudioBufferSpeechStoppedEvent:
    return InputAudioBufferSpeechStoppedEvent(
        event_id="evt_001",
        item_id="item_001",
        audio_end_ms=1000,
    )


def make_speech_started() -> InputAudioBufferSpeechStartedEvent:
    return InputAudioBufferSpeechStartedEvent(
        event_id="evt_002",
        item_id="item_001",
        audio_start_ms=2000,
    )


def make_response_created(response_id: str = "resp_001") -> ResponseCreatedEvent:
    return ResponseCreatedEvent(
        type=RealtimeServerEvent.RESPONSE_CREATED,
        event_id="evt_003",
        response=RealtimeResponseObject(id=response_id),
    )


class TestMonitoringStateTransitions:
    @pytest.mark.asyncio
    async def test_monitoring_starts_after_user_stops_speaking(
        self, event_bus: EventBus
    ) -> None:
        wt = UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=10.0)

        await event_bus.dispatch(make_speech_stopped())

        assert wt._is_monitoring is True

    @pytest.mark.asyncio
    async def test_monitoring_does_not_start_while_assistant_speaking(
        self, event_bus: EventBus
    ) -> None:
        wt = UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=10.0)

        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(make_speech_stopped())

        assert wt._is_monitoring is False

    @pytest.mark.asyncio
    async def test_monitoring_starts_after_both_user_and_assistant_finish(
        self, event_bus: EventBus
    ) -> None:
        wt = UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=10.0)

        await event_bus.dispatch(make_speech_stopped())
        await event_bus.dispatch(make_response_created())
        await event_bus.dispatch(AudioPlaybackCompletedEvent())

        assert wt._is_monitoring is True

    @pytest.mark.asyncio
    async def test_user_starts_speaking_stops_monitoring(
        self, event_bus: EventBus
    ) -> None:
        wt = UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=10.0)

        await event_bus.dispatch(make_speech_stopped())
        assert wt._is_monitoring is True

        await event_bus.dispatch(make_speech_started())
        assert wt._is_monitoring is False

    @pytest.mark.asyncio
    async def test_assistant_started_stops_monitoring(
        self, event_bus: EventBus
    ) -> None:
        wt = UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=10.0)

        await event_bus.dispatch(make_speech_stopped())
        assert wt._is_monitoring is True

        await event_bus.dispatch(make_response_created())
        assert wt._is_monitoring is False

    @pytest.mark.asyncio
    async def test_monitoring_does_not_start_while_agent_is_busy(
        self, event_bus: EventBus
    ) -> None:
        wt = UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=10.0)

        await event_bus.dispatch(SubAgentStartedEvent(agent_name="planner"))
        await event_bus.dispatch(make_speech_stopped())

        assert wt._is_monitoring is False


class TestTimeoutFiring:
    @pytest.mark.asyncio
    async def test_timeout_dispatches_user_inactivity_timeout_event(
        self, event_bus: EventBus
    ) -> None:
        received: list[UserInactivityTimeoutEvent] = []

        async def capture(e: UserInactivityTimeoutEvent) -> None:
            received.append(e)

        event_bus.subscribe(UserInactivityTimeoutEvent, capture)
        UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=0.01)

        await event_bus.dispatch(make_speech_stopped())
        await asyncio.sleep(0.6)

        assert len(received) == 1
        assert received[0].timeout_seconds == 0.01

    @pytest.mark.asyncio
    async def test_timeout_stops_monitoring_after_firing(
        self, event_bus: EventBus
    ) -> None:
        wt = UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=0.01)

        await event_bus.dispatch(make_speech_stopped())
        await asyncio.sleep(0.6)

        assert wt._is_monitoring is False

    @pytest.mark.asyncio
    async def test_user_speaking_cancels_timeout_before_firing(
        self, event_bus: EventBus
    ) -> None:
        received: list[UserInactivityTimeoutEvent] = []

        async def capture(e: UserInactivityTimeoutEvent) -> None:
            received.append(e)

        event_bus.subscribe(UserInactivityTimeoutEvent, capture)
        UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=0.5)

        await event_bus.dispatch(make_speech_stopped())
        await asyncio.sleep(0.05)
        await event_bus.dispatch(make_speech_started())
        await asyncio.sleep(0.6)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_timeout_fires_only_once(self, event_bus: EventBus) -> None:
        received: list[UserInactivityTimeoutEvent] = []

        async def capture(e: UserInactivityTimeoutEvent) -> None:
            received.append(e)

        event_bus.subscribe(UserInactivityTimeoutEvent, capture)
        UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=0.01)

        await event_bus.dispatch(make_speech_stopped())
        await asyncio.sleep(1.0)

        assert len(received) == 1


class TestCountdownDispatch:
    @pytest.mark.asyncio
    async def test_countdown_dispatches_each_second_only_once(
        self, event_bus: EventBus
    ) -> None:
        wt = UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=10.0)
        received: list[int] = []

        async def capture(event: UserInactivityCountdownEvent) -> None:
            received.append(event.remaining_seconds)

        event_bus.subscribe(UserInactivityCountdownEvent, capture)

        class FakeTimer:
            def __init__(self) -> None:
                self._remaining_values = iter([5, 5, 4, 4, 3])
                self._timed_out_values = iter([False, False, False, False, False, True])
                self._timeout_seconds = 10.0

            def has_timed_out(self) -> bool:
                return next(self._timed_out_values)

            def remaining(self) -> int:
                return next(self._remaining_values)

        wt._timer = FakeTimer()
        wt._is_monitoring = True

        await wt._monitor_timeout()

        assert received == [5, 4, 3]


class TestBusyStateTransitions:
    @pytest.mark.asyncio
    async def test_supervisor_finished_starts_monitoring_when_idle(
        self, event_bus: EventBus
    ) -> None:
        wt = UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=10.0)
        wt._agent_is_busy = True
        wt._assistant_is_speaking = False
        wt._user_has_stopped_speaking = True

        await event_bus.dispatch(SubAgentFinishedEvent(agent_name="planner"))

        assert wt._agent_is_busy is False
        assert wt._is_monitoring is True

    @pytest.mark.asyncio
    async def test_supervisor_finished_does_not_start_monitoring_if_assistant_speaking(
        self, event_bus: EventBus
    ) -> None:
        wt = UserInactivityTimeoutWatchdog(event_bus, timeout_seconds=10.0)
        wt._agent_is_busy = True
        wt._assistant_is_speaking = True
        wt._user_has_stopped_speaking = True

        await event_bus.dispatch(SubAgentFinishedEvent(agent_name="planner"))

        assert wt._agent_is_busy is False
        assert wt._is_monitoring is False
