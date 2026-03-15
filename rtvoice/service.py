import asyncio
import logging
from pathlib import Path
from typing import Annotated, Self

from typing_extensions import Doc

from rtvoice.audio import (
    AudioInputDevice,
    AudioOutputDevice,
    AudioSession,
)
from rtvoice.conversation import ConversationHistory
from rtvoice.events import EventBus
from rtvoice.events.views import (
    AgentErrorEvent,
    AgentSessionConnectedEvent,
    AgentStartingEvent,
    AgentStoppedEvent,
    AssistantInterruptedEvent,
    AssistantStartedRespondingEvent,
    AssistantStoppedRespondingEvent,
    AssistantTranscriptCompletedEvent,
    AssistantTranscriptDeltaEvent,
    StartAgentCommand,
    SubAgentFinishedEvent,
    SubAgentStartedEvent,
    UpdateSpeechSpeedCommand,
    UserInactivityCountdownEvent,
    UserInactivityTimeoutEvent,
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
    UserTranscriptCompletedEvent,
)
from rtvoice.mcp import MCPServer
from rtvoice.realtime.providers import OpenAIProvider, RealtimeProvider
from rtvoice.realtime.websocket import RealtimeWebSocket
from rtvoice.shared.decorators import timed
from rtvoice.subagent import SubAgent
from rtvoice.subagent.views import SubAgentResult
from rtvoice.tools import SpecialToolParameters, Tools
from rtvoice.views import (
    AgentListener,
    AgentResult,
    AssistantVoice,
    ClarificationCheckpoint,
    NoiseReduction,
    OutputModality,
    RealtimeModel,
    SemanticVAD,
    TranscriptionModel,
    TurnDetection,
)
from rtvoice.watchdogs import (
    AudioForwardWatchdog,
    AudioPlayerWatchdog,
    AudioRecordingWatchdog,
    ErrorWatchdog,
    InterruptionWatchdog,
    LifecycleWatchdog,
    SessionWatchdog,
    SpeechStateWatchdog,
    SubAgentInteractionWatchdog,
    ToolCallingWatchdog,
    TranscriptionWatchdog,
    UserInactivityTimeoutWatchdog,
)

logger = logging.getLogger(__name__)


class RealtimeAgent[T]:
    """Event-driven voice agent using the OpenAI Realtime API.

    Manages the full lifecycle of a real-time voice session: audio I/O,
    WebSocket connection, tool calling, optional subagent handoffs,
    MCP server integration, and inactivity timeouts.

    Call [prewarm()][rtvoice.service.RealtimeAgent.prewarm] before
    [run()][rtvoice.service.RealtimeAgent.run] to prewarm connections
    and avoid startup delays.

    Example:
        ```python
        agent = RealtimeAgent(
            instructions="You are Jarvis, a helpful home assistant.",
            voice=AssistantVoice.MARIN,
            inactivity_timeout_seconds=30,
            inactivity_timeout_enabled=True,
        )
        result = await agent.run()
        ```
    """

    def __init__(
        self,
        *,
        instructions: Annotated[
            str,
            Doc("System prompt defining the assistant's personality and behavior."),
        ] = "",
        model: Annotated[
            RealtimeModel,
            Doc("Realtime model variant to use. Defaults to `GPT_REALTIME_MINI`."),
        ] = RealtimeModel.GPT_REALTIME_MINI,
        voice: Annotated[
            AssistantVoice,
            Doc("TTS voice used for assistant responses."),
        ] = AssistantVoice.MARIN,
        speech_speed: Annotated[
            float,
            Doc(
                "Playback speed of the assistant's voice. "
                "Automatically clamped to `[0.5, 1.5]`."
            ),
        ] = 1.0,
        transcription_model: Annotated[
            TranscriptionModel | None,
            Doc(
                "STT model used to produce `UserTranscriptCompletedEvent` transcripts. "
                "Pass `None` to disable transcription entirely."
            ),
        ] = TranscriptionModel.WHISPER_1,
        output_modalities: Annotated[
            list[OutputModality] | None,
            Doc(
                "Assistant response output modalities sent to Realtime API. "
                'Defaults to `["audio"]`. Include `"text"` to receive streamed text events.'
            ),
        ] = None,
        noise_reduction: Annotated[
            NoiseReduction,
            Doc(
                "Microphone noise reduction profile. Use `FAR_FIELD` for desktop mics."
            ),
        ] = NoiseReduction.FAR_FIELD,
        turn_detection: Annotated[
            TurnDetection | None,
            Doc(
                "Voice activity detection strategy. "
                "Defaults to `SemanticVAD` when `None`."
            ),
        ] = None,
        tools: Annotated[
            Tools | None,
            Doc(
                "Pre-registered tool set exposed to the model. "
                "Tools receive the shared `context` and `event_bus` automatically."
            ),
        ] = None,
        subagents: Annotated[
            list[SubAgent] | None,
            Doc(
                "Optional sub-agents reachable via auto-registered handoff tools. "
                "Prefer attaching MCP servers to subagents rather than the agent."
            ),
        ] = None,
        mcp_servers: Annotated[
            list[MCPServer] | None,
            Doc(
                "MCP servers connected during `prewarm()`. "
                "Their tools are registered and forwarded to the model."
            ),
        ] = None,
        audio_input: Annotated[
            AudioInputDevice | None,
            Doc("Audio input device. Defaults to `MicrophoneInput`."),
        ] = None,
        audio_output: Annotated[
            AudioOutputDevice | None,
            Doc("Audio output device. Defaults to `SpeakerOutput`."),
        ] = None,
        context: Annotated[
            T | None,
            Doc(
                "Shared context object forwarded to all tool handlers "
                "and all subagents."
            ),
        ] = None,
        event_bus: Annotated[
            EventBus | None,
            Doc(
                "Event bus for session event dispatch. "
                "If omitted, a new bus is created automatically."
            ),
        ] = None,
        listener: Annotated[
            AgentListener | None,
            Doc(
                "Callback interface for session lifecycle events "
                "(transcripts, speaking state, errors, …)."
            ),
        ] = None,
        inactivity_timeout_seconds: Annotated[
            float | None,
            Doc(
                "Seconds of user silence before the agent stops automatically. "
                "Has no effect unless `inactivity_timeout_enabled=True`."
            ),
        ] = None,
        inactivity_timeout_enabled: Annotated[
            bool,
            Doc(
                "Activates the inactivity timeout watchdog. "
                "Requires `inactivity_timeout_seconds` to be set."
            ),
        ] = False,
        recording_path: Annotated[
            str | Path | None,
            Doc(
                "If provided, the full session audio is recorded to this path "
                "via `AudioRecordingWatchdog`."
            ),
        ] = None,
        provider: Annotated[
            RealtimeProvider | None,
            Doc(
                "Realtime API provider. Defaults to `OpenAIProvider`. "
                "Pass an `AzureOpenAIProvider` instance to use Azure OpenAI."
            ),
        ] = None,
        api_key: Annotated[
            str | None,
            Doc(
                "OpenAI API key. Shortcut for `OpenAIProvider(api_key=...)`. "
                "Deprecated — prefer passing `provider=OpenAIProvider(api_key=...)` explicitly."
            ),
        ] = None,
    ):
        self._subagents = list(subagents or [])
        self._validate_subagent_names(self._subagents)

        if self._subagents and mcp_servers:
            logger.warning(
                "mcp_servers are set on RealtimeAgent alongside subagents. "
                "Consider attaching MCP servers to subagents instead."
            )

        if api_key and provider:
            raise ValueError("Pass either `provider` or `api_key`, not both.")

        self._instructions = instructions
        self._model = model
        self._voice = voice
        self._speech_speed = self._clip_speech_speed(speech_speed)

        if transcription_model is None and self._subagents:
            logger.warning(
                "transcription_model is None but subagents are attached. "
                "Transcription is required for subagent handoffs — "
                "defaulting to TranscriptionModel.WHISPER_1."
            )
            transcription_model = TranscriptionModel.WHISPER_1

        self._transcription_model = transcription_model
        self._transcription_enabled = transcription_model is not None
        self._output_modalities = self._normalize_output_modalities(output_modalities)
        self._assistant_text_enabled = "text" in self._output_modalities
        self._noise_reduction = noise_reduction
        self._turn_detection: TurnDetection = turn_detection or SemanticVAD()
        self._mcp_servers = mcp_servers or []

        if inactivity_timeout_seconds is not None and not inactivity_timeout_enabled:
            logger.warning(
                "inactivity_timeout_seconds is set but inactivity_timeout_enabled is False. "
                "The timeout will not be active."
            )

        self._should_enable_inactivity_timeout = (
            inactivity_timeout_enabled and inactivity_timeout_seconds is not None
        )

        self._listener = listener
        self._inactivity_timeout_seconds = inactivity_timeout_seconds
        self._inactivity_timeout_enabled = inactivity_timeout_enabled
        self._context = context
        self._recording_path = Path(recording_path) if recording_path else None

        self._stopped = asyncio.Event()
        self._stop_called = False
        self._mcp_ready = asyncio.Event()

        self._event_bus = event_bus or EventBus()
        self._conversation_history = ConversationHistory(self._event_bus)

        self._tools = Tools()
        if tools:
            self._tools.merge(tools)

        self._tools.set_context(
            SpecialToolParameters(
                event_bus=self._event_bus,
                context=context,
                conversation_history=self._conversation_history,
            )
        )
        for subagent in self._subagents:
            self._register_subagent(subagent)

        self._websocket = RealtimeWebSocket(
            model=self._model,
            provider=provider or OpenAIProvider(api_key=api_key),
        )

        audio_session = AudioSession(
            input_device=audio_input or self._default_audio_input(),
            output_device=audio_output or self._default_audio_output(),
        )

        self._setup_shutdown_handlers()
        self._setup_watchdogs(audio_session)
        self._setup_listener()

    def _default_audio_input(self) -> AudioInputDevice:
        from rtvoice.audio import MicrophoneInput

        return MicrophoneInput()

    def _default_audio_output(self) -> AudioOutputDevice:
        from rtvoice.audio import SpeakerOutput

        return SpeakerOutput()

    def _clip_speech_speed(self, speed: float) -> float:
        clipped = max(0.25, min(speed, 1.5))

        if speed != clipped:
            logger.warning(
                "Speech speed %.2f is out of range [0.25, 1.5], clipping to %.2f",
                speed,
                clipped,
            )

        return clipped

    def _normalize_output_modalities(
        self, output_modalities: list[OutputModality] | None
    ) -> list[OutputModality]:
        modalities = output_modalities or ["audio"]
        return list(dict.fromkeys(modalities))

    def _validate_subagent_names(self, subagents: list[SubAgent]) -> None:
        seen_names: set[str] = set()
        for subagent in subagents:
            if subagent.name in seen_names:
                raise ValueError(
                    f"Duplicate subagent name '{subagent.name}'. "
                    "Subagent names must be unique."
                )
            seen_names.add(subagent.name)

    def _register_subagent(self, subagent: SubAgent) -> None:
        description = subagent.description
        if subagent.handoff_instructions:
            description = (
                f"{subagent.description}\n\n"
                f"Handoff instructions: {subagent.handoff_instructions}"
            )

        subagent_name = subagent.name
        self._register_subagent_handoff(subagent, subagent_name, description)

    def _register_subagent_handoff(
        self, subagent: SubAgent, subagent_name: str, description: str
    ) -> None:
        paused_for_clarification: ClarificationCheckpoint | None = None

        @self._tools.action(
            description,
            name=subagent_name,
            result_instruction=subagent.result_instructions,
            holding_instruction=subagent.holding_instruction,
        )
        async def _handoff(
            task: Annotated[
                str,
                "The task or question to delegate to this agent. Be specific and include enough context for the agent to act without clarification.",
            ],
            conversation_history: ConversationHistory,
            clarification_answer: Annotated[
                str | None,
                "If this is a follow-up call after a clarification request, provide the user's answer here. Leave empty for the initial call.",
            ] = None,
        ) -> SubAgentResult:
            nonlocal paused_for_clarification

            is_resuming = (
                paused_for_clarification is not None
                and clarification_answer is not None
            )

            if is_resuming:
                checkpoint = paused_for_clarification
                paused_for_clarification = None
                result = await subagent.run(
                    task,
                    resume_history=checkpoint.resume_history,
                    clarify_call_id=checkpoint.clarify_call_id,
                    clarification_answer=clarification_answer,
                )
            else:
                context = (
                    conversation_history.format() if conversation_history else None
                )
                result = await subagent.run(task, context=context)

            if result.clarification_needed:
                # Subagent yielded control back to the realtime agent to collect user input
                paused_for_clarification = ClarificationCheckpoint(
                    resume_history=result.resume_history,
                    clarify_call_id=result.clarify_call_id,
                )

            return result

    def _setup_shutdown_handlers(self) -> None:
        self._event_bus.subscribe(
            UserInactivityTimeoutEvent, self._on_inactivity_timeout
        )

    def _setup_watchdogs(self, audio_session: AudioSession) -> None:
        self._audio_player_watchdog = AudioPlayerWatchdog(
            event_bus=self._event_bus,
            audio_session=audio_session,
        )
        self._lifecycle_watchdog = LifecycleWatchdog(
            event_bus=self._event_bus, websocket=self._websocket
        )
        self._session_watchdog = SessionWatchdog(
            event_bus=self._event_bus, websocket=self._websocket
        )
        self._audio_forward_watchdog = AudioForwardWatchdog(
            event_bus=self._event_bus, websocket=self._websocket
        )
        self._interruption_watchdog = InterruptionWatchdog(
            event_bus=self._event_bus,
            websocket=self._websocket,
            audio_session=audio_session,
        )
        if self._transcription_enabled or self._assistant_text_enabled:
            self._transcription_watchdog = TranscriptionWatchdog(
                event_bus=self._event_bus
            )

        self._tool_calling_watchdog = ToolCallingWatchdog(
            event_bus=self._event_bus,
            tools=self._tools,
            websocket=self._websocket,
            subagent_tool_names={s.name for s in self._subagents} or None,
        )
        if self._subagents:
            self._subagent_watchdog = SubAgentInteractionWatchdog(
                event_bus=self._event_bus,
                tools=self._tools,
                websocket=self._websocket,
            )
            for subagent in self._subagents:
                self._subagent_watchdog.register_subagent(subagent.name, subagent)
        self._error_watchdog = ErrorWatchdog(event_bus=self._event_bus)
        self._speech_state_watchdog = SpeechStateWatchdog(event_bus=self._event_bus)

        if self._should_enable_inactivity_timeout:
            self._user_inactivity_timeout_watchdog = UserInactivityTimeoutWatchdog(
                event_bus=self._event_bus,
                timeout_seconds=self._inactivity_timeout_seconds,
            )

        if self._recording_path:
            self._recording_watchdog = AudioRecordingWatchdog(
                event_bus=self._event_bus,
                output_path=self._recording_path,
            )

    def _setup_listener(self) -> None:
        if not self._listener:
            return

        self._warn_listener_countdown_mismatch_if_necessary()
        self._warn_listener_subagent_mismatch_if_necessary()
        self._warn_listener_text_modality_mismatch_if_necessary()

        self._event_bus.subscribe(
            UserTranscriptCompletedEvent,
            lambda e: self._listener.on_user_transcript(e.transcript),
        )
        self._event_bus.subscribe(
            AssistantTranscriptCompletedEvent,
            lambda e: self._listener.on_assistant_transcript(e.transcript),
        )
        self._event_bus.subscribe(
            AssistantTranscriptDeltaEvent,
            lambda e: self._listener.on_assistant_transcript_delta(e.delta),
        )
        self._event_bus.subscribe(
            AgentStartingEvent,
            lambda _: self._listener.on_agent_starting(),
        )
        self._event_bus.subscribe(
            AgentSessionConnectedEvent,
            lambda _: self._listener.on_agent_session_connected(),
        )
        self._event_bus.subscribe(
            AssistantInterruptedEvent, lambda _: self._listener.on_agent_interrupted()
        )
        self._event_bus.subscribe(
            AgentErrorEvent, lambda e: self._listener.on_agent_error(e.error)
        )
        self._event_bus.subscribe(
            UserStartedSpeakingEvent,
            lambda _: self._listener.on_user_started_speaking(),
        )
        self._event_bus.subscribe(
            UserStoppedSpeakingEvent,
            lambda _: self._listener.on_user_stopped_speaking(),
        )
        self._event_bus.subscribe(
            AssistantStartedRespondingEvent,
            lambda _: self._listener.on_assistant_started_responding(),
        )
        self._event_bus.subscribe(
            AssistantStoppedRespondingEvent,
            lambda _: self._listener.on_assistant_stopped_responding(),
        )
        self._event_bus.subscribe(
            UserInactivityCountdownEvent,
            lambda e: self._listener.on_user_inactivity_countdown(e.remaining_seconds),
        )
        self._event_bus.subscribe(
            SubAgentStartedEvent,
            lambda e: self._listener.on_subagent_started(e.agent_name),
        )
        self._event_bus.subscribe(
            SubAgentFinishedEvent,
            lambda e: self._listener.on_subagent_finished(e.agent_name),
        )

    def _warn_listener_countdown_mismatch_if_necessary(self) -> None:
        overrides_countdown = self._listener_overrides_countdown()
        listener_name = type(self._listener).__name__

        if overrides_countdown and not self._should_enable_inactivity_timeout:
            logger.warning(
                "Listener '%s' overrides on_user_inactivity_countdown "
                "but inactivity_timeout_enabled is False — callback will never fire.",
                listener_name,
            )

        if self._should_enable_inactivity_timeout and not overrides_countdown:
            logger.warning(
                "inactivity_timeout_enabled is True but listener '%s' does not override "
                "on_user_inactivity_countdown — countdown events will be silently ignored.",
                listener_name,
            )

    def _listener_overrides_countdown(self) -> bool:
        cls = type(self._listener)
        listener_method = getattr(cls, "on_user_inactivity_countdown", None)
        if listener_method is None:
            return False
        return listener_method is not AgentListener.on_user_inactivity_countdown

    def _warn_listener_subagent_mismatch_if_necessary(self) -> None:
        if self._listener_overrides_subagent_callbacks() and not self._subagents:
            logger.warning(
                "Listener '%s' overrides on_subagent_started or on_subagent_finished "
                "but no subagents are configured — callbacks will never fire.",
                type(self._listener).__name__,
            )

    def _warn_listener_text_modality_mismatch_if_necessary(self) -> None:
        if (
            self._listener_overrides_assistant_transcript_delta()
            and not self._assistant_text_enabled
        ):
            logger.warning(
                "Listener '%s' overrides on_assistant_transcript_delta "
                "but output_modalities does not include 'text' — callback will never fire.",
                type(self._listener).__name__,
            )

    def _listener_overrides_assistant_transcript_delta(self) -> bool:
        cls = type(self._listener)
        delta = getattr(cls, "on_assistant_transcript_delta", None)
        return (
            delta is not None
            and delta is not AgentListener.on_assistant_transcript_delta
        )

    def _listener_overrides_subagent_callbacks(self) -> bool:
        cls = type(self._listener)
        started = getattr(cls, "on_subagent_started", None)
        finished = getattr(cls, "on_subagent_finished", None)
        return (
            started is not None and started is not AgentListener.on_subagent_started
        ) or (
            finished is not None and finished is not AgentListener.on_subagent_finished
        )

    async def _on_inactivity_timeout(self, event: UserInactivityTimeoutEvent) -> None:
        logger.info(
            "User inactivity timeout after %.1f seconds - triggering shutdown",
            event.timeout_seconds,
        )
        asyncio.ensure_future(self.stop())

    async def run(
        self,
    ) -> Annotated[
        AgentResult,
        Doc("Conversation history and recording path after the session ends."),
    ]:
        """Start the agent and block until the session ends.

        Dispatches a `StartAgentCommand` to kick off audio I/O and the WebSocket
        connection, then waits until `stop()` is called — either manually, via
        inactivity timeout, or through an error watchdog.
        """
        logger.info("Starting agent...")

        await self._event_bus.dispatch(AgentStartingEvent())
        await self.prewarm()

        await self._event_bus.dispatch(
            StartAgentCommand(
                model=self._model,
                instructions=self._instructions,
                voice=self._voice,
                speech_speed=self._speech_speed,
                transcription_model=self._transcription_model,
                output_modalities=self._output_modalities,
                noise_reduction=self._noise_reduction,
                turn_detection=self._turn_detection,
                tools=self._tools,
            )
        )
        logger.info("Agent started successfully")

        try:
            await self._stopped.wait()
        finally:
            await self.stop()

        return AgentResult(
            turns=self._conversation_history.turns,
            recording_path=self._recording_path,
        )

    @timed()
    async def prewarm(
        self,
    ) -> Annotated[Self, Doc("Returns `self` for optional chaining with `run()`.")]:
        """Prewarm MCP and subagent connections before `run()`.

        Calling this explicitly avoids a cold-start delay when the session begins.
        Safe to call multiple times — subsequent calls are no-ops for MCP servers
        that are already connected.
        """
        tasks = [self._connect_mcp_servers()]
        tasks.extend(subagent.prewarm() for subagent in self._subagents)

        await asyncio.gather(*tasks, return_exceptions=True)
        return self

    async def _connect_mcp_servers(self) -> None:
        if self._mcp_ready.is_set() or not self._mcp_servers:
            self._mcp_ready.set()
            return

        results = await asyncio.gather(
            *[self._connect_and_register_mcp_server(s) for s in self._mcp_servers],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("MCP server connection failed: %s", result)

        self._mcp_ready.set()

    async def _connect_and_register_mcp_server(self, server: MCPServer) -> None:
        await server.connect()
        tools = await server.list_tools()
        for tool in tools:
            self._tools.register_mcp(tool, server)
        logger.info("MCP server connected: %d tools loaded", len(tools))

    async def set_speech_speed(
        self,
        speed: Annotated[
            float,
            Doc("Target playback speed. Automatically clamped to ``[0.25, 1.5]``."),
        ],
    ) -> None:
        """Update the assistant's speech speed mid-session.

        Clamps the value to ``[0.25, 1.5]`` before applying. The change takes
        effect on the next response — audio that is already playing is unaffected.
        """
        clipped = self._clip_speech_speed(speed)
        await self._event_bus.dispatch(UpdateSpeechSpeedCommand(speed=clipped))

    @timed()
    async def stop(self) -> None:
        """Gracefully shut down the agent.

        Cleans up all MCP server connections, dispatches `AgentStoppedEvent`,
        and signals the `run()` coroutine to return. Idempotent — safe to call
        multiple times.
        """
        if self._stop_called:
            return
        self._stop_called = True

        logger.info("Stopping agent...")

        for server in self._mcp_servers:
            await server.cleanup()

        await self._event_bus.dispatch(AgentStoppedEvent())

        self._stopped.set()
        logger.info("Agent stopped successfully")

        if self._listener:
            await self._listener.on_agent_stopped()
