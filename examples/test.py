import asyncio

from rtvoice import AgentListener, RealtimeAgent


class AgentListenerImpl(AgentListener):
    async def on_user_inactivity_countdown(self, remaining_seconds):
        print(f"User inactivity countdown: {remaining_seconds} seconds remaining")


async def ramp_speech_speed(
    agent: RealtimeAgent, start: float, end: float, duration: float, steps: int = 10
):
    step_delay = duration / steps
    for i in range(steps + 1):
        speed = start + (end - start) * (i / steps)
        await agent.set_speech_speed(speed)
        if i < steps:
            await asyncio.sleep(step_delay)


async def main():
    agent = RealtimeAgent(
        instructions="Du bist Jarvis. Antworte kurz und bündig.",
        inactivity_timeout_enabled=True,
        inactivity_timeout_seconds=10,
        listener=AgentListenerImpl(),
        speech_speed=0.5,
    )

    async def start_ramp():
        await asyncio.sleep(3.0)
        await ramp_speech_speed(agent, start=0.5, end=1.5, duration=10.0)

    asyncio.create_task(start_ramp())

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
