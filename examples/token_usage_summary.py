import asyncio

from dotenv import load_dotenv

from rtvoice import RealtimeAgent

load_dotenv(override=True)


async def main() -> None:
    agent = RealtimeAgent(
        extends_system_prompt="Du bist ein hilfreicher Voice-Assistent.",
        inactivity_timeout_enabled=True,
        inactivity_timeout_seconds=5.0,
    )

    result = await agent.run()
    summary = result.token_usage
    usage, cost = summary.usage, summary.cost

    print(f"Input:   {usage.input_tokens} tokens  (${cost.input_usd:.6f})")
    print(
        f"Cached:  {usage.cached_input_tokens} tokens  (${cost.cached_input_usd:.6f})"
    )
    print(f"Output:  {usage.output_tokens} tokens  (${cost.output_usd:.6f})")
    print(f"Total:   {usage.total_tokens} tokens  (${cost.total_usd:.6f})")


if __name__ == "__main__":
    asyncio.run(main())
