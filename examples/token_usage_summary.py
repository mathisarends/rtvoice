import asyncio

from dotenv import load_dotenv

from rtvoice import AgentListener, RealtimeAgent, SubAgent, TranscriptionModel
from rtvoice.llm import ChatOpenAI
from rtvoice.token import TokenUsageRecord, TokenUsageSummary

load_dotenv(override=True)


class ConsoleTranscriptListener(AgentListener):
    async def on_user_transcript(self, transcript: str) -> None:
        print(f"\033[36mDu: {transcript}\033[0m")

    async def on_assistant_transcript(self, transcript: str) -> None:
        print(f"Assistent: {transcript}")


def build_planning_subagent() -> SubAgent:
    return SubAgent(
        name="Planungs Agent",
        description="Hilft bei Planungs- und Strukturierungsaufgaben.",
        instructions=(
            "Du bist ein strukturierter Planungs-Agent. "
            "Wenn die Aufgabe nach Planung, Priorisierung oder einer Schrittfolge klingt, "
            "übernimm sie. "
            "Erstelle zuerst kurz eine sinnvolle Struktur und rufe dann done() mit dem finalen Ergebnis auf."
        ),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        result_instructions="Fasse das Ergebnis natürlich und knapp auf Deutsch zusammen.",
        holding_instruction="Sag genau einen kurzen Satz wie 'Ich plane das kurz.' und stoppe dann.",
    )


def print_usage_summary(summary: TokenUsageSummary) -> None:
    usage = summary.usage
    cost = summary.cost

    print("\n" + "=" * 72)
    print("TOKEN USAGE SUMMARY")
    print("=" * 72)
    print(f"Input Tokens:          {usage.input_tokens}")
    print(f"Cached Input Tokens:   {usage.cached_input_tokens}")
    print(f"Output Tokens:         {usage.output_tokens}")
    print(f"Total Tokens:          {usage.total_tokens}")
    print(f"Input Text Tokens:     {usage.input_text_tokens}")
    print(f"Input Audio Tokens:    {usage.input_audio_tokens}")
    print(f"Output Text Tokens:    {usage.output_text_tokens}")
    print(f"Output Audio Tokens:   {usage.output_audio_tokens}")
    print(f"Input Cost:            ${cost.input_usd:.6f}")
    print(f"Cached Input Cost:     ${cost.cached_input_usd:.6f}")
    print(f"Output Cost:           ${cost.output_usd:.6f}")
    print(f"Duration Cost:         ${cost.duration_usd:.6f}")
    print(f"Total Cost:            ${cost.total_usd:.6f}")

    if summary.has_unpriced_usage:
        print("Hinweis: Ein Teil der Usage konnte nicht belastbar bepreist werden.")

    if summary.by_model:
        print("\nNach Modell")
        print("-" * 72)
        for model_summary in summary.by_model:
            print(
                f"{model_summary.model}: "
                f"total_tokens={model_summary.usage.total_tokens}, "
                f"total_cost=${model_summary.cost.total_usd:.6f}, "
                f"priced={model_summary.price_available}"
            )

    if summary.records:
        print("\nEinzelne Records")
        print("-" * 72)
        for record in summary.records:
            print_record(record)


def print_record(record: TokenUsageRecord) -> None:
    print(
        f"source={record.source}, "
        f"model={record.model}, "
        f"input={record.usage.input_tokens}, "
        f"cached={record.usage.cached_input_tokens}, "
        f"output={record.usage.output_tokens}, "
        f"total={record.usage.total_tokens}, "
        f"cost=${record.cost.total_usd:.6f}, "
        f"priced={record.price_available}"
    )


async def main() -> None:
    agent = RealtimeAgent(
        extends_system_prompt="Du bist ein hilfreicher Voice-Assistent. Nutze fuer Planungen den Planungs Agent.",
        listener=ConsoleTranscriptListener(),
        transcription_model=TranscriptionModel.WHISPER_1,
        subagents=[build_planning_subagent()],
        inactivity_timeout_enabled=True,
        inactivity_timeout_seconds=5.0,
    )

    print("Sag zum Beispiel:")
    print("- 'Plane mir meinen Arbeitstag in drei Blöcken.'")
    print("- 'Priorisiere meine Aufgaben für heute.'")
    print("- 'Gib mir einen Lernplan für die nächsten zwei Stunden.'")
    print("\nDer Timeout ist hier bewusst niedrig gesetzt.")

    result = await agent.run()
    print_usage_summary(result.token_usage)


if __name__ == "__main__":
    asyncio.run(main())
