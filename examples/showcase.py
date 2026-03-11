"""
rtvoice Showcase — When to use a SupervisorAgent
=================================================

Rule of thumb
-------------
Use the RealtimeAgent directly for anything that resolves in one tool call
and finishes in under a second (time lookup, unit conversion, reminders).

Use a SupervisorAgent when the task:

- requires **multiple sequential tool calls** to produce a final answer,
- takes **more than a second or two** (external APIs, LLM calls, DB queries),
- benefits from **progress narration** so the user isn't left in silence, or
- may need to **ask the user a question** mid-execution.

This example has one supervisor: a "deployment analyst" that runs several
steps (fetch cluster status → check image registry → diff Helm values →
summarise) and reports back after each one.

Try saying
----------
- "Is the vizro backend ready to deploy?"
  → Multi-step analysis with live status updates and a clarifying question
    about which cluster to check (dev vs demo).

- "What time is it?"
  → Answered immediately by the RealtimeAgent, no supervisor involved.

Running
-------
::

    OPENAI_API_KEY=sk-... python showcase.py
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import Annotated

from llmify import ChatOpenAI

from rtvoice import RealtimeAgent, SupervisorAgent, Tools

logging.getLogger("rtvoice.events.bus").setLevel(logging.WARNING)


def build_deployment_tools() -> Tools:
    """
    Four mock tools that simulate real DevOps checks.
    Each one sleeps to represent an actual network call.
    The agent will sequence them and narrate every step via status().
    """
    tools = Tools()

    @tools.action(
        "Fetch the current rollout status for a service on a given cluster. "
        "Returns pod count, readiness, and last-deployed timestamp.",
    )
    async def get_cluster_status(
        service: Annotated[str, "Kubernetes service name, e.g. 'vizro-backend'."],
        cluster: Annotated[str, "Target cluster: 'dev' or 'demo'."],
    ) -> dict:
        await asyncio.sleep(1.2)  # simulate kubectl / GKE API call
        ready = random.choice([True, True, False])
        return {
            "service": service,
            "cluster": cluster,
            "pods_ready": f"{'3/3' if ready else '1/3'}",
            "last_deployed": "2025-03-06T14:22:00Z",
            "status": "Healthy" if ready else "Degraded",
        }

    @tools.action(
        "Check whether the latest image tag in the registry matches "
        "what is currently running on the cluster.",
    )
    async def check_image_tag(
        service: Annotated[str, "Service name."],
        cluster: Annotated[str, "Target cluster: 'dev' or 'demo'."],
    ) -> dict:
        await asyncio.sleep(0.8)  # simulate registry API call
        in_sync = random.choice([True, False])
        return {
            "registry_tag": "sha256:abc123f",
            "deployed_tag": "sha256:abc123f" if in_sync else "sha256:old9999",
            "in_sync": in_sync,
        }

    @tools.action(
        "Diff the Helm values between the GitOps repo and the live release. "
        "Returns a list of changed keys, or an empty list if nothing has drifted.",
    )
    async def diff_helm_values(
        service: Annotated[str, "Service name."],
        cluster: Annotated[str, "Target cluster."],
    ) -> dict:
        await asyncio.sleep(0.9)  # simulate ArgoCD / Helm API call
        drifted_keys = random.choice(
            [[], [], ["resources.limits.memory", "replicaCount"]]
        )
        return {
            "drifted": bool(drifted_keys),
            "changed_keys": drifted_keys,
        }

    @tools.action(
        "Fetch the last 20 error lines from the service logs on a given cluster.",
    )
    async def get_recent_errors(
        service: Annotated[str, "Service name."],
        cluster: Annotated[str, "Target cluster."],
    ) -> dict:
        await asyncio.sleep(0.7)
        errors = random.choice(
            [
                [],
                ["WARN  DB pool exhausted (2×)", "ERROR Failed health check /readyz"],
            ]
        )
        return {"error_count": len(errors), "recent_errors": errors}

    return tools


def build_deployment_analyst() -> SupervisorAgent:
    """
    Coordinates four sequential checks and narrates every step.

    The instructions deliberately tell the model to:
    - use status() before each slow tool call so the user is never silent,
    - use clarify() when the cluster is ambiguous,
    - finish with done() summarising all findings in plain language.
    """
    return SupervisorAgent(
        name="Deployment Analyst",
        description=(
            "Analyses whether a service is healthy and ready to deploy by "
            "checking cluster status, image tags, Helm drift, and recent logs. "
            "Use this agent for any question about deployments, rollouts, "
            "service health, or 'is X ready to deploy'."
        ),
        handoff_instructions=(
            "Always include the exact service name. "
            "If the user said which cluster (dev / demo), include it. "
            "Otherwise leave it out — the agent will ask."
        ),
        instructions=(
            "You are a senior DevOps analyst.\n\n"
            "When asked whether a service is ready to deploy, run these steps "
            "in order and use status() before every slow call:\n\n"
            "1. If the target cluster was not specified, call clarify() and ask "
            "   'Should I check the dev cluster, the demo cluster, or both?'\n"
            "2. status('Fetching cluster rollout status…')\n"
            "   → call get_cluster_status()\n"
            "3. status('Checking image tag in the registry…')\n"
            "   → call check_image_tag()\n"
            "4. status('Diffing Helm values against GitOps repo…')\n"
            "   → call diff_helm_values()\n"
            "5. status('Scanning recent error logs…')\n"
            "   → call get_recent_errors()\n"
            "6. call done() with a concise plain-language verdict:\n"
            "   - Is it safe to deploy? Yes / No / With caveats?\n"
            "   - What are the blockers, if any?\n"
            "   - One sentence recommendation.\n\n"
            "Never skip status() before a slow step. "
            "The user should always know what you are doing."
        ),
        tools=build_deployment_tools(),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        max_iterations=12,
        holding_instruction=(
            "Say ONE warm sentence to acknowledge the request, "
            "e.g. 'Sure, let me run a full deployment check — give me a moment!'. "
            "Then stop. Do not speculate about results."
        ),
        result_instructions=(
            "Read out the deployment verdict clearly and conversationally. "
            "Lead with the yes/no answer, then briefly mention blockers if any."
        ),
    )


async def main() -> None:
    tools = Tools()

    @tools.action("Gets the current time in a human-friendly format.")
    def get_current_time() -> str:
        return datetime.now().strftime("%I:%M %p")

    agent = RealtimeAgent(
        instructions=(
            "You are Echo, a concise voice assistant for a Kubernetes platform team.\n\n"
            "Answer simple questions (time, definitions, quick maths) directly.\n\n"
            "For anything involving deployments, rollouts, service health, "
            "image tags, Helm charts, or 'is X ready', hand off to the "
            "Deployment Analyst — do not try to answer those yourself."
        ),
        supervisor_agent=build_deployment_analyst(),
        inactivity_timeout_seconds=90,
        inactivity_timeout_enabled=True,
        tools=tools,
    )

    print("🎙  Echo is ready.\n")
    print("  Try: 'Is the vizro backend ready to deploy?'")
    print("       'Check the demo cluster for vizro-worker.'")
    print("       'What time is it?'  ← answered directly, no supervisor")
    print("\n  Speak mid-analysis to test interruption. Ctrl+C to stop.\n")

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
