"""Per-dependency tool injection demo."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel

from rtvoice import Inject, RealtimeAgent, Tools

type CustomerId = str
type CustomerName = str


@dataclass(frozen=True, slots=True)
class CustomerDirectory:
    names: Mapping[CustomerId, CustomerName]

    def name_for(self, customer_id: CustomerId) -> CustomerName:
        return self.names.get(customer_id, "Unknown")


@dataclass(frozen=True, slots=True)
class PlanPolicy:
    premium_customer_ids: frozenset[CustomerId]

    def tier_for(self, customer_id: CustomerId) -> str:
        return "Premium" if customer_id in self.premium_customer_ids else "Free"


class CustomerParams(BaseModel):
    customer_id: CustomerId


tools = Tools()


@tools.action("Look up a customer's name and plan.", params=CustomerParams)
def describe_customer(
    params: CustomerParams,
    customers: Inject[CustomerDirectory],
    plans: Inject[PlanPolicy],
) -> str:
    name = customers.name_for(params.customer_id)
    tier = plans.tier_for(params.customer_id)
    return f"Customer {params.customer_id}: {name}, plan {tier}"


async def main() -> None:
    load_dotenv(override=True)
    customers = CustomerDirectory(names={"42": "Max", "7": "Ada"})
    plans = PlanPolicy(premium_customer_ids=frozenset({"42"}))
    agent = RealtimeAgent(
        system_prompt="Answer customer questions concisely. Use the customer tool.",
        tools=tools,
        tool_dependencies=(customers, plans),
    )

    print("Try asking: Which plan does customer 42 have?")
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
