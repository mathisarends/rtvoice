import asyncio

from llmify import ChatOpenAI

from rtvoice.skills import Skill
from rtvoice.subagent import SubAgent

PANDAS_SKILL = Skill.from_content(
    name="pandas",
    summary="Data analysis with pandas DataFrames",
    instructions="""
# Pandas Data Analysis Skill

## When to use this skill
Load this skill when the task involves analyzing tabular data, CSVs, or DataFrames.

## Best Practices
- Always use `df.info()` and `df.describe()` first to understand the data shape
- Prefer `df.query()` over boolean indexing for readability
- Use `df.pipe()` for chained transformations to keep code clean
- Handle missing values explicitly: `df.fillna()` or `df.dropna()` depending on context

## Common Patterns

### Load & inspect
```python
import pandas as pd
df = pd.read_csv("data.csv")
df.info()
df.describe()
```

### Aggregate
```python
summary = df.groupby("category")["value"].agg(["mean", "sum", "count"])
```

### Filter + transform
```python
result = (
    df
    .query("status == 'active' and age > 18")
    .assign(score_normalized=lambda x: x["score"] / x["score"].max())
    .sort_values("score_normalized", ascending=False)
)
```
""",
)

SQL_SKILL = Skill.from_content(
    name="sql",
    summary="Write and optimize SQL queries",
    instructions="""
# SQL Query Skill

## When to use this skill
Load this skill when the task requires writing, reviewing, or optimizing SQL.

## Best Practices
- Always use CTEs (`WITH`) for readability over nested subqueries
- Prefer explicit JOINs over implicit comma joins
- Use `EXPLAIN ANALYZE` to debug slow queries
- Avoid `SELECT *` in production queries

## Common Patterns

### CTE with aggregation
```sql
WITH active_users AS (
    SELECT user_id, COUNT(*) AS event_count
    FROM events
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY user_id
)
SELECT u.name, au.event_count
FROM users u
JOIN active_users au USING (user_id)
ORDER BY au.event_count DESC
LIMIT 20;
```

### Window function
```sql
SELECT
    user_id,
    revenue,
    SUM(revenue) OVER (PARTITION BY user_id ORDER BY created_at) AS running_total
FROM orders;
```
""",
)

VISUALIZATION_SKILL = Skill.from_content(
    name="visualization",
    summary="Create charts and visualizations with matplotlib/plotly",
    instructions="""
# Visualization Skill

## When to use this skill
Load this skill when the task requires producing charts, plots, or dashboards.

## Best Practices
- Use plotly for interactive output, matplotlib for static/print
- Always label axes and add a title
- Use color palettes that are colorblind-friendly (e.g. viridis, cividis)
- Export with `fig.write_html()` for sharing, `fig.write_image()` for reports

## Common Patterns

### Plotly bar chart
```python
import plotly.express as px

fig = px.bar(
    df,
    x="category",
    y="revenue",
    color="region",
    title="Revenue by Category and Region",
    labels={"revenue": "Revenue (€)", "category": "Category"},
    color_discrete_sequence=px.colors.qualitative.Vivid,
)
fig.update_layout(bargap=0.2)
fig.show()
```

### Matplotlib time series
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["date"], df["value"], linewidth=2, color="#2563eb")
ax.fill_between(df["date"], df["value"], alpha=0.1, color="#2563eb")
ax.set_title("Value Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
plt.tight_layout()
```
""",
)


# ---------------------------------------------------------------------------
# Mock Tools (simulieren echte Datenbankabfragen etc.)
# ---------------------------------------------------------------------------

from typing import Annotated

from rtvoice.tools import SubAgentTools


def build_mock_tools() -> SubAgentTools:
    tools = SubAgentTools()

    @tools.action(
        "Execute a SQL query against the analytics database and return results as CSV."
    )
    def run_sql(
        query: Annotated[str, "The SQL query to execute."],
    ) -> str:
        print(f"  [mock] SQL: {query}")
        # Spalten aus dem Query ableiten damit der Agent nicht verwirrt wird
        q = query.lower()
        if "revenue" in q and ("user" in q or "username" in q):
            return (
                "username,total_revenue\n"
                "Carol,6800.00\n"
                "Alice,4200.00\n"
                "Bob,3100.50\n"
                "Eve,2400.00\n"
                "Dave,1200.00\n"
            )
        # Fallback
        return "result\nno data matched\n"

    @tools.action(
        "Load a CSV file and return a summary of its structure and first rows."
    )
    def load_csv(
        filename: Annotated[str, "Path to the CSV file."],
    ) -> str:
        print(f"  [mock] Loading CSV: {filename}")
        return (
            "Shape: (1000, 6)\n"
            "Columns: date, username, category, revenue, region, status\n"
            "Dtypes: date=datetime64, revenue=float64\n"
            "Missing values: none\n"
        )

    @tools.action("Save a generated chart or result to disk.")
    def save_output(
        filename: Annotated[str, "Output filename."],
        content: Annotated[str, "The content or code to save."],
    ) -> str:
        print(f"  [mock] Saving: {filename} ({len(content)} chars)")
        return f"Successfully saved: {filename}"

    return tools


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o")
    agent = SubAgent(
        name="data_analyst",
        description="Analyzes datasets using SQL, pandas, and visualization tools.",
        instructions=(
            "You are an expert data analyst. "
            "You have access to a database and CSV files. "
            "Always load the appropriate skill before starting a task. "
            "Produce clean, well-commented code and explain your reasoning."
        ),
        llm=llm,
        tools=build_mock_tools(),
        skills=[PANDAS_SKILL, SQL_SKILL, VISUALIZATION_SKILL],
        dynamic_skills=True,
        max_iterations=15,
    )

    print("=" * 60)
    print("Task: Top 5 users by revenue als Bar Chart")
    print("=" * 60)

    result = await agent.run(
        task=(
            "Query the top 5 users by total revenue from the analytics database. "
            "The relevant tables are: 'sales' (columns: user_id, username, revenue) "
            "and 'users' (columns: user_id, username). "
            "Then generate a plotly bar chart and save it as 'top_users.html'."
        )
    )

    print("\n--- Result ---")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Tool calls executed: {[tc.name for tc in result.tool_calls]}")


if __name__ == "__main__":
    asyncio.run(main())
