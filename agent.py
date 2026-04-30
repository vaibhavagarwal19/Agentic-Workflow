import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from tools import get_weather, check_calendar, get_price

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Result schema (required by main.py)
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    goal: str
    steps_taken: list[str]
    final_answer: str
    errors_encountered: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. Groq client
# ---------------------------------------------------------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
MAX_TURNS = 12


# ---------------------------------------------------------------------------
# 3. Tool schemas — what the LLM can call
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "Check if a slot is free on a date (YYYY-MM-DD). Weekends are blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                },
                "required": ["date", "duration_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Get price for an item. Returns an error dict for unknown items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["item", "quantity"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 4. Run a tool by name (handles weather retries + price errors)
# ---------------------------------------------------------------------------

def run_tool(name: str, args: dict, errors: list[str]) -> dict:
    print(f"[tool] {name}({args})")

    if name == "get_weather":
        # Retry up to 3 times because get_weather fails ~20% of the time.
        result = {"error": "weather failed after 3 retries"}
        for attempt in range(1, 4):
            try:
                result = get_weather(args["city"])
                break
            except Exception as e:
                print(f"  retry {attempt}/3: {e}")
                errors.append(f"get_weather: {e}")

    elif name == "check_calendar":
        result = check_calendar(args["date"], args["duration_minutes"])

    elif name == "get_price":
        result = get_price(args["item"], args["quantity"])
        if "error" in result:
            errors.append(result["error"])

    else:
        result = {"error": f"unknown tool '{name}'"}
        errors.append(result["error"])

    print(f"[tool] -> {result}")
    return result


# ---------------------------------------------------------------------------
# 5. System prompt (today's date is injected at runtime)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an agent that completes goals by calling tools.

Today is {today} ({weekday}). Use this for "today", "this week", "next week".

Tools:
  - get_weather(city)
  - check_calendar(date, duration_minutes)
  - get_price(item, quantity)

Rules:
  1. Run independent tool calls in PARALLEL in a single turn.
  2. Do arithmetic yourself.
  3. Stop calling tools once you can answer; reply in plain English.
"""


# ---------------------------------------------------------------------------
# 6. The agent loop
# ---------------------------------------------------------------------------

def run_agent(goal: str) -> AgentResult:
    today = datetime.now()
    system = SYSTEM_PROMPT.format(
        today=today.strftime("%Y-%m-%d"),
        weekday=today.strftime("%A"),
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": goal},
    ]
    steps: list[str] = []
    errors: list[str] = []

    print(f"\n[agent] goal: {goal}")

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- turn {turn} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.2,
        )
        msg = response.choices[0].message

        # Case A: no tool calls -> this is the final answer.
        if not msg.tool_calls:
            print("[agent] final answer ready")
            return AgentResult(
                goal=goal,
                steps_taken=steps,
                final_answer=(msg.content or "").strip(),
                errors_encountered=errors,
            )

        # Case B: model asked for tool(s). Append its message, then run each tool.
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = run_tool(name, args, errors)

            steps.append(f"{name}({args}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    # Safety cap
    return AgentResult(
        goal=goal,
        steps_taken=steps,
        final_answer="Agent stopped — turn limit hit.",
        errors_encountered=errors + ["turn limit"],
    )
