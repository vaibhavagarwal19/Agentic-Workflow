"""
agent.py — Tool-calling agent (Groq + simple loop)
===================================================
The agent loop is just:
    ask LLM -> if it wants tools, run them and feed results back -> repeat
    -> when the LLM stops asking for tools, that message IS the answer.
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from tools import get_weather, check_calendar, get_price

load_dotenv()  # load GROQ_API_KEY from .env before constructing the client


# ---------------------------------------------------------------------------
# Result schema (required by main.py)
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    goal: str
    steps_taken: list[str]
    final_answer: str
    errors_encountered: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
MAX_TURNS = 12


# ---------------------------------------------------------------------------
# Tool schemas — what the LLM sees
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. May time out occasionally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "Check if a slot is free on a date (YYYY-MM-DD). Weekends are always blocked.",
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
            "description": "Look up unit price and total. Returns an error dict for unknown items.",
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
# Tool execution
# ---------------------------------------------------------------------------

def _call_weather_with_retry(city: str, errors: list[str]) -> dict:
    """get_weather fails ~20% of the time — retry up to 3 times."""
    for attempt in range(1, 4):
        try:
            return get_weather(city=city)
        except Exception as e:
            msg = f"get_weather('{city}') attempt {attempt}/3 failed: {e}"
            print(f"  [retry] {msg}")
            errors.append(msg)
    return {"error": f"get_weather('{city}') failed after 3 retries"}


def _execute_tool(name: str, args: dict, errors: list[str]) -> dict:
    """Run a tool by name and return a dict the LLM can read."""
    print(f"[agent] -> {name}({args})")

    if name == "get_weather":
        result = _call_weather_with_retry(args["city"], errors)

    elif name == "check_calendar":
        result = check_calendar(
            date=args["date"],
            duration_minutes=int(args["duration_minutes"]),
        )

    elif name == "get_price":
        result = get_price(item=args["item"], quantity=int(args["quantity"]))
        if "error" in result:
            errors.append(f"get_price: {result['error']}")

    else:
        result = {"error": f"unknown tool '{name}'"}
        errors.append(result["error"])

    print(f"[agent] <- {result}")
    return result


# ---------------------------------------------------------------------------
# System prompt — generic, no hardcoded cities/items/dates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous agent that completes user goals by calling tools.

Today is {today} ({weekday}). Use this for relative dates like "today", "this week", "next week".
When the user asks for weekdays, restrict to Mon-Fri.

Tools:
  - get_weather(city)               returns condition, temp_c, humidity_pct (may time out)
  - check_calendar(date, duration)  returns available and reason (weekends always blocked)
  - get_price(item, quantity)       returns unit_price and total, or an error with available_items

Rules:
  1. Plan the minimum tool calls needed.
  2. Run independent calls in PARALLEL (request multiple tools in one assistant turn).
  3. If a weather call returns an error, the runtime already retried — answer with what you have.
  4. If get_price returns an error key, pick a closest match from available_items or report it.
  5. Do arithmetic yourself; never call a tool for math.
  6. Stop calling tools once you can answer. Reply in clear plain English.
"""


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

def run_agent(goal: str) -> AgentResult:
    today = datetime.now()
    system = SYSTEM_PROMPT.format(
        today=today.strftime("%Y-%m-%d"),
        weekday=today.strftime("%A"),
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": goal},
    ]

    steps: list[str] = []
    errors: list[str] = []
    raw: dict[str, list] = {}

    print(f"\n[agent] goal: {goal}")

    for turn in range(1, MAX_TURNS + 1):
        print(f"[agent] --- turn {turn} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = response.choices[0].message

        # Case A: model wants to call tools
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                result = _execute_tool(name, args, errors)

                steps.append(f"{name}({json.dumps(args)}) -> {json.dumps(result)}")
                raw.setdefault(name, []).append({"args": args, "result": result})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result),
                })

            continue  # let the model see the tool results

        # Case B: no tool calls -> the message is the final answer
        final_answer = (msg.content or "").strip()
        print("[agent] final answer ready\n")
        return AgentResult(
            goal=goal,
            steps_taken=steps,
            final_answer=final_answer,
            errors_encountered=errors,
            raw_data=raw,
        )

    # Hit the safety cap
    errors.append(f"Exceeded {MAX_TURNS}-turn limit")
    return AgentResult(
        goal=goal,
        steps_taken=steps,
        final_answer="Agent stopped — exceeded turn limit.",
        errors_encountered=errors,
        raw_data=raw,
    )
