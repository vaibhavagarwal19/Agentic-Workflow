# Agentic Workflow — Candidate Submission

An AI agent that accepts a natural-language goal and completes it by calling a small set of mock tools. The agent plans tool calls, runs them (in parallel where possible), handles errors and unexpected results, and returns a structured `AgentResult`.

## Requirements satisfied

1. Uses only the three mock tools in `tools.py` (no real API calls).
2. Handles `get_weather()` timeouts (~20% rate) with up to 3 retries.
3. Handles `get_price()`'s error-dict pattern (returns `{error, available_items}` instead of raising).
4. Prints a reasoning trace for every tool call (`[agent] -> ...`, `[agent] <- ...`).
5. Returns an `AgentResult` per goal: `goal`, `steps_taken`, `final_answer`, `errors_encountered`, `raw_data`.

## Project layout

```
candidate_pack/
  agent.py        # the agent (this is what was implemented)
  main.py         # entry point — runs the 3 goals (do not modify)
  tools.py        # the 3 mock tools (do not modify)
  requirements.txt
  .env            # holds GROQ_API_KEY
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_groq_api_key
```

Get a key from <https://console.groq.com/keys>.

## Run

```bash
python main.py
```

This runs all three goals end-to-end and prints the trace + final answer + structured result for each.

## Design — why a custom loop instead of LangChain / LangGraph

The agent is implemented as a plain Python loop (~150 lines) using Groq's tool-calling API. The whole control flow fits in one function:

```
ask LLM ──► did it return tool_calls?
              ├─ yes ──► run them, append results, loop
              └─ no  ──► that message is the final answer
```

Reasons for going framework-free:

- **Readability.** Anyone can read the loop top-to-bottom and understand exactly what happens. No hidden state, no callback graphs, no DSL.
- **Debuggability.** Every `print` line maps directly to a line of code in `agent.py`. There is no abstraction between what the LLM decides and what the runtime executes.
- **Portability.** Groq's API is wire-compatible with OpenAI's tool-calling format, so the same loop would work on OpenAI / Together / any compatible provider with a one-line client swap.
- **No goal-specific code.** All three goals — multi-step planning, parallel pricing + calendar lookups, and conditional branching — flow through the same loop. The LLM decides what to call; the loop just dispatches.

## How the agent handles each goal

The three assignment goals demonstrate three different agent capabilities, all driven by the same loop:

### Goal 1 — Multi-step planning
> *"Find the cheapest day next week (Mon–Fri) to schedule a 60-minute outdoor team event in Mumbai..."*

The LLM resolves "next week" against today's date (injected at runtime), then calls `get_weather("Mumbai")` and `check_calendar(date, 60)` for the five weekdays — often in a single parallel turn. It combines the results to recommend a day. If `get_weather` times out, the retry helper kicks in.

### Goal 2 — Parallel tools + arithmetic
> *"I want to buy 3 solar panels and 2 battery packs. What is the total cost, and is there a free 30-minute slot this week?"*

In a single turn the LLM requests `get_price("solar panel", 3)`, `get_price("battery pack", 2)`, and several `check_calendar` calls in parallel. It performs the addition itself (no math tool) and reports the total + an available slot.

### Goal 3 — Conditional branching
> *"What's the weather in Delhi today? If rainy or windy, find the next available weekday slot for a 90-minute indoor meeting instead."*

The LLM calls `get_weather("Delhi")` first. If the condition is `rainy` or `windy`, it follows up with `check_calendar` calls for upcoming weekdays. If `get_weather` returns an error after retries, the agent answers with what it has rather than crashing.

## Failure handling

| Tool | Failure mode | Strategy |
|---|---|---|
| `get_weather()` | Raises ~20% of calls | Retry up to 3 times in `_call_weather_with_retry`; if all fail, return `{error: ...}` to the LLM and record the error |
| `get_price()` | Returns `{error, available_items}` for unknown items | Forward the dict to the LLM as-is; the system prompt instructs it to pick a substitute or report the failure |
| `check_calendar()` | Never raises | Return as-is; the LLM checks the `available` field |
| LLM call | Network / API error | Caught at the loop level; agent returns an `AgentResult` with the error logged instead of crashing |

The `MAX_TURNS = 12` cap stops the loop from running forever in case the model gets stuck.

## File walkthrough — `agent.py`

| Section | What it does |
|---|---|
| Imports + `load_dotenv()` | Loads `GROQ_API_KEY` from `.env` before constructing the client |
| `AgentResult` dataclass | The return contract `main.py` expects |
| Groq client + constants | `MODEL = "llama-3.3-70b-versatile"`, `MAX_TURNS = 12` |
| `TOOL_SCHEMAS` | JSON Schema descriptions of the 3 tools — what the LLM "sees" |
| `_call_weather_with_retry` | 3-attempt retry wrapper for `get_weather` |
| `_execute_tool` | Dispatches the LLM's tool call to the real Python function |
| `SYSTEM_PROMPT` | Generic agent instructions; today's date is injected at runtime so relative dates resolve dynamically |
| `run_agent(goal)` | The loop: ask LLM → if `tool_calls`, run them and feed results back as `role: "tool"` messages → if no `tool_calls`, return the answer |

## Sample run

```
>>> Running: Goal 1 — Multi-step planning

[agent] goal: Find the cheapest day next week (Mon–Fri) to schedule ...
[agent] --- turn 1 ---
[agent] -> get_weather({'city': 'Mumbai'})
  [retry] get_weather('Mumbai') attempt 1/3 failed: WeatherAPIError: Request timed out
[agent] <- {'city': 'Mumbai', 'condition': 'sunny', 'temp_c': 32.1, 'humidity_pct': 60}
[agent] -> check_calendar({'date': '2026-05-04', 'duration_minutes': 60})
[agent] <- {'date': '2026-05-04', 'available': True, 'reason': None}
... (parallel calls for 2026-05-05 ... 2026-05-08) ...
[agent] --- turn 2 ---
[agent] final answer ready

Final answer:
  Monday 2026-05-04 is the best day — sunny weather and the calendar slot is available.
```

## Notes for reviewers

- The system prompt is fully generic — no hardcoded city, item, or date strings. The same agent will handle any goal that fits the three tools.
- `MAX_TURNS = 12` and `temperature = 0.2` give consistent results across runs while still allowing enough turns for the longest goal.
- All errors (weather retries, unknown items, LLM exceptions) are recorded in `AgentResult.errors_encountered` so the runner can audit what went wrong.
