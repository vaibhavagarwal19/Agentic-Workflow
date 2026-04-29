"""
main.py — Entry Point
======================
We will run:  python main.py

This file runs your agent against all three goals and prints results.
Do not change the GOALS list or the output structure.
You may add setup/config code at the top (e.g. loading env vars).
"""

import os
import json
from dotenv import load_dotenv
from agent import run_agent, AgentResult

load_dotenv()  # loads OPENAI_API_KEY (or other keys) from .env

# ---------------------------------------------------------------------------
# The three goals — do not modify
# ---------------------------------------------------------------------------

GOALS = [
    (
        "Goal 1 — Multi-step planning",
        "Find the cheapest day next week (Mon–Fri) to schedule a 60-minute outdoor "
        "team event in Mumbai, considering both weather and calendar availability."
    ),
    (
        "Goal 2 — Parallel tools + arithmetic",
        "I want to buy 3 solar panels and 2 battery packs. What is the total cost, "
        "and is there a free 30-minute slot this week to schedule a delivery call?"
    ),
    (
        "Goal 3 — Conditional branching",
        "What's the weather like in Delhi today? If it's rainy or windy, find the "
        "next available weekday slot for a 90-minute indoor meeting instead."
    ),
]


# ---------------------------------------------------------------------------
# Runner — do not modify below this line
# ---------------------------------------------------------------------------

def print_result(label: str, result: AgentResult) -> None:
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"\nGoal:\n  {result.goal}\n")
    print("Steps taken:")
    for i, step in enumerate(result.steps_taken, 1):
        print(f"  {i}. {step}")
    print(f"\nFinal answer:\n  {result.final_answer}")
    if result.errors_encountered:
        print(f"\nErrors encountered:")
        for err in result.errors_encountered:
            print(f"  ! {err}")
    print()


def main():
    print("\n" + "=" * 60)
    print("  AGENTIC WORKFLOW — CANDIDATE SUBMISSION RUNNER")
    print("=" * 60)

    results = []
    for label, goal in GOALS:
        print(f"\n>>> Running: {label}")
        print(f"    {goal[:80]}{'...' if len(goal) > 80 else ''}\n")
        try:
            result = run_agent(goal)
        except NotImplementedError:
            print("  [ERROR] run_agent() is not implemented yet.")
            result = AgentResult(
                goal=goal,
                steps_taken=[],
                final_answer="Not implemented",
                errors_encountered=["NotImplementedError — agent not built yet"],
            )
        print_result(label, result)
        results.append(result)

    # Summary
    completed = sum(1 for r in results if r.final_answer != "Not implemented")
    print(f"\nSummary: {completed}/{len(GOALS)} goals completed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
