"""
Routing-accuracy eval harness (extension E7).

Hits the live HTTP API with a small fixture of (question, expected_route)
pairs and reports overall + per-route accuracy. Requires the server to
be running and a real GOOGLE_API_KEY (the agent will be invoked).

Usage:
    # in one terminal
    uv run uvicorn app.main:app
    # in another
    uv run python eval/run_eval.py
"""
from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass

import httpx

BASE_URL = os.getenv("HELIX_BASE_URL", "http://127.0.0.1:8000")
USER_ID = "u_eval_runner"
PLAN_TIER = "pro"


@dataclass
class EvalCase:
    question: str
    expected: str  # one of: knowledge, account, escalation, smalltalk


CASES: list[EvalCase] = [
    EvalCase("How do I rotate a deploy key?", "knowledge"),
    EvalCase("What does the secret-scanning feature detect?", "knowledge"),
    EvalCase("Show me my last 3 builds.", "account"),
    EvalCase("What's my current plan and storage usage?", "account"),
    EvalCase("Open a ticket for me — production is down.", "escalation"),
    EvalCase("Please escalate this to a human.", "escalation"),
    EvalCase("Hi there!", "smalltalk"),
    EvalCase("Thanks for your help.", "smalltalk"),
]


async def run_one(client: httpx.AsyncClient, session_id: str, case: EvalCase) -> str:
    resp = await client.post(
        f"/v1/chat/{session_id}", json={"content": case.question}, timeout=60.0
    )
    resp.raise_for_status()
    return resp.json()["routed_to"]


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # Each case gets its own session so context bleed doesn't pollute routing.
        correct = 0
        per_route_correct: dict[str, int] = defaultdict(int)
        per_route_total: dict[str, int] = defaultdict(int)

        for case in CASES:
            sess = await client.post(
                "/v1/sessions", json={"user_id": USER_ID, "plan_tier": PLAN_TIER}
            )
            sess.raise_for_status()
            session_id = sess.json()["session_id"]

            try:
                actual = await run_one(client, session_id, case)
            except Exception as exc:
                actual = f"ERROR({exc.__class__.__name__})"

            ok = actual == case.expected
            correct += int(ok)
            per_route_total[case.expected] += 1
            per_route_correct[case.expected] += int(ok)
            mark = "✓" if ok else "✗"
            print(f"  {mark} {case.expected:12s} → {actual:12s}  | {case.question}")

        print()
        print(f"Overall: {correct}/{len(CASES)} = {correct / len(CASES):.0%}")
        for route in sorted(per_route_total):
            t = per_route_total[route]
            c = per_route_correct[route]
            print(f"  {route:12s}: {c}/{t} = {c / t:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
