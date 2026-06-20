#!/usr/bin/env python3
"""Idle-stop reproducer for CERIT LLMs.

Demonstrates the baseline idle-stop rate (model outputs text instead of
calling a tool) and the effect of the CERIT_CONTINUATION system prompt.

Usage:
    python3 cerit_idle_stop_reproducer.py [--model MODEL] [--n N]

Output: JSON summary suitable for sending to CERIT admin (Lukáš Hejtmánek).
"""
from __future__ import annotations
import argparse
import json
import sys
import urllib.request
from pathlib import Path

TOKEN_FILE = Path.home() / ".config/cerit/token"
CERIT_API = "https://llm.ai.e-infra.cz"

TOOL_DEF = {
    "name": "list_files",
    "description": "List files in a directory.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory path"}},
        "required": ["path"],
    },
}

PROMPT = "List all .py files in /tmp using the list_files tool."

CONTINUATION = (
    "\n\nAGENTIC EXECUTION — NON-NEGOTIABLE\n"
    "When tools are available: call a tool immediately. Do not explain. Do not narrate.\n"
    "FORBIDDEN before a tool call: 'I will now' / 'Let me' / 'I'll first'\n"
    "end_turn is ONLY valid: (a) task 100% done — write your FULL final answer now,\n"
    "  OR (b) you need information only the human can provide.\n"
    "All other turns: CALL A TOOL. No exceptions.\n"
)


def call_api(token: str, model: str, system: str) -> dict:
    body = json.dumps({
        "model": model,
        "max_tokens": 512,
        "system": system,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": [TOOL_DEF],
    }).encode()
    req = urllib.request.Request(
        f"{CERIT_API}/v1/messages",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def classify(response: dict) -> str:
    if "error" in response:
        return f"ERROR: {response['error']}"
    stop = response.get("stop_reason", "unknown")
    content = response.get("content", [])
    types = [b.get("type") for b in content if isinstance(b, dict)]
    if "tool_use" in types:
        return "TOOL_CALL"
    if stop == "end_turn":
        return "IDLE_STOP"
    return f"OTHER({stop})"


def main():
    parser = argparse.ArgumentParser(description="CERIT idle-stop baseline reproducer")
    parser.add_argument("--model", default="qwen3.5-122b",
                        help="CERIT model to test (default: qwen3.5-122b)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of trials per condition (default: 5)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline (no continuation) — run fixed condition only")
    args = parser.parse_args()

    token = TOKEN_FILE.read_text().strip()
    results = {"model": args.model, "n_per_condition": args.n, "conditions": {}}

    for label, system in [
        ("baseline_no_continuation", "You are a helpful assistant."),
        ("with_continuation", "You are a helpful assistant." + CONTINUATION),
    ]:
        if args.no_baseline and label == "baseline_no_continuation":
            continue
        outcomes = []
        print(f"\n[{label}] running {args.n} trials with {args.model}...", flush=True)
        for i in range(args.n):
            r = call_api(token, args.model, system)
            outcome = classify(r)
            outcomes.append(outcome)
            print(f"  trial {i+1}/{args.n}: {outcome}", flush=True)

        n_tool = outcomes.count("TOOL_CALL")
        n_idle = outcomes.count("IDLE_STOP")
        n_err = sum(1 for o in outcomes if o.startswith("ERROR"))
        results["conditions"][label] = {
            "outcomes": outcomes,
            "tool_call_rate": round(n_tool / args.n, 2),
            "idle_stop_rate": round(n_idle / args.n, 2),
            "error_rate": round(n_err / args.n, 2),
        }

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for label, stats in results["conditions"].items():
        print(f"\n[{label}]")
        print(f"  tool_call_rate : {stats['tool_call_rate']:.0%}")
        print(f"  idle_stop_rate : {stats['idle_stop_rate']:.0%}")
        print(f"  outcomes       : {stats['outcomes']}")

    out = Path("/tmp/cerit_idle_stop_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
