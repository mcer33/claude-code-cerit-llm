#!/usr/bin/env python3
"""Idle-stop reproducer for CERIT LLMs.

Demonstrates the baseline idle-stop rate (model outputs text instead of
calling a tool) and the effect of two mitigations: system-prompt injection
and tool_choice enforcement.

Three conditions:
  A. baseline_no_continuation — plain system prompt, no tool_choice
     → reproduces real Claude Code scenario (no special hinting)
  B. with_continuation        — + agentic system-prompt block, no tool_choice
     → our client-side proxy fix
  C. tool_choice_any          — plain system prompt + tool_choice: {type: any}
     → server-side forcing: proves model CAN call tools; confirms the
       baseline problem is a model-side choice, not a broken API call

Usage:
    python3 cerit_idle_stop_reproducer.py [--model MODEL] [--n N]

Output: JSON summary suitable for sending to CERIT admin.
"""
from __future__ import annotations
import argparse
import json
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

# Deliberately explicit: names the tool, says "using the list_files tool".
# A properly-behaving agentic model should call the tool on every trial.
PROMPT = "List all .py files in /tmp using the list_files tool."

SYSTEM_BARE = "You are a helpful assistant."

CONTINUATION = (
    "\n\nAGENTIC EXECUTION — NON-NEGOTIABLE\n"
    "When tools are available: call a tool immediately. Do not explain. Do not narrate.\n"
    "FORBIDDEN before a tool call: 'I will now' / 'Let me' / 'I'll first'\n"
    "end_turn is ONLY valid: (a) task 100% done — write your FULL final answer now,\n"
    "  OR (b) you need information only the human can provide.\n"
    "All other turns: CALL A TOOL. No exceptions.\n"
)


def call_api(token: str, model: str, system: str, tool_choice: dict | None = None) -> dict:
    body: dict = {
        "model": model,
        "max_tokens": 512,
        "system": system,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": [TOOL_DEF],
    }
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{CERIT_API}/v1/messages",
        data=payload,
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
    if stop == "tool_use":
        return "TOOL_CALL"
    return f"OTHER({stop})"


CONDITIONS = [
    # (label, system_prompt, tool_choice)
    ("A_baseline_no_continuation",
     SYSTEM_BARE, None),
    ("B_with_continuation",
     SYSTEM_BARE + CONTINUATION, None),
    ("C_tool_choice_any",
     SYSTEM_BARE, {"type": "any"}),
]


def main():
    parser = argparse.ArgumentParser(description="CERIT idle-stop baseline reproducer")
    parser.add_argument("--model", default="qwen3.5-122b",
                        help="CERIT model to test (default: qwen3.5-122b)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of trials per condition (default: 5)")
    parser.add_argument("--conditions", default="A,B,C",
                        help="Comma-separated list of conditions to run: A,B,C (default: all)")
    args = parser.parse_args()

    wanted = set(args.conditions.upper().split(","))
    token = TOKEN_FILE.read_text().strip()
    results = {"model": args.model, "n_per_condition": args.n, "conditions": {}}

    for label, system, tool_choice in CONDITIONS:
        if label[0] not in wanted:
            continue
        outcomes = []
        tc_note = f"tool_choice={json.dumps(tool_choice)}" if tool_choice else "tool_choice=auto"
        print(f"\n[{label}] {tc_note} — {args.n} trials...", flush=True)
        for i in range(args.n):
            r = call_api(token, args.model, system, tool_choice)
            outcome = classify(r)
            outcomes.append(outcome)
            print(f"  trial {i+1}/{args.n}: {outcome}", flush=True)

        n_tool = outcomes.count("TOOL_CALL")
        n_idle = outcomes.count("IDLE_STOP")
        n_err = sum(1 for o in outcomes if o.startswith("ERROR"))
        results["conditions"][label] = {
            "system_prompt_length": len(system),
            "tool_choice": tool_choice,
            "outcomes": outcomes,
            "tool_call_rate": round(n_tool / args.n, 2),
            "idle_stop_rate": round(n_idle / args.n, 2),
            "error_rate": round(n_err / args.n, 2),
        }

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model} | N per condition: {args.n}")
    print()
    for label, stats in results["conditions"].items():
        tc = json.dumps(stats["tool_choice"]) if stats["tool_choice"] else "auto"
        print(f"[{label}] tool_choice={tc}")
        print(f"  tool_call_rate : {stats['tool_call_rate']:.0%}")
        print(f"  idle_stop_rate : {stats['idle_stop_rate']:.0%}")
        print(f"  outcomes       : {stats['outcomes']}")
        print()

    print("Interpretation:")
    conds = results["conditions"]
    a_idle = conds.get("A_baseline_no_continuation", {}).get("idle_stop_rate")
    b_idle = conds.get("B_with_continuation", {}).get("idle_stop_rate")
    c_idle = conds.get("C_tool_choice_any", {}).get("idle_stop_rate")
    if a_idle is not None:
        print(f"  A baseline idle-stop: {a_idle:.0%} — this is what Claude Code sees")
    if c_idle is not None:
        note = ("✓ model CAN call tools; idle-stop is a model-side choice, not a broken call"
                if c_idle < (a_idle or 1) else "tool_choice:any did not help — may be unsupported")
        print(f"  C tool_choice:any idle-stop: {c_idle:.0%} — {note}")
    if b_idle is not None:
        print(f"  B continuation idle-stop: {b_idle:.0%} — client-side proxy fix")

    out = Path("/tmp/cerit_idle_stop_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    main()
