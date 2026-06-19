#!/usr/bin/env python3
"""
CERIT Claude Code Workflow Test Runner — metrics-capturing unattended suite.

Uses --output-format stream-json to parse tool calls, stop reasons, and token
usage directly from stdout. No JSONL hunting needed.

Captures per-test:
  wall_time_sec, n_turns, n_tool_calls, n_idle_stops, idle_stop_rate
  input_tokens (initial/final/growth), stop_reason distribution
  tool_name frequency, proxy events, task output, completion heuristic

Results written to ~/dev/cerit-tests/results/run_<ISO>_<model>/
  metrics.json     — per-test structured metrics + aggregate
  summary.txt      — human-readable comparison table
  T<n>_output.txt  — raw text output per test (for manual quality scoring)
  T<n>_events.jsonl — raw stream-json events per test

Usage:
    python3 run_tests.py [--model rich|medium|long] [--tests T1,T3,T5]
"""

from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

HOME = Path.home()
CERIT_TOKEN_FILE = HOME / ".config/cerit/token"
PROXY_LOG = Path("/tmp/cerit-rewrite-proxy.log")
RESULTS_BASE = HOME / "dev/cerit-tests/results"

MODEL_PRESETS = {
    "rich": {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999/",
        "ANTHROPIC_MODEL": "claude-sonnet-4-6",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-sonnet-4-6",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-6",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "110000",
    },
    "medium": {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999/",
        "ANTHROPIC_MODEL": "claude-cerit-medium",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-cerit-medium",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-cerit-medium",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-cerit-medium",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "230000",
    },
    "long": {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999/",
        "ANTHROPIC_MODEL": "claude-cerit-long",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-cerit-long",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-cerit-long",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-cerit-long",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "270000",
    },
}
COMMON_ENV = {
    "MAX_THINKING_TOKENS": "0",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}

# ── Test definitions ───────────────────────────────────────────────────────────

TESTS = [
    {
        "id": "T1",
        "name": "A2507 tools catalogue",
        "cwd": str(HOME / "dev/protein_efield_toy_model/tools"),
        "prompt": (
            "List every Python file in the current directory. "
            "For each file: read it, find the module docstring or first comment block, "
            "count lines of code, and note whether it has a "
            "'if __name__ == \"__main__\":' entry point. "
            "Produce a markdown table: filename | lines | has_main | one-sentence purpose. "
            "Read ALL files completely before responding. Do not stop to ask."
        ),
        "min_tools": 5,
        "success_re": r"(filename|has.main|__main__|\.py.*\|)",
        "timeout": 900,
    },
    {
        "id": "T2",
        "name": "PBS script generation",
        "cwd": str(HOME / "dev/protein_efield_toy_model"),
        "prompt": (
            "Write a complete MetaCentrum PBS job script for a 100 ns GROMACS MD production run. "
            "Requirements exactly: "
            "GPU node (1x A100), 8 CPU cores, 24h walltime, queue=gpu. "
            "SCRATCHDIR staging with cp -rL (not cp -r) for topology/FF dirs. "
            "Explicit -nsteps: calculate 100ns / dt=0.002ps. "
            "Checkpoint -cpo every 15000 steps. "
            "Module: GROMACS/2024.3-foss-2023a-CUDA-12.1.1. "
            "rsync results back after run. "
            "Write the script to /tmp/cerit_test_run.pbs and print its full content."
        ),
        "min_tools": 2,
        "success_re": r"(cp -rL|SCRATCHDIR|-nsteps|50000000|#PBS|walltime)",
        "timeout": 540,
    },
    {
        "id": "T3",
        "name": "Zotero agent code review",
        "cwd": str(HOME / ".zotero-lit-agent"),
        "prompt": (
            "Read agent.py and any other .py files in the current directory. "
            "Review for robustness issues in these 4 specific areas: "
            "(1) HTTP requests: are all calls to external APIs given explicit timeouts? "
            "(2) Zotero local API (localhost:23119): what happens if it is down — "
            "crash or graceful retry? "
            "(3) Any hardcoded absolute paths or usernames that would break on another machine. "
            "(4) Error handling: are exceptions caught and logged, or will they silently stop the agent? "
            "Report each issue with file:line and a one-line suggested fix."
        ),
        "min_tools": 3,
        "success_re": r"(line \d+|timeout|localhost|23119|exception|error)",
        "timeout": 240,
    },
    {
        "id": "T4",
        "name": "Memory consistency audit",
        "cwd": str(HOME),
        "prompt": (
            "Read the file .claude/projects/-home-cifra/memory/MEMORY.md. "
            "Find the 5 memory file slugs linked in the TOP-PRIORITY RULES section "
            "(look for lines like [slug](filename.md)). "
            "For each slug: read .claude/projects/-home-cifra/memory/<slug>.md. "
            "Check: (a) does the file exist, (b) does the frontmatter 'description:' field "
            "match the one-line hook in MEMORY.md (roughly), (c) is there a 'type:' field. "
            "Report a table: slug | exists | description_matches | has_type"
        ),
        "min_tools": 6,
        "success_re": r"(slug|exists|description|type|MEMORY|match)",
        "timeout": 600,
    },
    {
        "id": "T5",
        "name": "Proxy robustness review",
        "cwd": str(HOME / "dev"),
        "model": "medium",   # qwen3.5-122b: kimi too slow (~10min) for 600-line file analysis
        "prompt": (
            "Read cerit-rewrite-proxy.py. Assess exactly these 5 robustness points: "
            "1. HTTP 503 from upstream — is it handled or does it propagate as an exception? "
            "2. Thread safety — CERIT_TOKEN is a module global read by ThreadingHTTPServer threads. Safe? "
            "3. Compact-detection regex — give one concrete input string that would be a false positive. "
            "4. Empty or whitespace-only token file — what happens at startup and at request time? "
            "5. SSE chunked streaming — if a chunk boundary splits a multi-byte UTF-8 character, "
            "does _relay_success handle it correctly? "
            "For each: quote the relevant line range and state SAFE / RISK / BUG + one-line fix."
        ),
        "min_tools": 1,
        "success_re": r"(SAFE|RISK|BUG|line \d+|503|thread|token|UTF)",
        "timeout": 300,
    },
    {
        "id": "T6",
        "name": "Script write + self-execute",
        "cwd": "/tmp",
        "prompt": (
            "Write a Python script at /tmp/cerit_session_stats.py that: "
            "reads a Claude Code stream-json JSONL file given as sys.argv[1], "
            "prints: (a) total assistant turns, (b) total tool calls, "
            "(c) top-5 tool names by call count, (d) last input_tokens value found, "
            "(e) count of idle-stop turns (assistant turns with no tool_use block). "
            "After writing: find the most recently modified *.jsonl in "
            "~/.claude/projects/-home-cifra/ (use Bash to find it), "
            "then run: python3 /tmp/cerit_session_stats.py <that_file> "
            "and print the full output."
        ),
        "min_tools": 3,
        "success_re": r"(assistant turns|tool calls|input_tokens|idle|cerit_session_stats)",
        "timeout": 720,
    },
    {
        "id": "T7",
        "name": "Deep multi-file index",
        "cwd": str(HOME / "dev/protein_efield_toy_model/tools"),
        "prompt": (
            "Read every Python file in the current directory. "
            "For each file extract: filename, number of def/class statements, "
            "list of top-level imports, one-sentence description of purpose. "
            "Identify cross-dependencies: which filenames are imported by other files here. "
            "Write a JSON index to /tmp/tools_index.json with structure: "
            "{\"generated\": \"<ISO8601 date>\", "
            "\"files\": [{\"name\": ..., \"n_defs\": ..., \"imports\": [...], "
            "\"purpose\": ..., \"imported_by\": [...]}]}. "
            "Confirm the write by printing the file size in bytes."
        ),
        "min_tools": 8,
        "success_re": r"(tools_index\.json|n_defs|imported_by|bytes|written|\d+ bytes)",
        "timeout": 900,
    },
]

# ── Stream-json parsing ────────────────────────────────────────────────────────

def parse_stream_json(raw: str) -> dict:
    """Parse claude --output-format=stream-json stdout into metrics."""
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue

    raw_asst_turns = [e for e in events if e.get("type") == "assistant"]
    result_ev = next((e for e in events if e.get("type") == "result"), {})

    # Merge events that share the same message.id — stream-json splits a single
    # assistant turn into multiple events when the model produces both a text block
    # and a tool_use block (thinking + action). Merging by ID gives real turn count.
    seen_ids: dict = {}
    merged_turns: list = []
    for t in raw_asst_turns:
        mid = t.get("message", {}).get("id", "")
        if mid and mid in seen_ids:
            # Merge content blocks into the first event for this ID
            existing = seen_ids[mid]
            existing_content = existing.get("message", {}).get("content") or []
            new_content = t.get("message", {}).get("content") or []
            existing.setdefault("message", {})["content"] = existing_content + new_content
        else:
            import copy
            t_copy = copy.deepcopy(t)
            if mid:
                seen_ids[mid] = t_copy
            merged_turns.append(t_copy)
    asst_turns = merged_turns

    # Extract text output (for success check + output file)
    text_output = result_ev.get("result", "")
    if not text_output:
        # Fallback: collect text blocks from assistant turns
        parts = []
        for t in asst_turns:
            content = t.get("message", {}).get("content") or []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
        text_output = "\n".join(parts)

    # Per-turn analysis. Idle-stop = text-only (no tool) in a non-final turn.
    # The last turn is always expected to be a text-only final response — excluded.
    turn_stats = []
    for idx, t in enumerate(asst_turns):
        is_last = (idx == len(asst_turns) - 1)
        content = t.get("message", {}).get("content") or []
        tool_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        usage = t.get("message", {}).get("usage") or {}
        input_tok = (
            (usage.get("input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0)
        )
        turn_stats.append({
            "n_tools": len(tool_blocks),
            "tool_names": [b.get("name", "?") for b in tool_blocks],
            "has_text": bool(text_blocks),
            # Idle-stop: text-only mid-task turn (excludes expected final response)
            "is_idle_stop": len(tool_blocks) == 0 and bool(text_blocks) and not is_last,
            "stop_reason": t.get("message", {}).get("stop_reason", "unknown"),
            "input_tokens": input_tok,
        })

    n_turns = len(turn_stats)
    n_tool_calls = sum(t["n_tools"] for t in turn_stats)
    n_idle_stops = sum(1 for t in turn_stats if t["is_idle_stop"])
    all_tools = [n for t in turn_stats for n in t["tool_names"]]
    tool_freq: dict[str, int] = {}
    for n in all_tools:
        tool_freq[n] = tool_freq.get(n, 0) + 1
    stop_reasons: dict[str, int] = {}
    for t in turn_stats:
        sr = t["stop_reason"]
        stop_reasons[sr] = stop_reasons.get(sr, 0) + 1
    tok_series = [t["input_tokens"] for t in turn_stats if t["input_tokens"] > 0]

    # Cost / duration / TTFT from result event
    sdk_duration_ms = result_ev.get("duration_ms") or result_ev.get("duration_api_ms")
    ttft_ms = result_ev.get("ttft_ms")
    cost_usd = result_ev.get("total_cost_usd") or result_ev.get("cost_usd", 0)
    session_id = result_ev.get("session_id", "")
    # output_tokens IS reported in the result usage block (unlike per-turn usage which CERIT zeros)
    result_usage = result_ev.get("usage") or {}
    output_tokens_total_result = result_usage.get("output_tokens") or 0

    return {
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "n_idle_stops": n_idle_stops,
        "idle_stop_rate": round(n_idle_stops / n_turns, 3) if n_turns else 0,
        "tool_freq": dict(sorted(tool_freq.items(), key=lambda x: -x[1])[:10]),
        "stop_reasons": stop_reasons,
        "input_tokens_initial": tok_series[0] if tok_series else 0,
        "input_tokens_final": tok_series[-1] if tok_series else 0,
        "context_growth_factor": round(tok_series[-1] / tok_series[0], 2)
            if len(tok_series) >= 2 and tok_series[0] else 1.0,
        "sdk_duration_ms": sdk_duration_ms,
        "ttft_ms": ttft_ms,
        "cost_usd_proxy": cost_usd,
        "output_tokens_total": output_tokens_total_result,
        "session_id": session_id,
        "n_raw_events": len(events),
        "text_output": text_output,
    }


def parse_proxy_log_delta(start_offset: int) -> dict:
    events = {
        "continuation_injections": 0,
        "max_tokens_boosts": 0,
        "model_rewrites": 0,
        "fallbacks": 0,
        "compact_upgrades": 0,
        "upstream_errors": 0,
    }
    try:
        with open(PROXY_LOG) as f:
            f.seek(start_offset)
            for line in f:
                if "injected continuation rule" in line:
                    events["continuation_injections"] += 1
                elif "boosted max_tokens" in line:
                    events["max_tokens_boosts"] += 1
                elif "rewrite model" in line:
                    events["model_rewrites"] += 1
                elif "auto-fallback to" in line:
                    events["fallbacks"] += 1
                elif "compact detected" in line:
                    events["compact_upgrades"] += 1
                elif re.search(r"upstream [45]\d\d", line):
                    events["upstream_errors"] += 1
    except Exception:
        pass
    return events


def proxy_log_offset() -> int:
    try:
        return PROXY_LOG.stat().st_size
    except Exception:
        return 0


# ── Runner ─────────────────────────────────────────────────────────────────────

def build_env(preset: str, token: str) -> dict:
    env = dict(os.environ)
    env.update(MODEL_PRESETS[preset])
    env.update(COMMON_ENV)
    env["ANTHROPIC_API_KEY"] = token
    env["ANTHROPIC_AUTH_TOKEN"] = token
    return env


def run_test(test: dict, env: dict, results_dir: Path, token: str) -> dict:
    tid = test["id"]
    cwd = test["cwd"]
    # Per-test model override (e.g. T5 uses medium/qwen3.5-122b for speed)
    test_model = test.get("model")
    if test_model and test_model in MODEL_PRESETS:
        env = dict(env)
        env.update(MODEL_PRESETS[test_model])
        env["ANTHROPIC_API_KEY"] = token
        env["ANTHROPIC_AUTH_TOKEN"] = token
    print(f"\n{'='*64}")
    print(f"  {tid}: {test['name']}  [model={test_model or 'default'}]")
    print(f"  cwd: {cwd}")
    print(f"{'='*64}", flush=True)

    # Fall back to HOME if cwd missing
    if not Path(cwd).exists():
        print(f"  [WARN] cwd not found, using {HOME}")
        cwd = str(HOME)

    proxy_offset = proxy_log_offset()
    t0 = time.time()
    timed_out = False

    try:
        result = subprocess.run(
            ["claude", "--print", "--verbose", "--dangerously-skip-permissions",
         "--output-format", "stream-json", "-p", test["prompt"]],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=test.get("timeout", 300),
        )
        raw_stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired as e:
        # Capture partial output — partial stream-json is still parseable for
        # turns/tools that completed before timeout. e.output is bytes.
        raw_stdout = (e.output or b"").decode("utf-8", errors="replace")
        stderr = "TIMEOUT"
        exit_code = -1
        timed_out = True

    wall_time = round(time.time() - t0, 2)
    print(f"  done in {wall_time}s  exit={exit_code}  timed_out={timed_out}", flush=True)

    # Save raw events JSONL
    (results_dir / f"{tid}_events.jsonl").write_text(raw_stdout)

    # Parse
    stream_metrics = parse_stream_json(raw_stdout)
    proxy_events = parse_proxy_log_delta(proxy_offset)

    text_output = stream_metrics.pop("text_output", "")
    (results_dir / f"{tid}_output.txt").write_text(
        text_output + ("\n\n[STDERR]\n" + stderr if stderr else "")
    )

    # Success heuristic
    pattern = test.get("success_re", ".")
    completed = exit_code == 0 and bool(re.search(pattern, text_output, re.IGNORECASE))

    print(f"  turns={stream_metrics.get('n_turns')}  tools={stream_metrics.get('n_tool_calls')}  "
          f"idle={stream_metrics.get('n_idle_stops')}  completed={completed}", flush=True)
    print(f"  proxy: {proxy_events}", flush=True)

    return {
        "test_id": tid,
        "test_name": test["name"],
        "cwd": cwd,
        "wall_time_sec": wall_time,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "completed": completed,
        "output_chars": len(text_output),
        "stderr_snippet": (stderr or "")[:200],
        "proxy_events": proxy_events,
        **stream_metrics,
    }


# ── Aggregate analysis ─────────────────────────────────────────────────────────

def aggregate(all_metrics: list[dict]) -> dict:
    completed = [m for m in all_metrics if m.get("completed")]
    n = len(all_metrics)
    nc = len(completed)

    def safe_avg(key):
        vals = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    total_proxy_continuations = sum(
        m.get("proxy_events", {}).get("continuation_injections", 0) for m in all_metrics
    )
    total_proxy_fallbacks = sum(
        m.get("proxy_events", {}).get("fallbacks", 0) for m in all_metrics
    )
    total_idle_stops = sum(m.get("n_idle_stops", 0) or 0 for m in all_metrics)
    total_tool_calls = sum(m.get("n_tool_calls", 0) or 0 for m in all_metrics)

    all_tools: dict[str, int] = {}
    for m in all_metrics:
        for k, v in (m.get("tool_freq") or {}).items():
            all_tools[k] = all_tools.get(k, 0) + v

    return {
        "n_tests": n,
        "n_completed": nc,
        "completion_rate": round(nc / n, 2) if n else 0,
        "avg_wall_time_sec": safe_avg("wall_time_sec"),
        "avg_n_turns": safe_avg("n_turns"),
        "avg_n_tool_calls": safe_avg("n_tool_calls"),
        "avg_idle_stop_rate": safe_avg("idle_stop_rate"),
        "avg_context_growth": safe_avg("context_growth_factor"),
        "total_idle_stops": total_idle_stops,
        "total_tool_calls": total_tool_calls,
        "total_proxy_continuations": total_proxy_continuations,
        "total_proxy_fallbacks": total_proxy_fallbacks,
        "global_idle_rate": round(total_idle_stops / (total_idle_stops + total_tool_calls), 3)
            if (total_idle_stops + total_tool_calls) else 0,
        "top_tools_overall": dict(sorted(all_tools.items(), key=lambda x: -x[1])[:10]),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CERIT workflow test runner")
    parser.add_argument("--model", choices=["rich", "medium", "long"], default="rich",
                        help="CERIT model preset (default: rich = gemma4 via proxy)")
    parser.add_argument("--tests", default="",
                        help="Comma-separated test IDs to run, e.g. T1,T3 (default: all)")
    args = parser.parse_args()

    if not CERIT_TOKEN_FILE.exists():
        sys.exit(f"[ERROR] Missing CERIT token at {CERIT_TOKEN_FILE}")
    token = CERIT_TOKEN_FILE.read_text().strip()
    if not token:
        sys.exit("[ERROR] CERIT token file is empty")

    run_ids = set(args.tests.split(",")) if args.tests else None
    tests_to_run = [t for t in TESTS if run_ids is None or t["id"] in (run_ids or set())]

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    results_dir = RESULTS_BASE / f"run_{ts}_{args.model}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID  : {ts}")
    print(f"Model   : {args.model}")
    print(f"Tests   : {[t['id'] for t in tests_to_run]}")
    print(f"Results : {results_dir}")

    env = build_env(args.model, token)
    all_metrics: list[dict] = []

    for test in tests_to_run:
        m = run_test(test, env, results_dir, token)
        all_metrics.append(m)
        # Save incrementally (partial runs are usable)
        (results_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

    agg = aggregate(all_metrics)
    (results_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))

    # Human-readable summary table
    hdr = (f"{'ID':<4} {'Name':<38} {'Time':>6} {'Turns':>5} {'Tools':>5} "
           f"{'Idle':>4} {'Idle%':>5} {'CtxX':>5} {'PxyInj':>6} {'Done':>5}")
    sep = "-" * len(hdr)
    rows = [
        f"CERIT Workflow Test Suite — {ts} — model={args.model}",
        hdr, sep,
    ]
    for m in all_metrics:
        idle_pct = (m.get("idle_stop_rate") or 0) * 100
        rows.append(
            f"{m['test_id']:<4} {m['test_name']:<38} {m['wall_time_sec']:>6.1f} "
            f"{str(m.get('n_turns', '-')):>5} {str(m.get('n_tool_calls', '-')):>5} "
            f"{str(m.get('n_idle_stops', '-')):>4} {idle_pct:>4.0f}% "
            f"{str(m.get('context_growth_factor', '-')):>5} "
            f"{str(m.get('proxy_events', {}).get('continuation_injections', '-')):>6} "
            f"{'Y' if m.get('completed') else 'N':>5}"
        )
    rows += [
        sep,
        f"  Completion: {agg['n_completed']}/{agg['n_tests']}  "
        f"Avg idle-stop rate: {agg['avg_idle_stop_rate']}  "
        f"Total proxy continuations: {agg['total_proxy_continuations']}  "
        f"Fallbacks: {agg['total_proxy_fallbacks']}",
    ]
    summary = "\n".join(rows)
    print("\n" + summary + "\n")
    (results_dir / "summary.txt").write_text(summary + "\n")
    print(f"All results saved to: {results_dir}")


if __name__ == "__main__":
    main()
