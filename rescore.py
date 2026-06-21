#!/usr/bin/env python3
"""
Rescore existing run results without re-running tests.

Usage:
    python3 rescore.py <results_dir> [<results_dir2> ...]

Reads <tid>_output.txt and metrics.json from each results dir,
calls the CERIT judge (qwen3.5-122b) for each test, then updates
metrics.json and summary.txt in-place.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

HOME = Path.home()
CERIT_TOKEN_FILE = HOME / ".config/cerit/token"
CERIT_API = "https://llm.ai.e-infra.cz"
JUDGE_MODEL = "qwen3.5-122b"

JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of AI coding assistant outputs. "
    "Score the response 1-10 where 1=completely wrong/empty, 5=partially correct, "
    "10=perfect and complete. Return ONLY valid JSON: "
    '{"score": <int 1-10>, "rationale": "<one sentence>", "issues": ["<issue1>", ...]}'
)
JUDGE_USER_TMPL = (
    "Task prompt: {prompt}\n\n"
    "Assistant output (truncated to 3000 chars):\n{output}\n\n"
    "Score this output 1-10."
)

# Reconstructed N-suite prompts (needed for judging)
N_PROMPTS = {
    "N01": "Read cerit-rewrite-proxy.py in the current directory. List the 6 proxy interventions described in the module docstring at the top of the file. Number them 1-6 exactly as written.",
    "N02": "Search cerit-rewrite-proxy.py for all sys.stderr.write() calls. Count the total number and list the first 5 unique message strings (strip leading/trailing whitespace from each).",
    "N03": "Write a Python function compute_stats(times: list[float]) -> dict that returns mean, median, std_dev, min, max of the input list. Save it to /tmp/comp_stats.py. Use only stdlib (statistics module is fine).",
    "N04": "Count the lines of code in every .py file in the current directory. Show the result as: filename | line_count, sorted by count descending. Then show most/fewest/total.",
    "N05": "Find the TASK_COMPLETE_MARKER string constant in cerit_prompts.py. Also find the GitHub URL mentioned anywhere in the .py files. Write both to /tmp/comp_extract.txt as: MARKER=<value> and URL=<value>.",
    "N06": "Edit run_tests.py: change the line that sets N_TRIALS to use the value 3 instead of whatever it currently is. Confirm the edit was made. Then revert it back to the original value and confirm the revert.",
    "N07": "List the 3 most recently modified files in the current directory (any file type). For each show: filename, last-modified date (YYYY-MM-DD), and file size in bytes.",
    "N08": "Read run_tests.py. Explain what the run_test() function does in exactly 5 bullet points. Each bullet must be one sentence and start with a dash.",
    "N09": "Write a bash script /tmp/comp_port_check.sh that checks if port 9999 is listening on localhost and prints either 'port 9999 OPEN' or 'port 9999 CLOSED'. Then execute it and show the output.",
    "N10": "Read both cerit-rewrite-proxy.py and cerit_prompts.py. List every name (function, class, or constant) that is defined in cerit_prompts.py and imported or used in cerit-rewrite-proxy.py.",
}


def judge_output(prompt: str, output: str, token: str) -> dict | None:
    if not output or not output.strip():
        return {"score": 1, "rationale": "No output produced.", "issues": ["Empty output"]}
    user_msg = JUDGE_USER_TMPL.format(prompt=prompt[:500], output=output[:3000])
    body = json.dumps({
        "model": JUDGE_MODEL,
        "max_tokens": 256,
        "system": JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }).encode("utf-8")
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        content = data.get("content") or []
        text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        if not text:
            text = str(data)
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        # Robust parse: try full text first, then extract first {...} object
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
            if not m:
                raise
            result = json.loads(m.group())
        score = int(result.get("score", 0))
        if not 1 <= score <= 10:
            raise ValueError(f"score out of range: {score}")
        return {
            "score": score,
            "rationale": str(result.get("rationale", ""))[:200],
            "issues": result.get("issues", []),
        }
    except Exception as e:
        print(f"    [judge error] {e}", flush=True)
        return None


def rescore_dir(results_dir: Path, token: str) -> None:
    metrics_file = results_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"  [SKIP] no metrics.json in {results_dir}", flush=True)
        return

    metrics = json.loads(metrics_file.read_text())
    all_metrics = metrics if isinstance(metrics, list) else [metrics]

    changed = False
    for m in all_metrics:
        tid = m.get("test_id")
        if not tid:
            continue
        output_file = results_dir / f"{tid}_output.txt"
        if not output_file.exists():
            print(f"  [{tid}] no output file, skipping", flush=True)
            continue
        if m.get("quality_score") is not None:
            print(f"  [{tid}] already scored: {m['quality_score']}", flush=True)
            continue
        prompt = N_PROMPTS.get(tid, "")
        if not prompt:
            print(f"  [{tid}] unknown prompt, skipping", flush=True)
            continue
        output = output_file.read_text(errors="replace")
        print(f"  [{tid}] judging...", end=" ", flush=True)
        result = judge_output(prompt, output, token)
        if result:
            m["quality_score"] = result["score"]
            m["quality_rationale"] = result["rationale"]
            m["quality_issues"] = result.get("issues", [])
            print(f"score={result['score']}  {result['rationale'][:60]}", flush=True)
            changed = True
        else:
            print("failed", flush=True)

    if not changed:
        print("  nothing to update", flush=True)
        return

    metrics_file.write_text(json.dumps(all_metrics, indent=2))

    # Rebuild summary table from flat list (no _aggregate entry)
    per_test = [m for m in all_metrics if m.get("test_id")]
    scores = [m["quality_score"] for m in per_test if m.get("quality_score") is not None]
    avg_q = round(sum(scores) / len(scores), 2) if scores else None
    n_done = len([m for m in per_test if m.get("completed")])
    total_tools = sum(m.get("n_tool_calls", 0) for m in per_test)
    proxy_inj = sum((m.get("proxy_events") or {}).get("continuation_injections", 0) for m in per_test)
    run_id = results_dir.name
    header = f"CERIT Workflow Test Suite — {run_id}\n"
    col = f"{'ID':<6}  {'Cat':<3} {'Name':<38} {'Time':>6} {'Turns':>5} {'Tools':>5} {'Idle':>5} {'CtxX':>6}  {'Q':>4}  {'Done':>4}\n"
    sep = "-" * 86 + "\n"
    rows = header + col + sep
    for m in per_test:
        q = m.get("quality_score")
        q_str = f"{q:4d}" if q is not None else "   -"
        rows += (
            f"{m['test_id']:<6}  {m.get('category','?'):<3} {m.get('test_name','')[:38]:<38} "
            f"{m.get('wall_time_sec', 0):6.1f} {m.get('n_turns', 0):5d} {m.get('n_tool_calls', 0):5d} "
            f"{m.get('idle_stop_rate', 0.0)*100:4.0f}% {m.get('context_growth_factor', 1.0):6.2f}  "
            f"{q_str}  {'Y' if m.get('completed') else 'N':>4}\n"
        )
    rows += sep
    rows += (
        f"  Completion: {n_done}/{len(per_test)}  "
        f"Avg quality: {avg_q}  "
        f"Avg idle: 0.0  "
        f"Proxy injections: {proxy_inj}\n"
    )
    (results_dir / "summary.txt").write_text(rows)
    print(f"  Updated summary.txt  avg_quality={avg_q}  scored={len(scores)}/{len(per_test)}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 rescore.py <results_dir> [...]")
        sys.exit(1)

    if not CERIT_TOKEN_FILE.exists():
        sys.exit(f"[ERROR] Missing CERIT token at {CERIT_TOKEN_FILE}")
    token = CERIT_TOKEN_FILE.read_text().strip()

    for arg in sys.argv[1:]:
        d = Path(arg)
        if not d.is_dir():
            print(f"[SKIP] not a directory: {d}", flush=True)
            continue
        print(f"\n=== {d.name} ===", flush=True)
        rescore_dir(d, token)


if __name__ == "__main__":
    main()
