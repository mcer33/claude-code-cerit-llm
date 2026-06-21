#!/usr/bin/env python3
"""
Rescore existing run results without re-running tests.

Usage:
    python3 rescore.py [--force] <results_dir> [<results_dir2> ...]

Reads <tid>_output.txt and <tid>_events.jsonl from each results dir,
extracts file-write artifacts from the events (so the judge sees what
was actually written to /tmp/), calls the CERIT judge (qwen3.5-122b),
then updates metrics.json and summary.txt in-place.

--force : re-score entries that already have a quality_score
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
    "10=perfect and complete. "
    "IMPORTANT: The assistant may have written files to disk using tools. "
    "Any [ARTIFACT] or [BASH] sections below the main output show what was "
    "actually written/executed — treat these as evidence of correct completion. "
    "Return ONLY valid JSON: "
    '{"score": <int 1-10>, "rationale": "<one sentence>", "issues": ["<issue1>", ...]}'
)
JUDGE_USER_TMPL = (
    "Task prompt:\n{prompt}\n\n"
    "Assistant output and tool artifacts (truncated):\n{output}\n\n"
    "Score this 1-10."
)

# N-suite prompts
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
    # Extended N-suite (N11-N42)
    "N11": "Read cerit-rewrite-proxy.py. List every module-level constant (ALL_CAPS name assigned at top level). For each: name, value, one-sentence purpose. Include at minimum the port, token file path, API endpoint, and any numeric limits.",
    "N12": "Read cerit-rewrite-proxy.py. Find every HTTP status code (4xx/5xx) handled specially. For each: status code, the function that handles it, and what the proxy does (retry, fallback, error response, etc.).",
    "N13": "Read cerit-rewrite-proxy.py. The proxy uses ThreadingHTTPServer. List every module-level mutable variable read or written by request-handler code. For each: is it protected by a lock? State SAFE or RACE RISK with justification.",
    "N14": "Write /tmp/comp_token_check.py that reads ~/.config/cerit/token, checks it is non-empty and at least 20 chars, and prints 'TOKEN OK (<N> chars)' or 'TOKEN MISSING' / 'TOKEN TOO SHORT'. Then run it.",
    "N15": "Write /tmp/comp_metrics_reader.py: accepts a path to a metrics.json as sys.argv[1] and prints: total tests, completed count, completion %, and avg quality_score (skip None; print N/A if none). Write the file and print its content.",
    "N16": "In rescore.py, find the JUDGE_MODEL constant. Change it to 'qwen3-coder'. Confirm by reading the line. Then revert to the original value and confirm the revert.",
    "N17": "For every file in the current directory (not recursive), show: filename, size in bytes, last-modified timestamp (YYYY-MM-DD HH:MM). Sort by modification time, newest first.",
    "N18": "For each .py file, count: (a) total lines, (b) blank lines, (c) comment-only lines (stripped line starts with #), (d) code lines = total - blank - comment. Show as a table.",
    "N19": "Write /tmp/comp_api_check.sh that curls https://llm.ai.e-infra.cz/v1/models with a 5-second timeout and prints 'CERIT API UP (HTTP <code>)' for 2xx or 'CERIT API DOWN: <code>' otherwise. Run it.",
    "N20": "Read cerit-rewrite-proxy.py. Find the 5 longest functions by line count. Report: function name, start line, line count.",
    "N21": "Read run_tests.py. Trace the execution path from script entry to when one test result is written to metrics.json. List as numbered steps covering: arg parsing, test pool selection, env construction, subprocess launch, output capture, metrics save.",
    "N22": "Read run_tests.py. Find MODEL_PRESETS. For each preset: name, ANTHROPIC_MODEL value, whether it uses proxy (has ANTHROPIC_BASE_URL), and CLAUDE_CODE_MAX_CONTEXT_TOKENS. Count proxy vs native presets. Show as table.",
    "N23": "Search all .py files for hardcoded numeric timeout/sleep values. For each: file, line number, value in seconds, what is being timed out.",
    "N24": "Read cerit-rewrite-proxy.py. Find every location where the proxy injects text into a system prompt or message list. For each: line range, what is injected, and condition that triggers it.",
    "N25": "Write /tmp/comp_test_stats.py: unittest file testing compute_stats(times) -> dict with keys mean, median, std_dev, min, max. Include 4 test cases: normal list, single element, all-same, empty list. Print the file contents.",
    "N26": "Read cerit-rewrite-proxy.py (at minimum the module docstring and do_POST handler). Write 3 paragraphs in plain English explaining what the proxy does and why it exists, targeting a developer unfamiliar with CERIT or Claude Code.",
    "N27": "Read cerit_prompts.py fully, then cerit-rewrite-proxy.py. For each name from cerit_prompts.py: imported by proxy? Used in which function(s)? Format: constant_name | imported | used_in_function | purpose.",
    "N28": "Run git log to show the last 5 commits. For each: short hash, date (YYYY-MM-DD), first line of message. Then run git diff --stat on the most recent commit to show changed files.",
    "N29": "In cerit-rewrite-proxy.py, find the line defining the proxy port (9999). Add inline comment '# override with PROXY_PORT env var'. Read the line to confirm. Then revert and confirm.",
    "N30": "Read install.sh. List every source file it installs and its destination path. Check whether each source file exists in the current directory. Report: source | destination | source_exists (Y/N).",
    "N31": "Read cerit-rewrite-proxy.py tool sanitizer. If a request has both 'web_search_20250305' and 'computer_20241022' tools, trace sanitize_tool_definitions() step by step for each and state the final tools list sent upstream.",
    "N32": "Read cerit-rewrite-proxy.py. If ~/.config/cerit/token exists but is empty, what happens at (a) proxy startup and (b) when the first request arrives? Quote the relevant lines and state crash/silent-fail/error-response.",
    "N33b": "Read README.md. Summarize in exactly 5 bullet points (one per major section): section title + one sentence of content. If no README.md exists, state that and describe the project from .py file docstrings.",
    "N34": "Using the benchmark data (GLM: 122.9s total, 28 proxy injections; native Sonnet: 149.4s total, 0 injections, both 10/10): (1) avg seconds per task each, (2) proxy injection overhead per injection, (3) is GLM faster or slower than native net of injections? Show arithmetic.",
    "N35": "Search all .py files for retry loops. For each: file, line range, what is being retried, max attempts, backoff strategy (fixed/exponential/none).",
    "N36": "Read run_tests.py. Find ALL_TESTS. For each unique 'category' field value, count tests and give one example test name. Show as table: category | count | example. Report which category has the longest average timeout.",
    "N37": "Read cerit_idle_stop_reproducer.py and cerit-rewrite-proxy.py. Explain in 4 bullets why an LLM might produce an idle stop. Then explain how the proxy's continuation injection prevents this, citing specific code.",
    "N38": "In rescore.py, add verbose: bool = True parameter to rescore_dir(). Add one line 'if verbose: print(f\"[verbose] {tid}\")' before the judge_output() call. Confirm both changes. Then revert.",
    "N39": "Read every .py file in the current directory. List every function definition (def ...): file, name, line number, one-sentence purpose. Sort by file then line number.",
    "N40": "Read module docstrings of cerit-rewrite-proxy.py and run_tests.py plus any README. Write a 300-400 word technical summary: what problem it solves, how the proxy works, how the benchmark measures quality. Include at least 2 concrete numbers.",
    "N41": "Run: python3 -c \"import cerit_prompts; names=[x for x in dir(cerit_prompts) if not x.startswith('_')]; print('\\n'.join(names))\". List each exported name and classify as: string constant, dict, list, or other.",
    "N42": "Run: python3 -m py_compile rescore.py && echo 'SYNTAX OK' || echo 'SYNTAX ERROR'. Then same for run_tests.py. Report result for each file.",
}


def extract_artifacts_from_events(events_file: Path) -> str:
    """
    Scan stream-json JSONL for Write/Edit/Bash tool calls and return
    a compact summary of what was actually written or executed.
    This lets the judge see file content even when the model didn't echo it.
    """
    if not events_file.exists():
        return ""
    artifacts = []
    try:
        events = []
        for line in events_file.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue

        # Build tool_result lookup (tool_use_id -> result text)
        tool_results: dict[str, str] = {}
        for e in events:
            if e.get("type") != "user":
                continue
            for block in (e.get("message", {}).get("content") or []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    uid = block.get("tool_use_id", "")
                    cnt = block.get("content", "")
                    if isinstance(cnt, list):
                        cnt = "\n".join(
                            b.get("text", "") for b in cnt if isinstance(b, dict)
                        )
                    tool_results[uid] = str(cnt)

        # Walk assistant turns for tool_use blocks
        for e in events:
            if e.get("type") != "assistant":
                continue
            for block in (e.get("message", {}).get("content") or []):
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                inp = block.get("input") or {}
                uid = block.get("id", "")

                if name == "Write":
                    path = inp.get("file_path", "")
                    content = inp.get("content", "")
                    if path and content:
                        artifacts.append(
                            f"[ARTIFACT written] {path}:\n```\n{content[:2000]}\n```"
                        )

                elif name == "Edit":
                    path = inp.get("file_path", "")
                    old = inp.get("old_string", "")
                    new = inp.get("new_string", "")
                    if path:
                        artifacts.append(
                            f"[ARTIFACT edited] {path}:\n"
                            f"  old: {repr(old[:120])}\n"
                            f"  new: {repr(new[:120])}"
                        )

                elif name == "Bash":
                    cmd = inp.get("command", "")
                    result = tool_results.get(uid, "")
                    # Include bash results for script execution or /tmp operations
                    if result and ("/tmp/" in cmd or "port" in cmd.lower()
                                   or "python" in cmd.lower() or "bash" in cmd.lower()):
                        artifacts.append(
                            f"[BASH] $ {cmd[:200]}\n{result[:500]}"
                        )

    except Exception:
        pass

    return "\n\n".join(artifacts)


def judge_output(prompt: str, output: str, artifacts: str, token: str) -> dict | None:
    if not output.strip() and not artifacts.strip():
        return {"score": 1, "rationale": "No output produced.", "issues": ["Empty output"]}

    full = output
    if artifacts:
        full = output + "\n\n---\n" + artifacts

    user_msg = JUDGE_USER_TMPL.format(prompt=prompt[:500], output=full[:4000])
    body = json.dumps({
        "model": JUDGE_MODEL,
        "max_tokens": 300,
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
        text = "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if not text:
            text = str(data)
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Extract first {...} with "score" if model emits trailing text
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


def rebuild_summary(results_dir: Path, all_metrics: list[dict]) -> None:
    per_test = [m for m in all_metrics if m.get("test_id")]
    scores = [m["quality_score"] for m in per_test if m.get("quality_score") is not None]
    avg_q = round(sum(scores) / len(scores), 2) if scores else None
    n_done = sum(1 for m in per_test if m.get("completed"))
    proxy_inj = sum(
        (m.get("proxy_events") or {}).get("continuation_injections", 0)
        for m in per_test
    )
    run_id = results_dir.name
    header = f"CERIT Workflow Test Suite — {run_id}\n"
    col = (
        f"{'ID':<6}  {'Cat':<3} {'Name':<38} {'Time':>6} {'Turns':>5} "
        f"{'Tools':>5} {'Idle':>5} {'CtxX':>6}  {'Q':>4}  {'Done':>4}\n"
    )
    sep = "-" * 86 + "\n"
    rows = header + col + sep
    for m in per_test:
        q = m.get("quality_score")
        q_str = f"{q:4d}" if q is not None else "   -"
        rows += (
            f"{m['test_id']:<6}  {m.get('category','?'):<3} "
            f"{m.get('test_name','')[:38]:<38} "
            f"{m.get('wall_time_sec', 0):6.1f} {m.get('n_turns', 0):5d} "
            f"{m.get('n_tool_calls', 0):5d} "
            f"{m.get('idle_stop_rate', 0.0)*100:4.0f}% "
            f"{m.get('context_growth_factor', 1.0):6.2f}  "
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
    print(
        f"  Updated summary.txt  avg_quality={avg_q}  "
        f"scored={len(scores)}/{len(per_test)}",
        flush=True,
    )


def rescore_dir(results_dir: Path, token: str, force: bool = False) -> None:
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
        events_file = results_dir / f"{tid}_events.jsonl"
        if not output_file.exists():
            print(f"  [{tid}] no output file, skipping", flush=True)
            continue
        if m.get("quality_score") is not None and not force:
            print(f"  [{tid}] already scored: {m['quality_score']}", flush=True)
            continue
        prompt = N_PROMPTS.get(tid, "")
        if not prompt:
            print(f"  [{tid}] unknown prompt, skipping", flush=True)
            continue

        output = output_file.read_text(errors="replace")
        artifacts = extract_artifacts_from_events(events_file)
        artifact_note = f" (+{len(artifacts)} artifact chars)" if artifacts else ""
        print(f"  [{tid}] judging{artifact_note}...", end=" ", flush=True)

        result = judge_output(prompt, output, artifacts, token)
        if result:
            m["quality_score"] = result["score"]
            m["quality_rationale"] = result["rationale"]
            m["quality_issues"] = result.get("issues", [])
            print(f"score={result['score']}  {result['rationale'][:70]}", flush=True)
            changed = True
        else:
            print("failed", flush=True)

    if not changed:
        print("  nothing to update", flush=True)
        return

    metrics_file.write_text(json.dumps(all_metrics, indent=2))
    rebuild_summary(results_dir, all_metrics)


def main():
    args = sys.argv[1:]
    force = "--force" in args
    dirs = [a for a in args if not a.startswith("--")]

    if not dirs:
        print("Usage: python3 rescore.py [--force] <results_dir> [...]")
        sys.exit(1)

    if not CERIT_TOKEN_FILE.exists():
        sys.exit(f"[ERROR] Missing CERIT token at {CERIT_TOKEN_FILE}")
    token = CERIT_TOKEN_FILE.read_text().strip()

    for arg in dirs:
        d = Path(arg)
        if not d.is_dir():
            print(f"[SKIP] not a directory: {d}", flush=True)
            continue
        print(f"\n=== {d.name} ===", flush=True)
        rescore_dir(d, token, force=force)


if __name__ == "__main__":
    main()
