#!/usr/bin/env python3
"""
CERIT Claude Code Workflow Test Runner — metrics-capturing unattended suite.

Uses --output-format stream-json to parse tool calls, stop reasons, and token
usage directly from stdout. No JSONL hunting needed.

Captures per-test:
  wall_time_sec, n_turns, n_tool_calls, n_idle_stops, idle_stop_rate
  input_tokens (initial/final/growth), stop_reason distribution
  tool_name frequency, proxy events, task output, completion heuristic
  quality_score (1-10, LLM judge via CERIT qwen3.5-122b)

Results written to ~/dev/cerit-tests/results/run_<ISO>_<model>/
  metrics.json     — per-test structured metrics + aggregate
  summary.txt      — human-readable comparison table
  T<n>_output.txt  — raw text output per test (for manual quality scoring)
  T<n>_events.jsonl — raw stream-json events per test

Usage:
    python3 run_tests.py [--model rich|medium|long|deep|glm]
                         [--tests T1,T3,A01]
                         [--suite base|ext|all]
                         [--category A,B,C]
                         [--parallel 1|2]
                         [--no-judge]
                         [--timeout-scale FLOAT]

CUSTOMISATION
-------------
The test suite references several directories. Set these environment variables
to point to your own project structure before running, or edit the path
constants in the "Base test suite paths" block below directly:

  CERIT_TOOLS_DIR   — directory of Python scripts to analyse/test
                      (default: ~/dev/my_project/tools)
  CERIT_MD_DIR      — root of your main project/research directory
                      (default: ~/dev/my_project)
  CERIT_ZOTERO_DIR  — a Zotero-based literature agent directory
                      (default: ~/.zotero-lit-agent)
  CERIT_DEV_DIR     — your ~/dev equivalent
                      (default: ~/dev)
  CERIT_MEM_DIR     — Claude Code memory directory
                      (default: ~/.claude/projects/memory)
"""

from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

HOME = Path.home()
CERIT_TOKEN_FILE = HOME / ".config/cerit/token"
ANTHROPIC_KEY_FILE = HOME / ".config/anthropic/ufe_training_key"
PROXY_LOG = Path("/tmp/cerit-rewrite-proxy.log")
RESULTS_BASE = HOME / "dev/cerit-tests/results"
CERIT_API = "https://llm.ai.e-infra.cz"
JUDGE_MODEL = "qwen3.5-122b"   # reliable guaranteed-tier model for scoring

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
    "deep": {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999/",
        "ANTHROPIC_MODEL": "claude-cerit-deep",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-cerit-deep",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-cerit-deep",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-cerit-deep",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "110000",
    },
    "glm": {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999/",
        "ANTHROPIC_MODEL": "claude-cerit-glm",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-cerit-glm",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-cerit-glm",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-cerit-glm",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "110000",
    },
    # Native Anthropic API — no ANTHROPIC_BASE_URL (uses real Anthropic endpoint)
    "native": {
        "ANTHROPIC_MODEL": "claude-sonnet-4-6",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000",
    },
    "haiku": {
        "ANTHROPIC_MODEL": "claude-haiku-4-5-20251001",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "200000",
    },
}
COMMON_ENV = {
    "MAX_THINKING_TOKENS": "0",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}

# ── Base test suite paths (customise via ENV or edit directly) ────────────────

TOOLS_DIR = os.environ.get("CERIT_TOOLS_DIR", str(HOME / "dev/my_project/tools"))
MD_DIR    = os.environ.get("CERIT_MD_DIR",    str(HOME / "dev/my_project"))
ZOTERO    = os.environ.get("CERIT_ZOTERO_DIR", str(HOME / ".zotero-lit-agent"))
DEV       = os.environ.get("CERIT_DEV_DIR",   str(HOME / "dev"))
MEM_DIR   = os.environ.get("CERIT_MEM_DIR",   str(HOME / ".claude/projects/memory"))
# Comparison benchmark: clone of public cerit-llm repo (self-contained, used for native vs CERIT tests)
COMP_DIR  = os.environ.get("CERIT_COMP_DIR",  str(HOME / "dev/cerit-comparison-bench"))

# ── Base test suite (T1–T7) ───────────────────────────────────────────────────

TESTS_BASE = [
    {
        "id": "T1", "suite": "base", "category": "F",
        "name": "Project tools catalogue",
        "cwd": TOOLS_DIR,
        "prompt": (
            "List every Python file in the current directory. "
            "For each file: read it, find the module docstring or first comment block, "
            "count lines of code, and note whether it has a "
            "'if __name__ == \"__main__\":' entry point. "
            "Produce a markdown table: filename | lines | has_main | one-sentence purpose. "
            "Read ALL files completely before responding. Do not stop to ask."
        ),
        "min_tools": 5, "success_re": r"(filename|has.main|__main__|\.py.*\|)", "timeout": 900,
    },
    {
        "id": "T2", "suite": "base", "category": "B",
        "name": "PBS script generation",
        "cwd": MD_DIR,
        "prompt": (
            "Write a complete MetaCentrum PBS job script for a 100 ns GROMACS MD production run. "
            "Requirements exactly: GPU node (1x A100), 8 CPU cores, 24h walltime, queue=gpu. "
            "SCRATCHDIR staging with cp -rL (not cp -r) for topology/FF dirs. "
            "Explicit -nsteps: calculate 100ns / dt=0.002ps. "
            "Checkpoint -cpo every 15000 steps. "
            "Module: GROMACS/2024.3-foss-2023a-CUDA-12.1.1. "
            "rsync results back after run. "
            "Write the script to /tmp/cerit_test_run.pbs and print its full content."
        ),
        "min_tools": 2, "success_re": r"(cp -rL|SCRATCHDIR|-nsteps|50000000|#PBS|walltime)", "timeout": 540,
    },
    {
        "id": "T3", "suite": "base", "category": "A",
        "name": "Zotero agent code review",
        "cwd": ZOTERO,
        "prompt": (
            "Read agent.py and any other .py files in the current directory. "
            "Review for robustness issues in these 4 specific areas: "
            "(1) HTTP requests: are all calls to external APIs given explicit timeouts? "
            "(2) Zotero local API (localhost:23119): what happens if it is down — crash or graceful retry? "
            "(3) Any hardcoded absolute paths or usernames that would break on another machine. "
            "(4) Error handling: are exceptions caught and logged, or will they silently stop the agent? "
            "Report each issue with file:line and a one-line suggested fix."
        ),
        "min_tools": 3, "success_re": r"(line \d+|timeout|localhost|23119|exception|error)", "timeout": 240,
    },
    {
        "id": "T4", "suite": "base", "category": "A",
        "name": "Memory consistency audit",
        "cwd": str(HOME),
        "prompt": (
            "Read the file .claude/projects/memory/MEMORY.md. "
            "Find the 5 memory file slugs linked in the TOP-PRIORITY RULES section. "
            "For each slug: read .claude/projects/memory/<slug>.md. "
            "Check: (a) does the file exist, (b) does the frontmatter 'description:' field "
            "match the one-line hook in MEMORY.md (roughly), (c) is there a 'type:' field. "
            "Report a table: slug | exists | description_matches | has_type"
        ),
        "min_tools": 6, "success_re": r"(slug|exists|description|type|MEMORY|match)", "timeout": 600,
    },
    {
        "id": "T5", "suite": "base", "category": "A",
        "name": "Proxy robustness review",
        "cwd": DEV,
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
        "min_tools": 1, "success_re": r"(SAFE|RISK|BUG|line \d+|503|thread|token|UTF)", "timeout": 300,
    },
    {
        "id": "T6", "suite": "base", "category": "F",
        "name": "Script write + self-execute",
        "cwd": "/tmp",
        "prompt": (
            "Write a Python script at /tmp/cerit_session_stats.py that: "
            "reads a Claude Code stream-json JSONL file given as sys.argv[1], "
            "prints: (a) total assistant turns, (b) total tool calls, "
            "(c) top-5 tool names by call count, (d) last input_tokens value found, "
            "(e) count of idle-stop turns (assistant turns with no tool_use block). "
            "After writing: find the most recently modified *.jsonl in "
            "~/.claude/projects/ (use Bash to find it), "
            "then run: python3 /tmp/cerit_session_stats.py <that_file> "
            "and print the full output."
        ),
        "min_tools": 3, "success_re": r"(assistant turns|tool calls|input_tokens|idle|cerit_session_stats)", "timeout": 720,
    },
    {
        "id": "T7", "suite": "base", "category": "F",
        "name": "Deep multi-file index",
        "cwd": TOOLS_DIR,
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
        "min_tools": 8, "success_re": r"(tools_index\.json|n_defs|imported_by|bytes|written|\d+ bytes)", "timeout": 900,
    },
]

# ── Extended test suite (A01–K05, 110 tests) ──────────────────────────────────

TESTS_EXT = [
    # ── Category A: Read & Understand ─────────────────────────────────────────
    {
        "id": "A01", "suite": "ext", "category": "A",
        "name": "trajectory_gate logic",
        "cwd": TOOLS_DIR,
        "prompt": "Read trajectory_gate.py. Explain the verdict logic step by step, list all possible exit codes with their meaning, and note any uncaught edge cases where the gate could silently pass a bad trajectory.",
        "min_tools": 1, "success_re": r"(verdict|exit|QC|gate|pass|fail)", "timeout": 120,
    },
    {
        "id": "A02", "suite": "ext", "category": "A",
        "name": "safe_qsub preflight sequence",
        "cwd": TOOLS_DIR,
        "prompt": "Read safe_qsub.py. List every preflight check in the exact order it is executed. For each check: what it tests, what happens on failure, whether --force bypasses it.",
        "min_tools": 1, "success_re": r"(preflight|qsub|--force|check|PBS)", "timeout": 120,
    },
    {
        "id": "A03", "suite": "ext", "category": "A",
        "name": "Audit scripts overview",
        "cwd": TOOLS_DIR,
        "prompt": "Find all Python files starting with '_audit' in the current directory. Read each one. Produce a markdown table: filename | what_it_audits | input_required | output_format | key_checks_count.",
        "min_tools": 3, "success_re": r"(_audit|audit|filename|check)", "timeout": 240,
    },
    {
        "id": "A04", "suite": "ext", "category": "A",
        "name": "Proxy request flow diagram",
        "cwd": DEV,
        "prompt": "Read cerit-rewrite-proxy.py. Draw an ASCII flow diagram showing the path of a single POST /v1/messages request through all 5 intervention stages, including decision branches (tool vs non-tool, overflow vs success, GLM vs other).",
        "min_tools": 1, "success_re": r"(→|->|stage|tool|GLM|fallback|sanitize)", "timeout": 150,
    },
    {
        "id": "A05", "suite": "ext", "category": "A",
        "name": "Proxy stderr classification",
        "cwd": DEV,
        "prompt": "Grep cerit-rewrite-proxy.py for every sys.stderr.write call. List them with line numbers and classify each as: ERROR / INFO / GUARD / ROUTING. Count per category.",
        "min_tools": 1, "success_re": r"(line \d+|ERROR|INFO|GUARD|ROUTING|stderr)", "timeout": 90,
    },
    {
        "id": "A06", "suite": "ext", "category": "A",
        "name": "NumPy usage audit",
        "cwd": TOOLS_DIR,
        "prompt": "Find all Python files in the current directory that import numpy. For each: list the numpy functions/attributes actually used in the code.",
        "min_tools": 2, "success_re": r"(numpy|np\.|import)", "timeout": 150,
    },
    {
        "id": "A07", "suite": "ext", "category": "A",
        "name": "TODO/FIXME catalogue",
        "cwd": TOOLS_DIR,
        "prompt": "Search for every TODO and FIXME comment in all Python files in the current directory. Produce a table: file | line | type (TODO/FIXME) | comment text. Sort by file.",
        "min_tools": 1, "success_re": r"(TODO|FIXME|line \d+)", "timeout": 120,
    },
    {
        "id": "A08", "suite": "ext", "category": "A",
        "name": "Zotero agent event loop",
        "cwd": ZOTERO,
        "prompt": "Read agent.py. Explain: (1) what triggers a new ingestion run, (2) the sequence of steps from trigger to Zotero write, (3) how errors at each step are handled (crash / retry / skip / log).",
        "min_tools": 1, "success_re": r"(ingestion|trigger|Zotero|error|retry|step)", "timeout": 150,
    },
    {
        "id": "A09", "suite": "ext", "category": "A",
        "name": "sys.exit usage audit",
        "cwd": TOOLS_DIR,
        "prompt": "Find every function or top-level call that uses sys.exit() or raises SystemExit in the Python files in this directory. List: file, line, exit code or message, and the condition that triggers it.",
        "min_tools": 2, "success_re": r"(sys\.exit|SystemExit|line \d+)", "timeout": 120,
    },
    {
        "id": "A10", "suite": "ext", "category": "A",
        "name": "Codebase size breakdown",
        "cwd": MD_DIR,
        "prompt": "Count Python files and total lines of code per subdirectory under the current directory (tools/, scripts/). For each subdir: file count, total lines, average lines per file. Sort by total lines descending.",
        "min_tools": 2, "success_re": r"(tools|scripts|lines|count|avg)", "timeout": 90,
    },

    # ── Category B: Code Generation ───────────────────────────────────────────
    {
        "id": "B01", "suite": "ext", "category": "B",
        "name": "PDB residue counter",
        "cwd": TOOLS_DIR,
        "prompt": "Write a Python script tools/count_residues.py. It reads a PDB file given as sys.argv[1], counts residues per chain, and prints a CSV: chain,residue_count,first_residue,last_residue. Include argparse help text and handle FileNotFoundError.",
        "min_tools": 1, "success_re": r"(count_residues|argparse|chain|residue|CSV)", "timeout": 120,
    },
    {
        "id": "B02", "suite": "ext", "category": "B",
        "name": "CERIT health monitor script",
        "cwd": "/tmp",
        "prompt": "Write a bash script /tmp/cerit_monitor.sh that loops every 30s, curls http://127.0.0.1:9999/ for health, logs timestamp+HTTP_status+response_time to /tmp/cerit_health.log, and exits after 5 iterations. Make it executable.",
        "min_tools": 1, "success_re": r"(cerit_monitor|curl|sleep 30|cerit_health)", "timeout": 90,
    },
    {
        "id": "B03", "suite": "ext", "category": "B",
        "name": "Batch PDB renamer",
        "cwd": "/tmp",
        "prompt": "Write /tmp/batch_rename.py. CLI: --dir PATH --pattern REGEX --replacement STR --dry-run. Finds .pdb files matching pattern, shows planned renames (dry-run) or executes them. Includes validation that replacement won't cause name collisions.",
        "min_tools": 1, "success_re": r"(batch_rename|dry.run|argparse|pdb|collision)", "timeout": 120,
    },
    {
        "id": "B04", "suite": "ext", "category": "B",
        "name": "TimedSection context manager",
        "cwd": "/tmp",
        "prompt": "Write /tmp/timed_section.py with a TimedSection context manager class. On __exit__ it prints 'Section <name> took X.XXXs'. Include a __main__ block that demonstrates nesting two sections. Use only stdlib.",
        "min_tools": 1, "success_re": r"(TimedSection|__exit__|__enter__|took|timer)", "timeout": 90,
    },
    {
        "id": "B05", "suite": "ext", "category": "B",
        "name": "Proxy smoke test script",
        "cwd": "/tmp",
        "prompt": "Write /tmp/check_proxy.py. It connects to http://127.0.0.1:9999/ with a 10s timeout, sends a minimal Anthropic-format POST to /v1/messages (model=test, messages=[{role:user,content:ping}], max_tokens=10), prints: status_code, first 200 chars of response, and elapsed time. Handles ConnectionRefused gracefully.",
        "min_tools": 1, "success_re": r"(check_proxy|ConnectionRefused|timeout|status_code|/v1/messages)", "timeout": 90,
    },
    {
        "id": "B06", "suite": "ext", "category": "B",
        "name": "Environment report script",
        "cwd": "/tmp",
        "prompt": "Write /tmp/env_report.sh (bash), make it executable, then run it. The script should print: OS name+version, Python version, nvidia-smi GPU count (or 'no GPU'), /tmp free disk space, current username, uptime. Each on its own labeled line.",
        "min_tools": 2, "success_re": r"(env_report|Python|GPU|disk|uptime|username)", "timeout": 120,
    },
    {
        "id": "B07", "suite": "ext", "category": "B",
        "name": "GRO atom counter",
        "cwd": "/tmp",
        "prompt": "Write /tmp/gro_atom_count.py. Reads a GROMACS .gro file (sys.argv[1]). Outputs: total atom count (from header line 2), box vector (last line parsed as 3 floats), and a dict of residue_type:count. Handles malformed files with a clear error message.",
        "min_tools": 1, "success_re": r"(gro_atom_count|atom|box|residue|header)", "timeout": 90,
    },
    {
        "id": "B08", "suite": "ext", "category": "B",
        "name": "SimRun dataclass",
        "cwd": "/tmp",
        "prompt": "Write /tmp/sim_run.py with a Python dataclass SimRun: fields run_id (str), peptide (str), forcefield (str), temperature_K (float), n_atoms (int), duration_ns (float). Include __str__, to_dict(), from_dict() classmethod, and a __main__ block demonstrating round-trip serialization.",
        "min_tools": 1, "success_re": r"(SimRun|dataclass|to_dict|from_dict|round.trip)", "timeout": 90,
    },
    {
        "id": "B09", "suite": "ext", "category": "B",
        "name": "GROMACS log parser",
        "cwd": "/tmp",
        "prompt": "Write /tmp/parse_gromacs_log.py. Reads a GROMACS .log file (sys.argv[1]). Extracts and prints: final potential energy (kJ/mol), final temperature (K), total steps, total wall time (hh:mm:ss). Returns None for missing fields rather than crashing.",
        "min_tools": 1, "success_re": r"(parse_gromacs_log|potential|temperature|steps|wall.time)", "timeout": 90,
    },
    {
        "id": "B10", "suite": "ext", "category": "B",
        "name": "GROMACS workflow Makefile",
        "cwd": "/tmp",
        "prompt": "Write a Makefile at /tmp/Makefile for a GROMACS pre-MD workflow. Targets: pdb2gmx, solvate, ionise, grompp, mdrun, clean. Each target should echo its action and call a placeholder gmx command. Include .PHONY and a help target that lists all targets.",
        "min_tools": 1, "success_re": r"(Makefile|\.PHONY|pdb2gmx|solvate|ionise|help)", "timeout": 90,
    },
    {
        "id": "B11", "suite": "ext", "category": "B",
        "name": "JSON diff CLI",
        "cwd": "/tmp",
        "prompt": "Write /tmp/json_diff.py. CLI: two JSON file paths as positional args. Prints: keys only in A, keys only in B, keys in both with different values (show old/new). Handles nested dicts with dotted key notation (e.g. 'a.b.c'). Test it by creating two small test JSON files and running it.",
        "min_tools": 2, "success_re": r"(json_diff|only in|different|dotted|key)", "timeout": 150,
    },
    {
        "id": "B12", "suite": "ext", "category": "B",
        "name": "Benchmark wall-time plotter",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Write /tmp/cerit_bench_plot.py. Reads a metrics.json from a cerit-tests run (sys.argv[1]). Uses matplotlib to draw a horizontal bar chart of wall_time_sec per test, colored green if completed else red. Saves to /tmp/bench_plot.png and prints the file size.",
        "min_tools": 1, "success_re": r"(bench_plot|matplotlib|wall_time|png|bar)", "timeout": 120,
    },
    {
        "id": "B13", "suite": "ext", "category": "B",
        "name": "PBC unwrap stub",
        "cwd": "/tmp",
        "prompt": "Write /tmp/pbc_unwrap_stub.py. Contains a function unwrap_trajectory(positions, box_vectors) -> np.ndarray with: full docstring including the mathematical formula for minimum-image convention, type hints, and a NotImplementedError body with a clear TODO message. Include a __main__ block that prints the function's docstring.",
        "min_tools": 1, "success_re": r"(pbc_unwrap|unwrap|minimum.image|NotImplementedError|docstring)", "timeout": 90,
    },
    {
        "id": "B14", "suite": "ext", "category": "B",
        "name": "MD run YAML config template",
        "cwd": "/tmp",
        "prompt": "Write a YAML config template /tmp/md_run_config.yaml covering: forcefield (str), water_model (str), temperature_K (float), pressure_bar (float), n_steps (int), output_frequency (int), restraint_type (str, options: none/backbone/heavy-atoms), box_type (str), and a nested electric_field section with direction and magnitude_V_per_nm.",
        "min_tools": 1, "success_re": r"(md_run_config|forcefield|water_model|electric_field|yaml)", "timeout": 60,
    },
    {
        "id": "B15", "suite": "ext", "category": "B",
        "name": "Retry decorator",
        "cwd": "/tmp",
        "prompt": "Write /tmp/retry_decorator.py with a @retry(max_attempts=3, delay_s=1.0, backoff=2.0, exceptions=(Exception,)) decorator. Thread-safe. Logs each retry attempt with attempt number and exception. Include a __main__ demo that retries a function that fails twice then succeeds.",
        "min_tools": 1, "success_re": r"(retry|backoff|max_attempts|thread|decorator)", "timeout": 120,
    },

    # ── Category C: Edit & Refactor ───────────────────────────────────────────
    {
        "id": "C01", "suite": "ext", "category": "C",
        "name": "Add request counter to proxy",
        "cwd": DEV,
        "prompt": "Read cerit-rewrite-proxy.py. Add a thread-safe REQUEST_COUNTER (threading.Lock + int) that increments on every request handled. Log the total count in the startup banner and on each request as '[proxy] req #{N}'. Do not break existing logic.",
        "min_tools": 2, "success_re": r"(REQUEST_COUNTER|threading|Lock|req #|counter)", "timeout": 150,
    },
    {
        "id": "C02", "suite": "ext", "category": "C",
        "name": "Add HTTP timeout to check_proxy",
        "cwd": "/tmp",
        "prompt": "If /tmp/check_proxy.py exists, read it and add a 10s socket timeout to the HTTP connection and a clear 'Proxy is down' message on ConnectionRefused. If the file does not exist, write it from scratch with these features included.",
        "min_tools": 2, "success_re": r"(timeout|ConnectionRefused|Proxy is down|10)", "timeout": 90,
    },
    {
        "id": "C03", "suite": "ext", "category": "C",
        "name": "Add --dry-run to run_tests.py",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Read run_tests.py. Add a --dry-run flag to the argparse CLI. When set, print what would be executed (test ID, name, cwd, timeout, model preset env vars) but do not call subprocess.run. Verify the flag appears in --help output by running python3 run_tests.py --help.",
        "min_tools": 3, "success_re": r"(dry.run|would execute|--help|argparse)", "timeout": 180,
    },
    {
        "id": "C04", "suite": "ext", "category": "C",
        "name": "Add --max-papers to Zotero agent",
        "cwd": ZOTERO,
        "prompt": "Read agent.py. Add a --max-papers INT CLI argument that limits how many new papers are processed in a single run. The limit should be applied before any API calls. Default: unlimited (None). Update any argparse setup already present, or add one.",
        "min_tools": 2, "success_re": r"(max.papers|argparse|limit|None|agent)", "timeout": 180,
    },
    {
        "id": "C05", "suite": "ext", "category": "C",
        "name": "Extend compact regex for Czech/German",
        "cwd": DEV,
        "prompt": "Read cerit-rewrite-proxy.py. The COMPACT_RE regex only matches English compact requests. Add patterns for: German ('Erstell.*Zusammenfassung', 'fass.*zusammen') and Czech ('vytvo.*souhrn', 'shrn.*konverzaci'). Keep the regex as a single compiled pattern.",
        "min_tools": 2, "success_re": r"(COMPACT_RE|German|Czech|Zusammenfassung|souhrn)", "timeout": 120,
    },
    {
        "id": "C06", "suite": "ext", "category": "C",
        "name": "Refactor sanitize_tool_definitions",
        "cwd": DEV,
        "prompt": "Read cerit-rewrite-proxy.py. Extract the per-tool processing logic inside sanitize_tool_definitions into a helper function _patch_single_tool(tool: dict) -> tuple[dict | None, bool] where the bool indicates whether the tool was changed. The outer function should call this helper. Preserve all existing behavior.",
        "min_tools": 2, "success_re": r"(_patch_single_tool|helper|tuple|sanitize|bool)", "timeout": 150,
    },
    {
        "id": "C07", "suite": "ext", "category": "C",
        "name": "Add --list-preflights to safe_qsub",
        "cwd": TOOLS_DIR,
        "prompt": "Read safe_qsub.py. Add a --list-preflights flag that prints the names and descriptions of all registered preflight checks, then exits, without submitting any job. Run python3 safe_qsub.py --list-preflights to verify.",
        "min_tools": 3, "success_re": r"(list.preflights|preflight|--list|safe_qsub)", "timeout": 150,
    },
    {
        "id": "C08", "suite": "ext", "category": "C",
        "name": "Add HTTP 429 retry to proxy",
        "cwd": DEV,
        "prompt": "Read cerit-rewrite-proxy.py. The fallback chain only triggers on HTTP 400. Add HTTP 429 (rate limit) handling: on 429 from upstream, wait 5s then retry the same model, up to 3 retries, before falling through to the fallback chain. Log each retry attempt.",
        "min_tools": 2, "success_re": r"(429|rate.limit|retry|5s|sleep|3 retries)", "timeout": 180,
    },
    {
        "id": "C09", "suite": "ext", "category": "C",
        "name": "Split _relay_success into SSE/non-SSE",
        "cwd": DEV,
        "prompt": "Read cerit-rewrite-proxy.py. The _relay_success method handles both SSE and non-SSE responses in one function. Split it into _relay_sse(resp) and _relay_non_sse(resp) private methods, with _relay_success calling the appropriate one. Keep all existing logic.",
        "min_tools": 2, "success_re": r"(_relay_sse|_relay_non_sse|split|SSE|chunked)", "timeout": 150,
    },
    {
        "id": "C10", "suite": "ext", "category": "C",
        "name": "Add success_fn to run_tests.py",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Read run_tests.py. The success check currently only uses success_re (regex on text output). Add an optional success_fn field per test: a string containing a Python lambda that is eval()'d with output as argument. If both success_re and success_fn are present, both must pass. Add a test using success_fn to verify.",
        "min_tools": 2, "success_re": r"(success_fn|lambda|eval|both.*pass)", "timeout": 180,
    },
    {
        "id": "C11", "suite": "ext", "category": "C",
        "name": "Write broken parser then fix it",
        "cwd": "/tmp",
        "prompt": "Write /tmp/broken_parser.py: a function parse_csv_line(line) that splits on comma and returns a list of values, but has a deliberate off-by-one bug (returns items[1:] instead of items). Then find the bug by running a test, fix it, and verify the fix works.",
        "min_tools": 3, "success_re": r"(broken_parser|off.by.one|bug|fixed|parse_csv_line)", "timeout": 150,
    },
    {
        "id": "C12", "suite": "ext", "category": "C",
        "name": "Extract prompts to cerit_prompts.py",
        "cwd": DEV,
        "prompt": "Read cerit-rewrite-proxy.py. Extract CERIT_CONTINUATION, TURN_WARN_SOFT_TMPL, TURN_WARN_HARD_TMPL, and REPEAT_GUARD_TMPL into a new file cerit_prompts.py in the same directory. Update cerit-rewrite-proxy.py to import them from cerit_prompts. Verify the proxy still starts: python3 cerit-rewrite-proxy.py --help or just import it.",
        "min_tools": 3, "success_re": r"(cerit_prompts|import|extract|CERIT_CONTINUATION)", "timeout": 180,
    },
    {
        "id": "C13", "suite": "ext", "category": "C",
        "name": "Python-free fallback in bashrc context check",
        "cwd": DEV,
        "prompt": "Read cerit-bashrc.snippet. The _cerit_context_check function uses Python to read the jsonl. Rewrite it to first check if python/python3 is available; if not, print a warning and skip the check (return 0) rather than crashing. Keep all existing Python logic when Python is available.",
        "min_tools": 2, "success_re": r"(command -v|python|fallback|skip|warning)", "timeout": 120,
    },
    {
        "id": "C14", "suite": "ext", "category": "C",
        "name": "Add --proxy-log to run_tests.py",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Read run_tests.py. The PROXY_LOG path is a module-level constant. Add a --proxy-log CLI argument that overrides this path at runtime. Update parse_proxy_log_delta and proxy_log_offset to use the configurable path.",
        "min_tools": 2, "success_re": r"(proxy.log|--proxy-log|argparse|override)", "timeout": 120,
    },
    {
        "id": "C15", "suite": "ext", "category": "C",
        "name": "Add cache TTL to abstract_fetcher",
        "cwd": ZOTERO,
        "prompt": "Read abstract_fetcher.py. Add a cache_ttl_hours parameter (default 24) to the main fetching function/class. Before making an HTTP request for an abstract, check if a cached version exists and is younger than cache_ttl_hours. Use a simple JSON file cache in /tmp/abstract_cache.json.",
        "min_tools": 2, "success_re": r"(cache_ttl|cache|ttl|json|abstract_fetcher)", "timeout": 180,
    },

    # ── Category D: Bash / Shell ──────────────────────────────────────────────
    {
        "id": "D01", "suite": "ext", "category": "D",
        "name": "Python LOC stats",
        "cwd": TOOLS_DIR,
        "prompt": "Count total non-blank non-comment Python lines in the current directory. Report: total LOC, average per file, top 5 largest files by LOC. Use bash/grep/awk.",
        "min_tools": 1, "success_re": r"(total|average|LOC|top.5|lines)", "timeout": 60,
    },
    {
        "id": "D02", "suite": "ext", "category": "D",
        "name": "Recently modified tools",
        "cwd": TOOLS_DIR,
        "prompt": "Find all Python files in the current directory modified in the last 7 days. List each with its modification timestamp (human readable). Sort newest first.",
        "min_tools": 1, "success_re": r"(modified|days|timestamp|\.py)", "timeout": 60,
    },
    {
        "id": "D03", "suite": "ext", "category": "D",
        "name": "Compare tool help outputs",
        "cwd": TOOLS_DIR,
        "prompt": "Run 'python3 trajectory_gate.py --help' and 'python3 safe_qsub.py --help'. Extract all argument names from each. Show arguments unique to each tool and arguments they share.",
        "min_tools": 1, "success_re": r"(trajectory_gate|safe_qsub|unique|shared|argument)", "timeout": 90,
    },
    {
        "id": "D04", "suite": "ext", "category": "D",
        "name": "Disk usage check",
        "cwd": str(HOME),
        "prompt": "Check disk usage: report free space on home directory, /tmp, and /scratch if it exists. Also report total used by ~/dev/. Use df and du. Format output clearly.",
        "min_tools": 1, "success_re": r"(free|used|/tmp|home|disk)", "timeout": 60,
    },
    {
        "id": "D05", "suite": "ext", "category": "D",
        "name": "Recent log file inspection",
        "cwd": str(HOME),
        "prompt": "Find all .log files in /tmp/ owned by the current user. Sort by modification time (newest first). Show the last 10 lines of the most recent one. Report filename and size.",
        "min_tools": 1, "success_re": r"(\.log|modification|newest|/tmp|last)", "timeout": 60,
    },
    {
        "id": "D06", "suite": "ext", "category": "D",
        "name": "Venv package versions",
        "cwd": ZOTERO,
        "prompt": "Check what Python packages are installed in the venv/ in the current directory. List all packages whose name contains any of: anthropic, openai, requests, httpx, aiohttp. Show name and version.",
        "min_tools": 1, "success_re": r"(anthropic|requests|httpx|version|venv)", "timeout": 60,
    },
    {
        "id": "D07", "suite": "ext", "category": "D",
        "name": "Proxy HTTP health check",
        "cwd": DEV,
        "prompt": "Check if the rewrite proxy is running: curl http://127.0.0.1:9999/ with a 3s timeout. Report: is it up or down, HTTP status code if up, response body first 100 chars. If down, check if the process is running via ps.",
        "min_tools": 1, "success_re": r"(proxy|up|down|9999|status|curl)", "timeout": 60,
    },
    {
        "id": "D08", "suite": "ext", "category": "D",
        "name": "Git repos status",
        "cwd": str(HOME),
        "prompt": "Find all Git repositories under ~/dev/. For each: report current branch, number of uncommitted files (git status --short), and the subject line of the last commit. Format as a table.",
        "min_tools": 1, "success_re": r"(branch|commit|uncommitted|git|repo)", "timeout": 120,
    },
    {
        "id": "D09", "suite": "ext", "category": "D",
        "name": "Proxy startup timing",
        "cwd": DEV,
        "prompt": "Measure proxy startup time using a SAFE test port — do NOT touch port 9999. Start a NEW instance of cerit-rewrite-proxy.py with env var PROXY_PORT=9998 in background, poll http://127.0.0.1:9998/ every 0.2s until it responds (max 20s), measure elapsed time, then kill that test instance only.",
        "min_tools": 1, "success_re": r"(startup|elapsed|9998|0\.\d+|seconds|proxy)", "timeout": 120,
    },
    {
        "id": "D10", "suite": "ext", "category": "D",
        "name": "PBS file resource lines",
        "cwd": str(HOME),
        "prompt": "Find all *.pbs files anywhere under ~/dev/. For each: extract all #PBS -l resource lines and the last qsub command if present. Report as: filepath | resources | qsub_cmd.",
        "min_tools": 1, "success_re": r"(\.pbs|#PBS|-l |qsub|resource)", "timeout": 90,
    },
    {
        "id": "D11", "suite": "ext", "category": "D",
        "name": "Bench workspace setup",
        "cwd": "/tmp",
        "prompt": "Create directory /tmp/bench_workspace/ with 5 subdirectories run_01 through run_05. In each: create output.txt with content 'run N placeholder', config.json with {\"run_id\": N, \"status\": \"pending\"}. Verify all files exist.",
        "min_tools": 1, "success_re": r"(bench_workspace|run_0[1-5]|output\.txt|config\.json)", "timeout": 60,
    },
    {
        "id": "D12", "suite": "ext", "category": "D",
        "name": "Python processes inventory",
        "cwd": str(HOME),
        "prompt": "List all running Python processes owned by the current user. For each: PID, elapsed time (etime), VSZ (virtual memory MB), and command line truncated to 100 chars. Sort by elapsed time descending.",
        "min_tools": 1, "success_re": r"(PID|python|elapsed|VSZ|command)", "timeout": 60,
    },
    {
        "id": "D13", "suite": "ext", "category": "D",
        "name": "Proxy log event counts",
        "cwd": str(HOME),
        "prompt": "Find the most recent cerit proxy log in /tmp/ (cerit-proxy-*.log or cerit-rewrite-proxy.log). Count lines by type: injected continuation rule, REPEAT_GUARD, TURN_WARN, fallback, upstream error, GLM thinking disabled. Report a table.",
        "min_tools": 1, "success_re": r"(continuation|REPEAT_GUARD|TURN_WARN|fallback|upstream|GLM)", "timeout": 60,
    },
    {
        "id": "D14", "suite": "ext", "category": "D",
        "name": "Syntax check all tools",
        "cwd": TOOLS_DIR,
        "prompt": "Run 'python3 -m py_compile' on every .py file in the current directory. Report: how many pass, how many fail, and for failures show the exact error message.",
        "min_tools": 1, "success_re": r"(py_compile|pass|fail|syntax|compile)", "timeout": 120,
    },
    {
        "id": "D15", "suite": "ext", "category": "D",
        "name": "System snapshot",
        "cwd": "/tmp",
        "prompt": "Generate a system snapshot: CPU model, total RAM, available RAM, Python version (from python3 --version), nvidia-smi GPU count (or 'no GPU'), hostname, uptime, kernel version. Write everything to /tmp/system_snapshot.txt and print it.",
        "min_tools": 1, "success_re": r"(CPU|RAM|Python|hostname|uptime|system_snapshot)", "timeout": 90,
    },

    # ── Category E: Search & Audit ────────────────────────────────────────────
    {
        "id": "E01", "suite": "ext", "category": "E",
        "name": "Hardcoded IPs and ports audit",
        "cwd": DEV,
        "prompt": "Search all Python files under ~/dev/ for hardcoded IP addresses (regex: \\d+\\.\\d+\\.\\d+\\.\\d+) and port numbers in strings (e.g. ':9999', 'port=23119'). List: file, line, the hardcoded value. Exclude obvious test/example values in comments.",
        "min_tools": 1, "success_re": r"(IP|port|hardcoded|\d+\.\d+\.\d+\.\d+|9999)", "timeout": 90,
    },
    {
        "id": "E02", "suite": "ext", "category": "E",
        "name": "Broad exception catches audit",
        "cwd": TOOLS_DIR,
        "prompt": "Find all 'except Exception' and bare 'except:' clauses in all Python files in the current directory. List: file, line number, the full except clause, and whether there is any logging inside the handler.",
        "min_tools": 1, "success_re": r"(except Exception|bare except|line \d+|logging|handler)", "timeout": 180,
    },
    {
        "id": "E03", "suite": "ext", "category": "E",
        "name": "subprocess shell=True audit",
        "cwd": TOOLS_DIR,
        "prompt": "Find every 'import subprocess' in Python files in this directory. For each file that imports subprocess, check whether any calls use shell=True (shell injection risk). List: file, line, the exact call, and risk level (HIGH if shell=True, LOW otherwise).",
        "min_tools": 1, "success_re": r"(subprocess|shell=True|shell.injection|HIGH|LOW)", "timeout": 90,
    },
    {
        "id": "E04", "suite": "ext", "category": "E",
        "name": "Mixed path-join styles",
        "cwd": TOOLS_DIR,
        "prompt": "Find Python files in this directory that use BOTH os.path.join (or pathlib) AND plain string concatenation with '/' for file path construction. List each file with one example of each style.",
        "min_tools": 1, "success_re": r"(os\.path\.join|pathlib|string concatenation|both|/)", "timeout": 240,
    },
    {
        "id": "E05", "suite": "ext", "category": "E",
        "name": "Functions without docstrings",
        "cwd": TOOLS_DIR,
        "prompt": "Find all Python function definitions (def ...) in this directory that have no docstring (no triple-quoted string as first statement in the body). List: file, line number, function name. Count total.",
        "min_tools": 2, "success_re": r"(docstring|def |line \d+|function|total)", "timeout": 180,
    },
    {
        "id": "E06", "suite": "ext", "category": "E",
        "name": "print vs stderr audit",
        "cwd": DEV,
        "prompt": "Search cerit-rewrite-proxy.py and cerit-tests/run_tests.py for all print() calls. For each: is it informational output (OK as print) or diagnostic/error output (should be sys.stderr.write)? List with your assessment.",
        "min_tools": 1, "success_re": r"(print|stderr|informational|diagnostic|assessment)", "timeout": 90,
    },
    {
        "id": "E07", "suite": "ext", "category": "E",
        "name": "Files opened without context manager",
        "cwd": TOOLS_DIR,
        "prompt": "Find Python files in this directory that open files using open() NOT as 'with open(...) as ...'. List file, line, the open() call. These are resource leak risks.",
        "min_tools": 1, "success_re": r"(open\(|context manager|with open|resource leak|line \d+)", "timeout": 120,
    },
    {
        "id": "E08", "suite": "ext", "category": "E",
        "name": "Assert statements in non-test code",
        "cwd": TOOLS_DIR,
        "prompt": "Find all assert statements in Python files in this directory. List: file, line, the full assert expression. Flag any in production code (not in test_ files or __main__ blocks) as potentially dangerous (asserts are disabled with -O).",
        "min_tools": 1, "success_re": r"(assert|line \d+|production|disabled|-O|flag)", "timeout": 90,
    },
    {
        "id": "E09", "suite": "ext", "category": "E",
        "name": "Deprecated Python features",
        "cwd": DEV,
        "prompt": "Search all Python files under ~/dev/ for deprecated or removed features: os.system(), commands module, distutils, imp module, asynchat, asyncore. List any found with file and line.",
        "min_tools": 1, "success_re": r"(deprecated|os\.system|distutils|imp|commands|asynchat)", "timeout": 90,
    },
    {
        "id": "E10", "suite": "ext", "category": "E",
        "name": "main() without guard",
        "cwd": TOOLS_DIR,
        "prompt": "Find Python files in this directory that define a main() function but lack an 'if __name__ == \"__main__\": main()' guard. List each file and whether it has a __main__ guard at all.",
        "min_tools": 1, "success_re": r"(__main__|main\(\)|guard|missing|define)", "timeout": 120,
    },

    # ── Category F: Multi-tool Workflows ──────────────────────────────────────
    {
        "id": "F01", "suite": "ext", "category": "F",
        "name": "Shebang line audit report",
        "cwd": TOOLS_DIR,
        "prompt": "Scan all Python files in the current directory. Check whether each has a shebang line (#!/usr/bin/env python3 or similar) as its first line. Write /tmp/shebang_report.md: two sections — 'Has shebang' and 'Missing shebang', each with a list of filenames.",
        "min_tools": 2, "success_re": r"(shebang|#!/|Missing shebang|Has shebang|shebang_report)", "timeout": 180,
    },
    {
        "id": "F02", "suite": "ext", "category": "F",
        "name": "Proxy integration smoke test",
        "cwd": DEV,
        "prompt": "Start a test instance of cerit-rewrite-proxy.py on port 9998 (PROXY_PORT=9998 python3 cerit-rewrite-proxy.py &). Wait for it to be ready (poll /dev/tcp/127.0.0.1/9998 or curl). Send a test POST to http://127.0.0.1:9998/v1/messages with a minimal body. Capture the response. Kill the instance. Report: started OK, response status, any errors.",
        "min_tools": 2, "success_re": r"(9998|started|response|kill|smoke.test|proxy)", "timeout": 240,
    },
    {
        "id": "F03", "suite": "ext", "category": "F",
        "name": "MetaCentrum reference report",
        "cwd": TOOLS_DIR,
        "prompt": "Find all Python files in the current directory that contain the word 'MetaCentrum' or 'metacentrum'. For each match: read the file, extract up to 3 lines of context around each occurrence. Write /tmp/metacentrum_refs.md with file, line number, and context snippet.",
        "min_tools": 2, "success_re": r"(MetaCentrum|metacentrum_refs|context|line \d+)", "timeout": 240,
    },
    {
        "id": "F04", "suite": "ext", "category": "F",
        "name": "Add and run new test T8",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Read run_tests.py. Add a new test T8 to TESTS_BASE with id='T8', name='Proxy startup verify', cwd='/tmp', prompt='Start cerit-rewrite-proxy.py on port 9997, verify it listens, kill it, report OK or FAIL.', timeout=60, success_re='(OK|FAIL|9997|proxy)'. Then run: python3 run_tests.py --tests T8 --suite all --model glm --no-judge and report the result.",
        "min_tools": 3, "success_re": r"(T8|Proxy startup|test added|done|completed)", "timeout": 360,
    },
    {
        "id": "F05", "suite": "ext", "category": "F",
        "name": "Zotero agent state summary",
        "cwd": ZOTERO,
        "prompt": "Read agent_state.json and pending_papers.json if they exist. Compute: number of pending papers, number of processed papers, last run timestamp (if stored), any error flags. Write a summary to /tmp/zotero_status.txt and print it.",
        "min_tools": 2, "success_re": r"(pending|processed|timestamp|zotero_status|summary)", "timeout": 120,
    },
    {
        "id": "F06", "suite": "ext", "category": "F",
        "name": "Add prog= to ArgumentParsers",
        "cwd": TOOLS_DIR,
        "prompt": "Find all Python files in this directory that use argparse.ArgumentParser() without a prog= argument. For up to 5 files, add prog=os.path.basename(__file__) to the ArgumentParser call. Report: how many files were found total, which 5 were modified.",
        "min_tools": 3, "success_re": r"(prog=|ArgumentParser|os\.path\.basename|modified|argparse)", "timeout": 300,
    },
    {
        "id": "F07", "suite": "ext", "category": "F",
        "name": "Bashrc function inventory",
        "cwd": DEV,
        "prompt": "Read cerit-bashrc.snippet. Extract all function names (lines matching '^function_name()' pattern). For each function: run it with --help if it supports it, otherwise note 'no --help'. List: function name, purpose (from the comment above it), help support.",
        "min_tools": 2, "success_re": r"(claude-cerit|function|--help|purpose|bashrc)", "timeout": 150,
    },
    {
        "id": "F08", "suite": "ext", "category": "F",
        "name": "Run all audit scripts",
        "cwd": TOOLS_DIR,
        "prompt": "Run every _audit_*.py script in the current directory that accepts no required positional arguments (try each with no args, catch errors). For each: show the first 5 lines of stdout, or the error. Write a consolidated report to /tmp/audit_consolidated.txt.",
        "min_tools": 2, "success_re": r"(audit|_audit|consolidated|report|stdout)", "timeout": 360,
    },
    {
        "id": "F09", "suite": "ext", "category": "F",
        "name": "Test suite inventory",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Read run_tests.py. Extract all test IDs, names, cwds, and timeouts from the TESTS_BASE and TESTS_EXT lists. Run python3 run_tests.py --help to confirm CLI options. Write /tmp/test_inventory.md: table of all tests plus CLI reference.",
        "min_tools": 2, "success_re": r"(test_inventory|TESTS|ID|timeout|CLI)", "timeout": 300,
    },
    {
        "id": "F10", "suite": "ext", "category": "F",
        "name": "gmx command catalogue",
        "cwd": TOOLS_DIR,
        "prompt": "Find all Python files in this directory that reference 'gmx' or 'GROMACS'. Read each. Extract every distinct gmx subcommand used (e.g. gmx pdb2gmx, gmx solvate). Deduplicate. Write /tmp/gmx_commands.txt: one command per line, sorted, with the source file(s) that use it.",
        "min_tools": 2, "success_re": r"(gmx|GROMACS|pdb2gmx|solvate|gmx_commands)", "timeout": 420,
    },
    {
        "id": "F11", "suite": "ext", "category": "F",
        "name": "Extract CERIT_MODELS dict",
        "cwd": DEV,
        "prompt": "Read cerit-rewrite-proxy.py. Extract all CERIT model name strings (values in MODEL_OVERRIDES, FALLBACK_CHAIN keys, DEFAULT_TARGET, DEFAULT_TOOL_TARGET) into a consolidated CERIT_MODELS dict at the top of the file with categories: 'guaranteed' and 'experimental'. Verify the proxy still starts after the change.",
        "min_tools": 3, "success_re": r"(CERIT_MODELS|guaranteed|experimental|extract|starts)", "timeout": 240,
    },
    {
        "id": "F12", "suite": "ext", "category": "F",
        "name": "Recent tools summary",
        "cwd": TOOLS_DIR,
        "prompt": "Find the 5 most recently modified Python files in the current directory. For each: read it, extract: purpose (from docstring/first comment), any CLI entry-point command, and last-modified timestamp. Write /tmp/recent_tools_summary.md.",
        "min_tools": 3, "success_re": r"(recent_tools|most recently|docstring|modified|purpose)", "timeout": 180,
    },
    {
        "id": "F13", "suite": "ext", "category": "F",
        "name": "Ecosystem commit digest",
        "cwd": str(HOME / "dev/ecosystem") if (HOME / "dev/ecosystem").exists() else DEV,
        "prompt": "Run 'git log --oneline -10' in the current directory. For the 3 most recent commits, run 'git show --stat <hash>' to see what changed. Summarize each commit: what files changed and why (inferred from commit message). Write to /tmp/ecosystem_changes.md.",
        "min_tools": 2, "success_re": r"(commit|changed|ecosystem|git|summary)", "timeout": 180,
    },
    {
        "id": "F14", "suite": "ext", "category": "F",
        "name": "Proxy memory profile",
        "cwd": DEV,
        "prompt": "Start cerit-rewrite-proxy.py in background. Find its PID. Every 2 seconds for 10 seconds, read its RSS memory from /proc/PID/status. Kill it. Report: peak RSS (kB), average RSS, and whether memory grew over time. Write to /tmp/proxy_memory.txt.",
        "min_tools": 2, "success_re": r"(RSS|memory|peak|average|proxy_memory|/proc)", "timeout": 180,
    },
    {
        "id": "F15", "suite": "ext", "category": "F",
        "name": "Benchmark history aggregate",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Find all metrics.json files under results/. For each run: read it, extract run_id (from dirname), model, completion rate, avg wall time, total idle stops. Aggregate across all runs. Write /tmp/benchmark_history.md: per-run table plus overall summary statistics.",
        "min_tools": 2, "success_re": r"(benchmark_history|completion|idle|wall.time|aggregate)", "timeout": 360,
    },

    # ── Category G: Long-context Stress ──────────────────────────────────────
    {
        "id": "G01", "suite": "ext", "category": "G",
        "name": "All audit scripts deep analysis",
        "cwd": TOOLS_DIR,
        "prompt": "Read ALL Python files starting with '_audit' in the current directory. For each one document: (1) what input it requires, (2) what it validates/checks, (3) what output it produces, (4) what it does on failure. Produce a comprehensive reference document.",
        "min_tools": 5, "success_re": r"(_audit|input|output|failure|validates)", "timeout": 600,
    },
    {
        "id": "G02", "suite": "ext", "category": "G",
        "name": "Proxy + test runner unified architecture",
        "cwd": DEV,
        "prompt": "Read cerit-rewrite-proxy.py AND cerit-tests/run_tests.py completely. Produce a unified architecture document covering: (1) data flow from Claude Code → proxy → CERIT → back, (2) all configuration knobs in both files, (3) failure modes and how each is handled, (4) the 5 proxy intervention stages with code evidence.",
        "min_tools": 2, "success_re": r"(architecture|data flow|configuration|failure|intervention)", "timeout": 360,
    },
    {
        "id": "G03", "suite": "ext", "category": "G",
        "name": "_add_* scripts pattern analysis",
        "cwd": TOOLS_DIR,
        "prompt": "Read ALL Python files in this directory that start with '_add_'. They all modify HTML research reports. Document: (1) the common pattern they follow, (2) what each one adds/modifies, (3) divergences from the common pattern, (4) shared utility functions they call.",
        "min_tools": 5, "success_re": r"(_add_|HTML|pattern|common|diverge)", "timeout": 600,
    },
    {
        "id": "G04", "suite": "ext", "category": "G",
        "name": "Benchmark longitudinal analysis",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Read all metrics.json files under results/. Perform a longitudinal analysis: for each test ID that appears in multiple runs, track how wall_time, n_turns, n_tool_calls, and idle_stop_rate evolved. Identify regressions (metric got worse). Write a full analysis to /tmp/benchmark_longitudinal.md.",
        "min_tools": 3, "success_re": r"(longitudinal|regression|evolved|wall_time|benchmark)", "timeout": 600,
    },
    {
        "id": "G05", "suite": "ext", "category": "G",
        "name": "Class catalogue",
        "cwd": TOOLS_DIR,
        "prompt": "Read ALL Python files in this directory that contain 'class '. For each class: extract class name, parent classes, all method names and their docstrings. Write a complete class catalogue to /tmp/class_catalogue.md organized by file.",
        "min_tools": 5, "success_re": r"(class_catalogue|class |method|docstring|parent)", "timeout": 600,
    },

    # ── Category H: Vibe Style Variants ──────────────────────────────────────
    {
        "id": "H01", "suite": "ext", "category": "H",
        "name": "Terse: fix retry logic",
        "cwd": TOOLS_DIR,
        "prompt": "Fix the retry logic in safe_qsub.py",
        "min_tools": 1, "success_re": r"(retry|safe_qsub|fixed|attempt|logic)", "timeout": 150,
    },
    {
        "id": "H02", "suite": "ext", "category": "H",
        "name": "Terse: write gro parser",
        "cwd": "/tmp",
        "prompt": "Write a gro parser",
        "min_tools": 1, "success_re": r"(gro|parser|atom|box|residue|\.gro)", "timeout": 300,
    },
    {
        "id": "H03", "suite": "ext", "category": "H",
        "name": "Terse: check proxy",
        "cwd": DEV,
        "prompt": "Check if the proxy is running",
        "min_tools": 1, "success_re": r"(proxy|running|9999|up|down|status)", "timeout": 90,
    },
    {
        "id": "H04", "suite": "ext", "category": "H",
        "name": "Overspecified: trivial add()",
        "cwd": "/tmp",
        "prompt": "Write a Python function add(a: int, b: int) -> int in /tmp/add_function.py. Requirements: (1) PEP 484 type hints, (2) Google-style docstring with Args and Returns sections, (3) handles non-integer inputs by raising TypeError with message 'Expected int, got TYPE', (4) a unittest.TestCase class with 3 tests: positive, negative, type error. Run the tests and confirm they pass.",
        "min_tools": 2, "success_re": r"(add_function|TypeError|TestCase|pass|docstring)", "timeout": 150,
    },
    {
        "id": "H05", "suite": "ext", "category": "H",
        "name": "Conversational: GLM proxy walkthrough",
        "cwd": DEV,
        "prompt": "Hey, I'm trying to understand how the proxy handles GLM. Can you take a look at cerit-rewrite-proxy.py and walk me through what specifically happens differently for GLM requests compared to other models?",
        "min_tools": 1, "success_re": r"(GLM|glm-5\.2|thinking|enable_thinking|chat_template)", "timeout": 180,
    },
    {
        "id": "H06", "suite": "ext", "category": "H",
        "name": "Vague: make Zotero agent faster",
        "cwd": ZOTERO,
        "prompt": "The Zotero agent feels slow. Can you take a look and suggest or implement improvements to make it faster?",
        "min_tools": 2, "success_re": r"(slow|faster|performance|async|cache|batch|improvement)", "timeout": 500,
    },
    {
        "id": "H07", "suite": "ext", "category": "H",
        "name": "Aggressive: proxy memory profile now",
        "cwd": DEV,
        "prompt": "I need the proxy memory and CPU profile RIGHT NOW. Start it, measure for 10 seconds, show me RSS, VSZ, CPU%. Go.",
        "min_tools": 1, "success_re": r"(RSS|VSZ|CPU|memory|proxy|/proc|%)", "timeout": 120,
    },
    {
        "id": "H08", "suite": "ext", "category": "H",
        "name": "Incremental: build CLI tool step by step",
        "cwd": "/tmp",
        "prompt": "Build /tmp/cerit_status.py incrementally: first write just the argparse setup with --verbose and --format (json|text) flags. Then add the main logic that checks proxy status and last benchmark. Then add error handling for missing files. Run it at each stage.",
        "min_tools": 3, "success_re": r"(cerit_status|argparse|stage|incremental|--verbose)", "timeout": 300,
    },
    {
        "id": "H09", "suite": "ext", "category": "H",
        "name": "Research-first: GROMACS GPU flags",
        "cwd": "/tmp",
        "prompt": "Look up the current recommended GROMACS 2024 mdrun GPU acceleration flags (nb, pme, bonded, update offload). Then write a PBS script /tmp/gromacs_gpu_best.pbs using those best-practice flags for a 1-GPU A100 job.",
        "min_tools": 2, "success_re": r"(mdrun|gpu|nb|pme|bonded|A100|PBS|gromacs_gpu)", "timeout": 300,
    },
    {
        "id": "H10", "suite": "ext", "category": "H",
        "name": "Expert role: PBS script review",
        "cwd": TOOLS_DIR,
        "prompt": "You are a senior MetaCentrum HPC administrator reviewing a PBS job script. Read mc_qsub.sh in the current directory. What would you change or flag as problematic? Be specific about resource efficiency, error handling, and portability.",
        "min_tools": 1, "success_re": r"(PBS|resource|efficiency|portability|flag|HPC)", "timeout": 150,
    },

    # ── Category I: Verification / Self-checking ──────────────────────────────
    {
        "id": "I01", "suite": "ext", "category": "I",
        "name": "Proxy import check",
        "cwd": DEV,
        "prompt": "Run: python3 -c 'import sys; sys.path.insert(0,\".\"); import importlib.util; spec=importlib.util.spec_from_file_location(\"p\",\"cerit-rewrite-proxy.py\"); m=importlib.util.module_from_spec(spec)' and report if it imports without error. If there are import errors, show them.",
        "min_tools": 1, "success_re": r"(import|error|clean|success|cerit-rewrite-proxy)", "timeout": 60,
    },
    {
        "id": "I02", "suite": "ext", "category": "I",
        "name": "trajectory_gate --help flags check",
        "cwd": TOOLS_DIR,
        "prompt": "Run 'python3 trajectory_gate.py --help'. Verify these flags are present: --require-verdict, --strict. If any are missing, report which are absent and read the source to explain why.",
        "min_tools": 2, "success_re": r"(--require-verdict|--strict|present|absent|trajectory_gate)", "timeout": 60,
    },
    {
        "id": "I03", "suite": "ext", "category": "I",
        "name": "Unit test sanitize_tool_definitions",
        "cwd": DEV,
        "prompt": "Write /tmp/test_sanitize.py with a unittest.TestCase that tests sanitize_tool_definitions from cerit-rewrite-proxy.py. Tests: (1) web_search_20250305 converted to web_search, (2) computer_20241022 stripped, (3) tool with missing input_schema gets default added, (4) empty tools list returns unchanged. Run with python3 -m pytest /tmp/test_sanitize.py -v and show results.",
        "min_tools": 3, "success_re": r"(test_sanitize|PASSED|sanitize_tool_definitions|unittest|pytest)", "timeout": 240,
    },
    {
        "id": "I04", "suite": "ext", "category": "I",
        "name": "Syntax check proxy and runner",
        "cwd": DEV,
        "prompt": "Run python3 -m py_compile on cerit-rewrite-proxy.py and cerit-tests/run_tests.py. Report PASS or FAIL for each. If FAIL, show the exact error message and line number.",
        "min_tools": 1, "success_re": r"(py_compile|PASS|FAIL|syntax|compile|cerit)", "timeout": 60,
    },
    {
        "id": "I05", "suite": "ext", "category": "I",
        "name": "Proxy routing integration test",
        "cwd": DEV,
        "prompt": "Write /tmp/test_proxy_routes.py. It starts cerit-rewrite-proxy.py on port 9997, then sends 3 POST requests with different 'model' values: 'claude-sonnet-4-6', 'claude-cerit-glm', 'claude-cerit-deep'. For each, check that the proxy logs show the correct model rewrite. Kill the proxy. Print PASS/FAIL per test.",
        "min_tools": 3, "success_re": r"(test_proxy_routes|PASS|FAIL|9997|model rewrite|routing)", "timeout": 300,
    },

    # ── Category J: Web Research + Implement ──────────────────────────────────
    {
        "id": "J01", "suite": "ext", "category": "J",
        "name": "GROMACS 2024 MDP template",
        "cwd": "/tmp",
        "prompt": "Search the web for GROMACS 2024 recommended mdp parameters for protein MD in TIP3P water (NVT equilibration and NPT production). Write a template /tmp/md.mdp with: integrator, nsteps, dt, tcoupl, pcoupl, coulombtype, vdwtype, constraints. Include comments explaining each parameter.",
        "min_tools": 2, "success_re": r"(md\.mdp|integrator|tcoupl|pcoupl|TIP3P|GROMACS)", "timeout": 360,
    },
    {
        "id": "J02", "suite": "ext", "category": "J",
        "name": "CERIT API docs fetch",
        "cwd": "/tmp",
        "prompt": "Fetch https://docs.cerit.io/en/docs/ai-as-a-service/ai-api and summarize: available model names, context window sizes, tool calling support, any rate limits mentioned. Write to /tmp/cerit_api_notes.txt.",
        "min_tools": 2, "success_re": r"(cerit_api_notes|model|context|tool|rate|docs\.cerit)", "timeout": 300,
    },
    {
        "id": "J03", "suite": "ext", "category": "J",
        "name": "GLM external benchmark comparison",
        "cwd": "/tmp",
        "prompt": "Search the web for any independent benchmarks of GLM-5 or GLM-5.2 language model from 2025-2026. Find at least one source. Compare what they report to our run9 results (7/7, 494s, 0 idle-stop). Write /tmp/glm_external_comparison.md noting similarities and differences.",
        "min_tools": 2, "success_re": r"(GLM|benchmark|run9|comparison|external|494)", "timeout": 360,
    },
    {
        "id": "J04", "suite": "ext", "category": "J",
        "name": "python-docx table demo",
        "cwd": "/tmp",
        "prompt": "Search for the current python-docx API for inserting a table. Write /tmp/docx_table_demo.py that creates a .docx with a 3-column benchmark results table (Model | Total time | Completion rate) with 4 rows of our GLM/qwen/deepseek results. Run it and confirm the file was created.",
        "min_tools": 3, "success_re": r"(docx_table_demo|python-docx|table|Document|created)", "timeout": 300,
    },
    {
        "id": "J05", "suite": "ext", "category": "J",
        "name": "MetaCentrum GPU queue limits",
        "cwd": "/tmp",
        "prompt": "Fetch https://docs.metacentrum.cz or search for MetaCentrum PBS queue list and GPU node resource limits (2024-2026). Write /tmp/metacentrum_queues.md with a table: queue_name | max_walltime | GPU_type | max_GPUs_per_job | notes.",
        "min_tools": 2, "success_re": r"(MetaCentrum|queue|walltime|GPU|metacentrum_queues)", "timeout": 360,
    },

    # ── Category K: Autonomous / Self-directed ────────────────────────────────
    {
        "id": "K01", "suite": "ext", "category": "K",
        "name": "Autonomous: audit cerit-tests",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Audit the cerit-tests/ directory for any issues: missing required files, broken CWD paths in test definitions, stale results older than 7 days, tests with success_re that would match empty strings. Fix anything trivially fixable. Write a report of what you found and what you fixed.",
        "min_tools": 3, "success_re": r"(audit|issue|found|fixed|stale|broken|cerit-tests)", "timeout": 360,
    },
    {
        "id": "K02", "suite": "ext", "category": "K",
        "name": "Autonomous: diagnose Zotero agent",
        "cwd": ZOTERO,
        "prompt": "The Zotero agent has likely logged some errors recently. Find any error indications in agent.log, agent_run.log, or agent_state.json. Diagnose the root cause. If it's a code bug, fix it. If it's a configuration issue, explain the fix. Write a clear bug report to /tmp/zotero_bug_report.txt.",
        "min_tools": 3, "success_re": r"(error|diagnose|bug.report|zotero|root.cause|fix)", "timeout": 500,
    },
    {
        "id": "K03", "suite": "ext", "category": "K",
        "name": "Autonomous: benchmark regression report",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Examine all benchmark runs in results/. For any test that appears in multiple runs, check if performance regressed (wall time increased >50%, or completion dropped from Y to N). Find the root cause where possible. Write a regression report to /tmp/regression_report.md.",
        "min_tools": 3, "success_re": r"(regression|root.cause|regressed|benchmark|report)", "timeout": 360,
    },
    {
        "id": "K04", "suite": "ext", "category": "K",
        "name": "Autonomous: add 3 new tests",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Read run_tests.py and the existing test suite. Identify 3 important scenario types NOT currently covered. Write the 3 new test definitions (as Python dicts in the TESTS_BASE or TESTS_EXT list format) and add them to the file. The tests must use real paths that exist on this machine.",
        "min_tools": 3, "success_re": r"(new test|added|scenario|TESTS|id.*T\d|id.*[A-K]\d)", "timeout": 360,
    },
    {
        "id": "K05", "suite": "ext", "category": "K",
        "name": "Autonomous: ASCII status dashboard",
        "cwd": str(HOME / "dev/cerit-tests"),
        "prompt": "Create /tmp/cerit_dashboard.py that prints a live ASCII dashboard showing: (1) proxy status (up/down, PID), (2) last benchmark run: model, completion rate, avg wall time, (3) disk usage of results/, (4) any running cerit test processes. Run it once and show the output.",
        "min_tools": 3, "success_re": r"(cerit_dashboard|proxy|benchmark|disk|dashboard|ASCII)", "timeout": 360,
    },

    # ── Category L: Model-specific verification ───────────────────────────────
    {
        "id": "L01", "suite": "ext", "category": "L", "model": "long",
        "name": "llama-4-scout tool verification",
        "cwd": DEV,
        "prompt": "Read cerit-rewrite-proxy.py. Count the total number of lines in the file and report the exact count.",
        "min_tools": 1, "success_re": r"(\d+\s+lines?|line count|total.*\d+|wc|count.*\d{3})", "timeout": 120,
    },
]

# ── Comparison suite (N01–N10) — native Anthropic vs CERIT GLM-5.2 ──────────
# CWD: COMP_DIR (clone of public github.com/mcer33/claude-code-cerit-llm repo)
# Set CERIT_COMP_DIR env or run: git clone https://github.com/mcer33/claude-code-cerit-llm ~/dev/cerit-comparison-bench
# Run both models:
#   python3 run_tests.py --suite comp --model glm
#   python3 run_tests.py --suite comp --model native

TESTS_COMP = [
    {
        "id": "N01", "suite": "comp", "category": "N",
        "name": "List proxy interventions",
        "cwd": COMP_DIR,
        "prompt": (
            "Read cerit-rewrite-proxy.py. "
            "List the six proxy interventions in order, each with a one-sentence description of what it does. "
            "Number them 1–6."
        ),
        "min_tools": 1,
        "success_re": r"(tool sanitiz|continuation|turn.*guard|thinking.*disab|fallback|retry|429)",
        "timeout": 90,
    },
    {
        "id": "N02", "suite": "comp", "category": "N",
        "name": "Grep stderr calls",
        "cwd": COMP_DIR,
        "prompt": (
            "Search cerit-rewrite-proxy.py for all lines containing 'sys.stderr.write'. "
            "Report: total count, then line numbers and content for the first 5 matches."
        ),
        "min_tools": 1,
        "success_re": r"(line \d+|sys\.stderr|\d+ (lines|occurrences|match))",
        "timeout": 60,
    },
    {
        "id": "N03", "suite": "comp", "category": "N",
        "name": "Write benchmark stats function",
        "cwd": COMP_DIR,
        "prompt": (
            "Write a Python function `compute_stats(results: list) -> dict` where each element "
            "is a dict with keys 'time_s' (float), 'completed' (bool), 'quality' (int or None). "
            "Return a dict with 'mean_time', 'completion_rate', 'mean_quality' (None if no scores). "
            "Include a proper docstring. Save to /tmp/comp_stats.py and print its full contents."
        ),
        "min_tools": 2,
        "success_re": r"(def compute_stats|/tmp/comp_stats|mean_time|completion_rate)",
        "timeout": 90,
    },
    {
        "id": "N04", "suite": "comp", "category": "N",
        "name": "Line count all .py files",
        "cwd": COMP_DIR,
        "prompt": (
            "Run wc -l on every .py file in the current directory. "
            "Report: which file has the most lines, which has the fewest, and the grand total across all files."
        ),
        "min_tools": 1,
        "success_re": r"(\d{3,}|most lines|fewest|cerit.rewrite.proxy|grand total|total)",
        "timeout": 60,
    },
    {
        "id": "N05", "suite": "comp", "category": "N",
        "name": "Extract constants + write file",
        "cwd": COMP_DIR,
        "prompt": (
            "Read cerit_prompts.py and README.md. "
            "From cerit_prompts.py: extract the exact string value of TASK_COMPLETE_MARKER. "
            "From README.md: find the GitHub repo URL (starts with https://github.com/). "
            "Write both to /tmp/comp_extract.txt in format:\n"
            "TASK_COMPLETE_MARKER=<value>\n"
            "GITHUB_URL=<url>\n"
            "Then print the file contents to confirm."
        ),
        "min_tools": 3,
        "success_re": r"(TASK_COMPLETE|github\.com|/tmp/comp_extract|mcer33)",
        "timeout": 120,
    },
    {
        "id": "N06", "suite": "comp", "category": "N",
        "name": "Edit variable + revert",
        "cwd": COMP_DIR,
        "prompt": (
            "In cerit_idle_stop_reproducer.py, find the constant N_TRIALS. "
            "Change its numeric value to 3. Read the file to confirm the change. "
            "Then revert it back to its original value (5). Read again to confirm the revert."
        ),
        "min_tools": 4,
        "success_re": r"(N_TRIALS|changed|reverted|confirmed|= 3|= 5)",
        "timeout": 120,
    },
    {
        "id": "N07", "suite": "comp", "category": "N",
        "name": "3 most recently modified files",
        "cwd": COMP_DIR,
        "prompt": (
            "Find the 3 most recently modified files in the current directory (any extension). "
            "For each: filename, human-readable modification date, file size in bytes."
        ),
        "min_tools": 1,
        "success_re": r"(\d{4}-\d{2}|\w+\.\w+|\d+ bytes?|modified|size)",
        "timeout": 60,
    },
    {
        "id": "N08", "suite": "comp", "category": "N",
        "name": "Explain run_test() function",
        "cwd": COMP_DIR,
        "prompt": (
            "Read the run_test() function in run_tests.py (it handles one test end to end). "
            "Explain in exactly 5 bullet points what happens from when the function is called "
            "to when it returns, covering: subprocess launch, timeout handling, "
            "stream-json parsing, quality judging, and returned result structure."
        ),
        "min_tools": 1,
        "success_re": r"(subprocess|stream.json|judge|timeout|result|parse)",
        "timeout": 90,
    },
    {
        "id": "N09", "suite": "comp", "category": "N",
        "name": "Write + run port-check script",
        "cwd": COMP_DIR,
        "prompt": (
            "Write a bash script at /tmp/comp_port_check.sh that: "
            "checks whether port 9999 is in use (using ss -tlnp or lsof -i :9999 or netstat), "
            "prints 'PROXY_UP PID=<pid>' if a process is listening, 'PROXY_DOWN' otherwise. "
            "Make it executable with chmod +x and run it. Show the output."
        ),
        "min_tools": 2,
        "success_re": r"(PROXY_UP|PROXY_DOWN|9999|/tmp/comp_port_check)",
        "timeout": 90,
    },
    {
        "id": "N10", "suite": "comp", "category": "N",
        "name": "Cross-file import analysis",
        "cwd": COMP_DIR,
        "prompt": (
            "Read cerit-rewrite-proxy.py and cerit_prompts.py. "
            "Identify which names (constants, strings, or functions) defined in cerit_prompts.py "
            "are imported and used in cerit-rewrite-proxy.py. "
            "For each: name, its value/purpose in cerit_prompts.py, which function in the proxy uses it."
        ),
        "min_tools": 2,
        "success_re": r"(CERIT_CONTINUATION|cerit_prompts|import|used in|function|proxy)",
        "timeout": 120,
    },
]

ALL_TESTS = TESTS_BASE + TESTS_EXT + TESTS_COMP


# ── LLM Quality Judge ─────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a precise evaluator of AI coding assistant responses. "
    "Evaluate ONLY what was asked. Be strict but fair. "
    "Return ONLY valid JSON with no other text."
)

JUDGE_USER_TMPL = """TASK PROMPT:
{prompt}

AI RESPONSE (may be truncated to 3000 chars):
{output}

Score the AI response 1-10:
10 = task 100% complete, correct, efficient, exactly what was asked
7-9 = mostly complete, minor gaps or inefficiencies
4-6 = partially complete, significant parts missing or incorrect
1-3 = task failed, wrong approach, crashes, or no meaningful output

Also list up to 3 specific issues (empty list if score >= 8).

Return ONLY this JSON (no markdown, no explanation outside the JSON):
{{"score": <int 1-10>, "rationale": "<one sentence>", "issues": ["<issue1>", "<issue2>"]}}"""


def judge_output(prompt: str, output: str, token: str) -> dict | None:
    """Call CERIT API directly (not via proxy) to score a test output."""
    if not output or not output.strip():
        return {"score": 1, "rationale": "No output produced.", "issues": ["Empty output"]}
    user_msg = JUDGE_USER_TMPL.format(
        prompt=prompt[:500],
        output=output[:3000],
    )
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
        # Extract text from Anthropic-format response
        content = data.get("content") or []
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        if not text:
            text = data.get("completion", "") or str(data)
        # Parse JSON from response
        text = text.strip()
        # Handle markdown code fences
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        result = json.loads(text)
        # Validate fields
        score = int(result.get("score", 0))
        if not 1 <= score <= 10:
            raise ValueError(f"score out of range: {score}")
        return {
            "score": score,
            "rationale": str(result.get("rationale", ""))[:200],
            "issues": list(result.get("issues", []))[:3],
        }
    except Exception as e:
        return {"score": None, "rationale": f"judge_error: {e}", "issues": []}


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

    seen_ids: dict = {}
    merged_turns: list = []
    for t in raw_asst_turns:
        mid = t.get("message", {}).get("id", "")
        if mid and mid in seen_ids:
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

    text_output = result_ev.get("result", "")
    if not text_output:
        parts = []
        for t in asst_turns:
            content = t.get("message", {}).get("content") or []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
        text_output = "\n".join(parts)

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

    sdk_duration_ms = result_ev.get("duration_ms") or result_ev.get("duration_api_ms")
    ttft_ms = result_ev.get("ttft_ms")
    cost_usd = result_ev.get("total_cost_usd") or result_ev.get("cost_usd", 0)
    session_id = result_ev.get("session_id", "")
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
    if preset in ("native", "haiku"):
        # Native Anthropic API — use real key, not CERIT token
        if ANTHROPIC_KEY_FILE.exists():
            native_key = ANTHROPIC_KEY_FILE.read_text().strip()
        else:
            native_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not native_key:
            sys.exit(f"[ERROR] Native preset needs Anthropic API key at {ANTHROPIC_KEY_FILE}")
        env["ANTHROPIC_API_KEY"] = native_key
        env["ANTHROPIC_AUTH_TOKEN"] = native_key
    else:
        env["ANTHROPIC_API_KEY"] = token
        env["ANTHROPIC_AUTH_TOKEN"] = token
    # For native/haiku presets (no ANTHROPIC_BASE_URL in preset), unset any
    # inherited ANTHROPIC_BASE_URL so requests go to the real Anthropic API.
    if "ANTHROPIC_BASE_URL" not in MODEL_PRESETS[preset]:
        env.pop("ANTHROPIC_BASE_URL", None)
    return env


_write_lock = threading.Lock()


def run_test(test: dict, env: dict, results_dir: Path, token: str,
             use_judge: bool = True) -> dict:
    tid = test["id"]
    cwd = test["cwd"]
    test_model = test.get("model")
    if test_model and test_model in MODEL_PRESETS:
        env = dict(env)
        env.update(MODEL_PRESETS[test_model])
        # Only apply CERIT token for proxy-based presets
        if "ANTHROPIC_BASE_URL" in MODEL_PRESETS[test_model]:
            env["ANTHROPIC_API_KEY"] = token
            env["ANTHROPIC_AUTH_TOKEN"] = token

    cat = test.get("category", "?")
    print(f"\n{'='*64}", flush=True)
    print(f"  {tid} [{cat}]: {test['name']}  [model={test_model or 'default'}]", flush=True)
    print(f"  cwd: {cwd}", flush=True)
    print(f"{'='*64}", flush=True)

    if not Path(cwd).exists():
        print(f"  [WARN] cwd not found, using {HOME}", flush=True)
        cwd = str(HOME)

    # Ensure proxy is alive before each test; skip for native/haiku (no proxy needed)
    is_proxy = env.get("ANTHROPIC_BASE_URL", "").startswith("http://127.0.0.1:9999")
    if is_proxy:
        try:
            import urllib.request as _ur
            _ur.urlopen("http://127.0.0.1:9999/", timeout=3).close()
        except Exception:
            print(f"  [WARN] proxy down before {tid} — restarting...", flush=True)
            subprocess.Popen(
                ["python3", str(HOME / "dev/cerit-rewrite-proxy.py")],
                stdout=subprocess.DEVNULL, stderr=open(str(PROXY_LOG), "a"),
            )
            time.sleep(4)

    proxy_offset = proxy_log_offset()
    t0 = time.time()
    timed_out = False

    try:
        result = subprocess.run(
            ["claude", "--print", "--verbose", "--dangerously-skip-permissions",
             "--output-format", "stream-json", "-p", test["prompt"]],
            cwd=cwd, env=env, capture_output=True, text=True,
            timeout=test.get("timeout", 300),
        )
        raw_stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired as e:
        raw_stdout = (e.output or b"").decode("utf-8", errors="replace")
        stderr = "TIMEOUT"
        exit_code = -1
        timed_out = True

    wall_time = round(time.time() - t0, 2)
    print(f"  done in {wall_time}s  exit={exit_code}  timed_out={timed_out}", flush=True)

    (results_dir / f"{tid}_events.jsonl").write_text(raw_stdout)
    stream_metrics = parse_stream_json(raw_stdout)
    proxy_events = parse_proxy_log_delta(proxy_offset)

    text_output = stream_metrics.pop("text_output", "")
    (results_dir / f"{tid}_output.txt").write_text(
        text_output + ("\n\n[STDERR]\n" + stderr if stderr else "")
    )

    pattern = test.get("success_re", ".")
    task_complete_signaled = bool(re.search(r"^TASK_COMPLETE$", text_output, re.MULTILINE))
    success_re_match = bool(re.search(pattern, text_output, re.IGNORECASE))
    # task_complete_signaled acts as OR only when min_tools threshold is met,
    # guarding against the model outputting the marker without doing any work.
    min_tools_met = stream_metrics.get("n_tool_calls", 0) >= test.get("min_tools", 1)
    completed = exit_code == 0 and (success_re_match or (task_complete_signaled and min_tools_met))

    # LLM quality judge
    quality = None
    if use_judge:
        print(f"  [judge] scoring output...", flush=True)
        quality = judge_output(test["prompt"], text_output, token)
        score_str = f"score={quality.get('score')}" if quality else "score=N/A"
        print(f"  [judge] {score_str}: {(quality or {}).get('rationale', '')[:80]}", flush=True)

    print(f"  turns={stream_metrics.get('n_turns')}  tools={stream_metrics.get('n_tool_calls')}  "
          f"idle={stream_metrics.get('n_idle_stops')}  completed={completed}", flush=True)
    print(f"  proxy: {proxy_events}", flush=True)

    row = {
        "test_id": tid,
        "test_name": test["name"],
        "category": cat,
        "suite": test.get("suite", "?"),
        "cwd": cwd,
        "wall_time_sec": wall_time,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "completed": completed,
        "task_complete_signaled": task_complete_signaled,
        "output_chars": len(text_output),
        "stderr_snippet": (stderr or "")[:200],
        "proxy_events": proxy_events,
        "quality": quality,
        **stream_metrics,
    }
    return row


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate(all_metrics: list[dict]) -> dict:
    completed = [m for m in all_metrics if m.get("completed")]
    n = len(all_metrics)
    nc = len(completed)

    def safe_avg(key):
        vals = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    scores = [m["quality"]["score"] for m in all_metrics
              if m.get("quality") and m["quality"].get("score") is not None]
    avg_quality = round(sum(scores) / len(scores), 2) if scores else None

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

    by_cat: dict[str, dict] = {}
    for m in all_metrics:
        cat = m.get("category", "?")
        if cat not in by_cat:
            by_cat[cat] = {"n": 0, "nc": 0, "total_time": 0.0, "scores": []}
        by_cat[cat]["n"] += 1
        if m.get("completed"):
            by_cat[cat]["nc"] += 1
        by_cat[cat]["total_time"] += m.get("wall_time_sec", 0)
        q = m.get("quality") or {}
        if q.get("score") is not None:
            by_cat[cat]["scores"].append(q["score"])

    cat_summary = {}
    for cat, d in sorted(by_cat.items()):
        avg_q = round(sum(d["scores"]) / len(d["scores"]), 1) if d["scores"] else None
        cat_summary[cat] = {
            "completion": f"{d['nc']}/{d['n']}",
            "total_time_s": round(d["total_time"], 1),
            "avg_quality": avg_q,
        }

    return {
        "n_tests": n,
        "n_completed": nc,
        "completion_rate": round(nc / n, 2) if n else 0,
        "avg_wall_time_sec": safe_avg("wall_time_sec"),
        "avg_n_turns": safe_avg("n_turns"),
        "avg_n_tool_calls": safe_avg("n_tool_calls"),
        "avg_idle_stop_rate": safe_avg("idle_stop_rate"),
        "avg_context_growth": safe_avg("context_growth_factor"),
        "avg_quality_score": avg_quality,
        "total_idle_stops": total_idle_stops,
        "total_tool_calls": total_tool_calls,
        "total_proxy_continuations": total_proxy_continuations,
        "total_proxy_fallbacks": total_proxy_fallbacks,
        "global_idle_rate": round(total_idle_stops / (total_idle_stops + total_tool_calls), 3)
            if (total_idle_stops + total_tool_calls) else 0,
        "top_tools_overall": dict(sorted(all_tools.items(), key=lambda x: -x[1])[:10]),
        "by_category": cat_summary,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global PROXY_LOG
    parser = argparse.ArgumentParser(description="CERIT workflow test runner — extended suite")
    parser.add_argument("--model", choices=list(MODEL_PRESETS), default="glm",
                        help="Model preset: glm (GLM-5.2 via proxy), native (claude-sonnet-4-6 real API), haiku (claude-haiku-4-5 real API), rich/medium/long/deep (CERIT via proxy)")
    parser.add_argument("--tests", default="",
                        help="Comma-separated test IDs to run, e.g. T1,A01,B03")
    parser.add_argument("--suite", choices=["base", "ext", "all", "comp"], default="ext",
                        help="Test suite: base (T1-T7), ext (110 new), all (117 total), comp (N01-N10 native vs CERIT comparison)")
    parser.add_argument("--category", default="",
                        help="Comma-separated categories to run, e.g. A,B,D")
    parser.add_argument("--parallel", type=int, default=1, choices=[1, 2],
                        help="Parallel test workers (max 2 to stay within CERIT 3-concurrent limit)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM quality judge (faster, no secondary API calls)")
    parser.add_argument("--proxy-log", default=str(PROXY_LOG),
                        help=f"Path to proxy log file (default: {PROXY_LOG})")
    parser.add_argument("--timeout-scale", type=float, default=1.0, metavar="FLOAT",
                        help="Multiply all test timeouts by this factor (e.g. 2.0 for overnight runs)")
    args = parser.parse_args()

    # Override PROXY_LOG if specified
    PROXY_LOG = Path(args.proxy_log)

    if not CERIT_TOKEN_FILE.exists():
        sys.exit(f"[ERROR] Missing CERIT token at {CERIT_TOKEN_FILE}")
    token = CERIT_TOKEN_FILE.read_text().strip()
    if not token:
        sys.exit("[ERROR] CERIT token file is empty")

    # Select test pool
    if args.suite == "base":
        pool = list(TESTS_BASE)
    elif args.suite == "ext":
        pool = list(TESTS_EXT)
    elif args.suite == "comp":
        pool = list(TESTS_COMP)
    else:
        pool = list(ALL_TESTS)

    # Filter by --tests
    if args.tests:
        run_ids = set(args.tests.split(","))
        pool = [t for t in pool if t["id"] in run_ids]

    # Filter by --category
    if args.category:
        cats = set(args.category.upper().split(","))
        pool = [t for t in pool if t.get("category", "") in cats]

    if args.timeout_scale != 1.0:
        pool = [{**t, "timeout": int(t.get("timeout", 300) * args.timeout_scale)} for t in pool]

    if not pool:
        sys.exit("[ERROR] No tests match the given filters.")

    use_judge = not args.no_judge

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    results_dir = RESULTS_BASE / f"run_{ts}_{args.model}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID    : {ts}")
    print(f"Model     : {args.model}")
    print(f"Suite     : {args.suite}")
    print(f"Tests     : {[t['id'] for t in pool]} ({len(pool)} total)")
    print(f"Parallel  : {args.parallel}")
    print(f"Judge     : {'yes (qwen3.5-122b)' if use_judge else 'disabled'}")
    print(f"Results   : {results_dir}")

    env = build_env(args.model, token)
    all_metrics: list[dict] = []
    metrics_lock = threading.Lock()

    def save_incremental():
        with metrics_lock:
            (results_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

    def task(test):
        m = run_test(test, env, results_dir, token, use_judge=use_judge)
        with metrics_lock:
            all_metrics.append(m)
        save_incremental()
        return m

    if args.parallel == 1:
        for test in pool:
            task(test)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = [ex.submit(task, t) for t in pool]
            concurrent.futures.wait(futures)

    # Sort metrics to match original pool order
    order = {t["id"]: i for i, t in enumerate(pool)}
    all_metrics.sort(key=lambda m: order.get(m["test_id"], 9999))

    agg = aggregate(all_metrics)
    (results_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (results_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

    # Summary table
    hdr = (f"{'ID':<5} {'Cat':>3} {'Name':<36} {'Time':>6} {'Turns':>5} {'Tools':>5} "
           f"{'Idle':>4} {'CtxX':>5} {'Q':>3} {'Done':>5}")
    sep = "-" * len(hdr)
    rows = [
        f"CERIT Workflow Test Suite — {ts} — model={args.model}  suite={args.suite}",
        hdr, sep,
    ]
    for m in all_metrics:
        idle_pct = (m.get("idle_stop_rate") or 0) * 100
        q = m.get("quality") or {}
        score_str = str(q.get("score", "-")) if q else "-"
        rows.append(
            f"{m['test_id']:<5} {m.get('category','?'):>3} {m['test_name']:<36} "
            f"{m['wall_time_sec']:>6.1f} "
            f"{str(m.get('n_turns', '-')):>5} {str(m.get('n_tool_calls', '-')):>5} "
            f"{idle_pct:>3.0f}% "
            f"{str(m.get('context_growth_factor', '-')):>5} "
            f"{score_str:>3} "
            f"{'Y' if m.get('completed') else 'N':>5}"
        )
    rows.append(sep)
    rows.append(
        f"  Completion: {agg['n_completed']}/{agg['n_tests']}  "
        f"Avg quality: {agg['avg_quality_score']}  "
        f"Avg idle: {agg['avg_idle_stop_rate']}  "
        f"Proxy injections: {agg['total_proxy_continuations']}"
    )
    rows.append("")
    rows.append("Category breakdown:")
    for cat, cs in agg.get("by_category", {}).items():
        rows.append(f"  [{cat}] {cs['completion']:>5}  {cs['total_time_s']:>7.1f}s  "
                    f"avg_quality={cs['avg_quality']}")

    summary = "\n".join(rows)
    print("\n" + summary + "\n")
    (results_dir / "summary.txt").write_text(summary + "\n")
    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()
