"""Test whether Grep denial correctly triggers delegate_explorer fallback.

Tests three scenarios:
  G1: Single-file Grep task — simplest delegation case
  G2: Multi-file pattern search — needs Glob+Grep, harder to delegate
  G3: Task that uses Bash(grep) — verify the Bash deny also redirects

Run from NTB with:
  python tests/test_grep_delegation.py
"""
import json, os, pathlib, socket, subprocess, sys, time

home = pathlib.Path.home()
settings = home / "dev/cerit-subagents/cerit-claude-settings.json"
token_path = home / ".config/cerit/token"
proxy_path = home / "dev/cerit-rewrite-proxy.py"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mINFO\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    results.append(condition)

def info(msg):
    print(f"  [{INFO}] {msg}")


# ── Prerequisites ─────────────────────────────────────────────────────────────
print("\n=== Prerequisites ===")
check("token exists", token_path.exists())
check("settings.json exists", settings.exists())
check("proxy exists", proxy_path.exists())
if not token_path.exists():
    print("FATAL: no token"); sys.exit(1)
token = token_path.read_text().strip()

# ── Proxy ─────────────────────────────────────────────────────────────────────
def proxy_alive():
    try:
        s = socket.create_connection(("127.0.0.1", 9999), timeout=1); s.close(); return True
    except Exception: return False

print("\n=== Proxy ===")
if proxy_alive():
    info("already running on :9999")
else:
    env_p = {**os.environ, "PYTHONPATH": str(home / "dev")}
    proc = subprocess.Popen([sys.executable, str(proxy_path)],
                            stdout=subprocess.DEVNULL,
                            stderr=open(str(home / "AppData/Local/Temp/proxy_grep_test.log"), "w"),
                            env=env_p)
    for _ in range(8):
        time.sleep(0.5)
        if proxy_alive(): break
    check("proxy started", proxy_alive(), f"pid={proc.pid}")

# ── CC environment (full cerit-rich setup including settings.json deny rules) ─
cc_env = {
    **os.environ,
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999/",
    "ANTHROPIC_AUTH_TOKEN": token,
    "ANTHROPIC_API_KEY": token,
    "ANTHROPIC_MODEL": "claude-cerit-medium",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-cerit-medium",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-cerit-medium",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-cerit-medium",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "110000",
    "MAX_THINKING_TOKENS": "0",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "PYTHONPATH": str(home / "dev"),
}

TARGET_DIR = str(home / "dev/claude-code-cerit-llm")
MCP_CONFIG = home / "dev/cerit-subagents/cerit-mcp.json"

def run_cc(prompt, label, timeout=120):
    """Run CC with settings.json (deny rules active) + cerit-mcp.json (strict).
    --strict-mcp-config ensures cerit-delegate loads ONLY for this session; subagents
    spawned by cerit_subagents.py use setting_sources=[] so they don't load it again.
    """
    args = [
        "claude", "--verbose",
        "--settings", str(settings),
        "--mcp-config", str(MCP_CONFIG),
        "--strict-mcp-config",
        "--output-format", "stream-json", "--print",
        prompt,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(args, env=cc_env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, timeout, [], [], "", f"TIMEOUT after {timeout}s"
    elapsed = time.time() - t0

    tools_called = []
    denials = []
    final_answer = ""
    for line in proc.stdout.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "assistant":
            for b in (e.get("message", {}).get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tools_called.append(b["name"])
        if e.get("type") == "tool":
            # tool_result with is_error=true = permission denied
            for b in (e.get("tool", {}).get("content") or []):
                if isinstance(b, dict):
                    txt = b.get("text", "")
                    if "permission" in txt.lower() or "denied" in txt.lower() or "not allowed" in txt.lower():
                        denials.append(txt[:120])
        if e.get("type") == "result":
            final_answer = e.get("result", "")
    return proc.returncode, elapsed, tools_called, denials, final_answer, proc.stderr


# ── G1: Simple grep task ───────────────────────────────────────────────────────
print("\n=== G1: Simple grep task (find FALLBACK_CHAIN in Python files) ===")

g1_prompt = (
    f"In the directory {TARGET_DIR}, find every Python file that contains the "
    f"string 'FALLBACK_CHAIN'. Report the file name and line number for each match. "
    f"Work directory: {TARGET_DIR}"
)

rc, elapsed, tools, denials, answer, stderr = run_cc(g1_prompt, "G1", timeout=480)

if rc is None:
    check("G1: timed out", False, "480s")
else:
    check("G1: CC exited cleanly", rc == 0, f"rc={rc}")
    check("G1: within time", elapsed < 480, f"{elapsed:.1f}s")
    used_delegate = any("delegate_explorer" in t for t in tools)
    used_grep_directly = "Grep" in tools
    check("G1: delegate_explorer was called", used_delegate, f"tools={tools}")
    check("G1: Grep NOT called directly (denied)", not used_grep_directly, f"tools={tools}")
    has_answer = "FALLBACK_CHAIN" in answer or "cerit-rewrite-proxy" in answer or len(answer) > 50
    check("G1: answer mentions result", has_answer, answer[:200])
    info(f"Tools called in order: {tools}")
    info(f"Denials: {denials[:2]}")
    info(f"Answer: {answer[:300]}")
    if stderr and "Error" in stderr:
        info(f"CC stderr snippet: {stderr[:200]}")


# ── G2: Multi-file pattern search ─────────────────────────────────────────────
print("\n=== G2: Multi-file search (count tool occurrences in tests/*.py) ===")

g2_prompt = (
    f"Search for the string 'REPEAT_GUARD' in all Python files under {TARGET_DIR}. "
    f"Report each file name and line number where it appears. "
    f"The target directory is {TARGET_DIR}."
)

rc, elapsed, tools, denials, answer, stderr = run_cc(g2_prompt, "G2", timeout=480)

if rc is None:
    check("G2: timed out", False, "480s")
else:
    check("G2: CC exited cleanly", rc == 0, f"rc={rc}")
    check("G2: within time", elapsed < 480, f"{elapsed:.1f}s")
    used_delegate = any("delegate_explorer" in t for t in tools)
    check("G2: delegate_explorer was called", used_delegate, f"tools={tools}")
    has_answer2 = "REPEAT_GUARD" in answer or "cerit-rewrite-proxy" in answer or len(answer) > 50
    check("G2: answer mentions REPEAT_GUARD", has_answer2, answer[:200])
    info(f"Tools: {tools}")
    info(f"Answer: {answer[:300]}")


# ── G3: Bash(grep) also redirects ─────────────────────────────────────────────
print("\n=== G3: Bash grep task (verify Bash grep is also denied) ===")

g3_prompt = (
    f"Run: grep -r 'DEFAULT_TOOL_TARGET' {TARGET_DIR} and report matches. "
    f"Use only shell commands."
)

rc, elapsed, tools, denials, answer, stderr = run_cc(g3_prompt, "G3", timeout=480)

if rc is None:
    check("G3: timed out", False, "480s")
else:
    check("G3: CC exited cleanly", rc == 0, f"rc={rc}")
    bash_denied = any("grep" in d.lower() or "denied" in d.lower() for d in denials)
    # Either Bash grep was denied and delegate_explorer was used, OR model skipped Bash grep
    redirected = any("delegate_explorer" in t or "delegate_bash_worker" in t for t in tools)
    check("G3: redirected to delegate tool", redirected, f"tools={tools}")
    info(f"Tools: {tools}")
    info(f"Denials: {denials[:2]}")
    info(f"Answer: {answer[:200]}")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*56}")
passed = sum(results)
total = len(results)
print(f"  Result: {passed}/{total} checks passed")
print(f"{'='*56}")
if passed < total:
    sys.exit(1)
