"""
Test that the proxy correctly injects tool_choice:WebSearch on turn-0
web-search tasks.

Usage:
    python cerit-tests/test_websearch_injection.py

Reads proxy log from $TEMP/cerit-proxy.log (set by -RedirectStandardError).
Requires proxy already running on localhost:9999.
"""
import json, os, pathlib, socket, sys, time, urllib.request, urllib.error

PROXY_URL = "http://127.0.0.1:9999/v1/messages"
LOG_PATH = pathlib.Path(os.environ.get("TEMP", "/tmp")) / "cerit-proxy.log"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    results.append(condition)

def proxy_alive():
    try:
        s = socket.create_connection(("127.0.0.1", 9999), timeout=1)
        s.close()
        return True
    except Exception:
        return False

def read_log_all():
    if not LOG_PATH.exists():
        return ""
    return LOG_PATH.read_text(encoding="utf-8", errors="replace")

def read_log_tail(n=50):
    lines = read_log_all().splitlines()
    return "\n".join(lines[-n:])

def log_line_count():
    return len(read_log_all().splitlines())


def post_to_proxy(messages, tools, extra=None):
    """POST a synthetic messages/create body to the proxy, return (status, response_text)."""
    token_path = pathlib.Path.home() / ".config/cerit/token"
    token = token_path.read_text().strip() if token_path.exists() else "fake-token"
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 512,
        "messages": messages,
        "tools": tools,
        **(extra or {}),
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        PROXY_URL, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as ex:
        return 0, str(ex)

# ── Prerequisite ────────────────────────────────────────────────────────────
print("\n" + "=" * 56)
print("  WebSearch injection test")
print("=" * 56)

check("P1: proxy alive on :9999", proxy_alive())
check("P2: proxy log file exists", LOG_PATH.exists(), str(LOG_PATH))

if not proxy_alive():
    print("FATAL: proxy not running — start it first")
    sys.exit(1)

def log_since(mark):
    lines = read_log_all().splitlines()
    return "\n".join(lines[mark:])

log_lines_before = log_line_count()

# ── T1: turn-0 with "latest" keyword → should inject WebSearch ─────────────
print("\nT1: turn-0 'latest' keyword + WebSearch in tools -> inject tool_choice:WebSearch")
messages_t1 = [
    {"role": "user", "content": "What is the latest GROMACS release? Search online and report the version."}
]
tools_t1 = [
    {"name": "WebSearch", "description": "Search the web.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
]

log_mark = log_line_count()
status_t1, resp_t1 = post_to_proxy(messages_t1, tools_t1)
time.sleep(0.5)  # let proxy flush stderr
new_log = log_since(log_mark)

injected_ws = "injected tool_choice:WebSearch" in new_log
check("T1a: proxy returned non-5xx", status_t1 not in (500, 502, 503), f"status={status_t1}")
check("T1b: injection fired in log", injected_ws, "(look for '[proxy] injected tool_choice:WebSearch')")
if not injected_ws:
    print(f"  [debug] new log lines:\n{new_log[-600:]}")

# ── T2: turn-0 with "websearch" keyword → should inject WebSearch ───────────
print("\nT2: turn-0 explicit 'websearch' keyword → inject tool_choice:WebSearch")
messages_t2 = [
    {"role": "user", "content": "Use WebSearch to find the GROMACS 2024 release notes. Report the URL."}
]
log_mark2 = log_line_count()
status_t2, resp_t2 = post_to_proxy(messages_t2, tools_t1)
time.sleep(0.5)
new_log2 = log_since(log_mark2)

injected_ws2 = "injected tool_choice:WebSearch" in new_log2
check("T2a: proxy returned non-5xx", status_t2 not in (500, 502, 503), f"status={status_t2}")
check("T2b: injection fired in log", injected_ws2)
if not injected_ws2:
    print(f"  [debug] new log lines:\n{new_log2[-400:]}")

# ── T3: turn-0 WITHOUT web keywords → should NOT inject WebSearch ────────────
print("\nT3: turn-0 file-search prompt -> should inject delegate_explorer, NOT WebSearch")
messages_t3 = [
    {"role": "user", "content": "Find all Python files in ~/dev that import cerit_prompts."}
]
tools_t3 = tools_t1 + [
    {"name": "mcp__cerit-delegate__delegate_explorer", "description": "Delegate file search.", "input_schema": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}}
]
log_mark3 = log_line_count()
status_t3, resp_t3 = post_to_proxy(messages_t3, tools_t3)
time.sleep(0.5)
new_log3 = log_since(log_mark3)

injected_del = "injected tool_choice:delegate_explorer" in new_log3
not_ws = "injected tool_choice:WebSearch" not in new_log3
check("T3a: proxy returned non-5xx", status_t3 not in (500, 502, 503), f"status={status_t3}")
check("T3b: delegate_explorer injected (not WebSearch)", injected_del and not_ws)
if not (injected_del and not_ws):
    print(f"  [debug] new log lines:\n{new_log3[-400:]}")

# ── T4: turn-1 (existing assistant msg) → no injection ──────────────────────
print("\nT4: turn-1 (assistant already replied) -> no injection at all")
messages_t4 = [
    {"role": "user", "content": "latest GROMACS version? Search online."},
    {"role": "assistant", "content": "I will search for that."},
    {"role": "user", "content": "Go ahead."},
]
log_mark4 = log_line_count()
status_t4, resp_t4 = post_to_proxy(messages_t4, tools_t1)
time.sleep(0.5)
new_log4 = log_since(log_mark4)

no_injection = "injected tool_choice" not in new_log4
check("T4a: proxy returned non-5xx", status_t4 not in (500, 502, 503), f"status={status_t4}")
check("T4b: no injection on turn-1", no_injection)
if not no_injection:
    print(f"  [debug] new log lines:\n{new_log4[-400:]}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 56)
passed = sum(results)
total = len(results)
print(f"  Result: {passed}/{total} checks passed")
print(f"  Proxy log: {LOG_PATH}")
print("=" * 56)
if passed < total:
    sys.exit(1)
