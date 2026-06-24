"""
End-to-end test: does qwen3.5-122b actually call WebSearch when
the proxy injects tool_choice:WebSearch?

Sends a multi-tool request (so the short-circuit doesn't fire) to the proxy
with model=claude-cerit-medium and a web-search intent message, then parses
the SSE response to check if the model emitted a WebSearch tool_use block.

Usage:
    python cerit-tests/test_qwen_websearch.py
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

def log_line_count():
    return len(read_log_all().splitlines())

def log_since(mark):
    lines = read_log_all().splitlines()
    return "\n".join(lines[mark:])

def parse_sse_for_tool_calls(raw: str) -> list[str]:
    """Parse SSE stream and return list of tool names called."""
    tools = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            evt = json.loads(payload)
        except Exception:
            continue
        if evt.get("type") == "content_block_start":
            block = evt.get("content_block", {})
            if block.get("type") == "tool_use":
                tools.append(block.get("name", "?"))
        elif evt.get("type") == "message_start":
            for block in evt.get("message", {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tools.append(block.get("name", "?"))
    return tools

def post_stream(messages, tools, model="claude-cerit-medium", extra=None, timeout=60):
    token_path = pathlib.Path.home() / ".config/cerit/token"
    token = token_path.read_text().strip() if token_path.exists() else "fake"
    body = {
        "model": model,
        "max_tokens": 256,
        "stream": True,
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
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as ex:
        return 0, str(ex)

# Tool definitions matching a real CC session (multi-tool so no short-circuit)
WEBSEARCH_TOOL = {
    "name": "WebSearch",
    "description": "Search the web for current information.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
}
DELEGATE_TOOL = {
    "name": "mcp__cerit-delegate__delegate_explorer",
    "description": "Search local filesystem.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
}
BASH_TOOL = {
    "name": "Bash",
    "description": "Run shell commands.",
    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
}

ALL_TOOLS = [WEBSEARCH_TOOL, DELEGATE_TOOL, BASH_TOOL]

# ── Prerequisite ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  qwen3.5-122b WebSearch honour test")
print("=" * 60)

_alive = False
for _i in range(6):
    if proxy_alive():
        _alive = True
        break
    time.sleep(0.5)
check("P1: proxy alive on :9999", _alive)
if not _alive:
    print("FATAL: proxy not running")
    sys.exit(1)

# ── T1: Multi-tool + "latest" keyword -> proxy injects tool_choice:WebSearch
#         -> does qwen actually call WebSearch?
print("\nT1: multi-tool, 'latest' keyword, model=claude-cerit-medium")
print("    Expected: proxy injects tool_choice:WebSearch, qwen calls WebSearch")
messages_t1 = [{"role": "user", "content": "What is the latest GROMACS release? Search online and report the version number."}]

log_mark = log_line_count()
t0 = time.time()
status_t1, resp_t1 = post_stream(messages_t1, ALL_TOOLS, timeout=90)
elapsed_t1 = time.time() - t0
time.sleep(0.3)
new_log = log_since(log_mark)

tools_called = parse_sse_for_tool_calls(resp_t1)
injected = "injected tool_choice:WebSearch" in new_log
shortcircuit = "web_search short-circuit" in new_log

check("T1a: proxy returned non-5xx", status_t1 not in (500, 502, 503), f"status={status_t1}")
check("T1b: finished within 90s", elapsed_t1 < 90, f"{elapsed_t1:.1f}s")
check("T1c: injection fired", injected)
check("T1d: no short-circuit (real CERIT call)", not shortcircuit)
check("T1e: model called WebSearch", "WebSearch" in tools_called, f"tools_called={tools_called}")

if not ("WebSearch" in tools_called):
    print(f"  [debug] proxy new log:\n{new_log[-500:]}")
    # Show the raw response start (SSE events)
    sse_sample = "\n".join(l for l in resp_t1.splitlines()[:30] if l.strip())
    print(f"  [debug] SSE start (30 lines):\n{sse_sample}")

# ── T2: Without web keywords -> model should call delegate_explorer instead
print("\nT2: multi-tool, file-search prompt, model=claude-cerit-medium")
print("    Expected: proxy injects tool_choice:delegate_explorer, qwen calls it")
messages_t2 = [{"role": "user", "content": "Find Python files in ~/dev that import cerit_prompts. List the file paths."}]

log_mark2 = log_line_count()
t0 = time.time()
status_t2, resp_t2 = post_stream(messages_t2, ALL_TOOLS, timeout=90)
elapsed_t2 = time.time() - t0
time.sleep(0.3)
new_log2 = log_since(log_mark2)

tools_called2 = parse_sse_for_tool_calls(resp_t2)
injected_del = "injected tool_choice:delegate_explorer" in new_log2

check("T2a: proxy returned non-5xx", status_t2 not in (500, 502, 503), f"status={status_t2}")
check("T2b: delegate_explorer injected", injected_del)
check("T2c: model called delegate_explorer", "mcp__cerit-delegate__delegate_explorer" in tools_called2, f"tools_called={tools_called2}")
if not ("mcp__cerit-delegate__delegate_explorer" in tools_called2):
    print(f"  [debug] proxy new log:\n{new_log2[-400:]}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(results)
total = len(results)
print(f"  Result: {passed}/{total} checks passed")
print(f"  Proxy log: {LOG_PATH}")
print("=" * 60)
if passed < total:
    sys.exit(1)
