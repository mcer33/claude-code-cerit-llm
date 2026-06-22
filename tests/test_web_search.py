"""
Test suite for proxy-side web_search handling (v2).

Tests:
  T1: _ddg_search() returns non-empty results
  T2: inject_web_search_results() is a no-op when no web_search in conversation
  T3: inject_web_search_results() is a no-op when CC already provided good results
  T4: inject_web_search_results() injects DDG results when CC returns empty result
  T5: inject_web_search_results() injects when CC returns error string (< 50 chars)
  T6: Multiple web_search calls in one conversation — each gets filled independently
  T7: Non-web_search tool_results are never touched
  T7b: WebSearch function tool — inject also handles it (not just web_search lowercase)
  T8: is_web_search_execution() correctly identifies single-tool search calls
  T9: End-to-end: run CC --print through proxy with a web-search task,
      confirm WebSearch tool is called and returns a real answer
"""
import sys, json, importlib.util, pathlib, os

# ── Load proxy module ────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "proxy", "/home/cifra/dev/cerit-rewrite-proxy.py"
)
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
import unittest.mock as mock
with mock.patch("sys.argv", ["cerit-rewrite-proxy.py"]):
    proxy_mod = importlib.util.module_from_spec(spec)
    with mock.patch("http.server.ThreadingHTTPServer"):
        try:
            spec.loader.exec_module(proxy_mod)
        except SystemExit:
            pass

_ddg_search = proxy_mod._ddg_search
inject = proxy_mod.inject_web_search_results
is_ws_exec = proxy_mod.is_web_search_execution

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    results.append(condition)

# ── T1: _ddg_search returns real content ────────────────────────────────────
print("\nT1: _ddg_search() returns non-empty results")
try:
    r = _ddg_search("GROMACS molecular dynamics software", max_results=3)
    check("T1a: returned string", isinstance(r, str))
    check("T1b: non-empty (>50 chars)", len(r) > 50, f"len={len(r)}")
    check("T1c: contains 'GROMACS' or 'molecular'",
          "gromacs" in r.lower() or "molecular" in r.lower(), r[:100])
    check("T1d: no error sentinel", "returned no results" not in r)
except Exception as e:
    check("T1: exception", False, str(e))

# ── T2: no-op when no web_search in conversation ────────────────────────────
print("\nT2: no-op when no web_search in conversation")
body_no_ws = {
    "messages": [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "ls"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "file.txt\n"}
        ]}
    ]
}
result = inject(body_no_ws)
check("T2: same object returned (identity)", result is body_no_ws)

# ── T3: no-op when CC already provided good results ─────────────────────────
print("\nT3: no-op when CC already provided good results (>=50 chars)")
good_content = ("Search result: GROMACS is a versatile package for MD simulations. "
                "Version 2024.3 released March 2024 with improved GPU performance.")
body_good = {
    "messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "ws1", "name": "web_search",
             "input": {"query": "GROMACS latest version"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "ws1", "content": good_content}
        ]}
    ]
}
result = inject(body_good)
actual = result["messages"][1]["content"][0]["content"]
check("T3: content unchanged", actual == good_content)

# ── T4: injects DDG when CC returns empty result ─────────────────────────────
print("\nT4: inject DDG when CC returns empty string")
body_empty = {
    "messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "ws2", "name": "web_search",
             "input": {"query": "GROMACS 2024 release notes"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "ws2", "content": ""}
        ]}
    ]
}
result = inject(body_empty)
injected = result["messages"][1]["content"][0]["content"]
check("T4a: content was replaced", injected != "")
check("T4b: injected string is long enough", len(injected) > 50, f"len={len(injected)}")

# ── T5: injects when CC returns short error string ───────────────────────────
print("\nT5: inject DDG when CC returns short error (<50 chars)")
body_err = {
    "messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "ws3", "name": "web_search",
             "input": {"query": "MetaCentrum HPC Czech Republic"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "ws3", "content": "Error: tool not found"}
        ]}
    ]
}
result = inject(body_err)
injected = result["messages"][1]["content"][0]["content"]
check("T5a: content replaced (was error)", injected != "Error: tool not found")
check("T5b: injected is longer", len(injected) > 50, f"len={len(injected)}")

# ── T6: multiple web_search calls ────────────────────────────────────────────
print("\nT6: two web_search calls in one conversation")
body_multi = {
    "messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "a1", "name": "web_search",
             "input": {"query": "GROMACS documentation"}},
            {"type": "tool_use", "id": "a2", "name": "web_search",
             "input": {"query": "AmberTools installation"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a1", "content": ""},
            {"type": "tool_result", "tool_use_id": "a2", "content": ""},
        ]}
    ]
}
result = inject(body_multi)
c = result["messages"][1]["content"]
check("T6a: first result filled", len(c[0]["content"]) > 50)
check("T6b: second result filled", len(c[1]["content"]) > 50)
check("T6c: results differ (different queries)", c[0]["content"] != c[1]["content"])

# ── T7: non-web_search tool_results untouched ────────────────────────────────
print("\nT7: non-web_search tool_results are never modified")
body_mixed = {
    "messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "b1", "name": "web_search",
             "input": {"query": "Python asyncio"}},
            {"type": "tool_use", "id": "b2", "name": "Bash",
             "input": {"command": "ls"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "b1", "content": ""},
            {"type": "tool_result", "tool_use_id": "b2", "content": ""},
        ]}
    ]
}
result = inject(body_mixed)
c = result["messages"][1]["content"]
check("T7a: web_search result filled", len(c[0]["content"]) > 50)
check("T7b: Bash result untouched (empty)", c[1]["content"] == "")

# ── T7b: WebSearch (mixed-case) is also handled by inject ────────────────────
print("\nT7b: WebSearch (function tool, mixed-case) also handled")
body_ws_func = {
    "messages": [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "c1", "name": "WebSearch",
             "input": {"query": "tubulin MD simulation force field"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "API Error: 400"}
        ]}
    ]
}
result = inject(body_ws_func)
c = result["messages"][1]["content"][0]["content"]
check("T7b-a: WebSearch result replaced", c != "API Error: 400")
check("T7b-b: replacement is longer", len(c) > 50, f"len={len(c)}")

# ── T8: is_web_search_execution() ────────────────────────────────────────────
print("\nT8: is_web_search_execution() identifies 1-tool calls correctly")

# Should return query for single-tool body
single_ws = {
    "tools": [{"name": "web_search", "description": "search"}],
    "messages": [{"role": "user", "content": "GROMACS 2024"}]
}
check("T8a: single web_search tool → returns query",
      is_ws_exec(single_ws) == "GROMACS 2024")

# Should return None for multi-tool body (normal CC request)
multi_tools = {
    "tools": [{"name": "WebSearch"}, {"name": "Bash"}, {"name": "Read"}],
    "messages": [{"role": "user", "content": "search gromacs"}]
}
check("T8b: multi-tool body → returns None", is_ws_exec(multi_tools) is None)

# Should return None for no-tool body
no_tools = {
    "tools": [],
    "messages": [{"role": "user", "content": "hello"}]
}
check("T8c: no-tool body → returns None", is_ws_exec(no_tools) is None)

# Should handle WebSearch (function tool name)
single_WebSearch = {
    "tools": [{"name": "WebSearch", "description": "search"}],
    "messages": [{"role": "user", "content": "tubulin electric field"}]
}
check("T8d: single WebSearch tool → returns query",
      is_ws_exec(single_WebSearch) == "tubulin electric field")

# ── T9: end-to-end via proxy (live test) ─────────────────────────────────────
print("\nT9: end-to-end: CC through proxy with web search task")
import subprocess, time

token = pathlib.Path.home() / ".config/cerit/token"
if not token.exists():
    check("T9: skipped (no CERIT token)", True, "SKIP")
else:
    # Clear proxy log so we can check DDG injection cleanly
    err_log = pathlib.Path("/tmp/proxy_err.log")
    if err_log.exists():
        err_log.write_text("")

    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999/",
        "ANTHROPIC_API_KEY": token.read_text().strip(),
        "ANTHROPIC_AUTH_TOKEN": token.read_text().strip(),
        "ANTHROPIC_MODEL": "claude-sonnet-4-6",
        "MAX_THINKING_TOKENS": "0",
    }
    prompt = ("Use the WebSearch tool to search for 'GROMACS 2024 release' "
              "and tell me the top result title and URL. Do not use Bash or Read.")
    t0 = time.time()
    proc = subprocess.run(
        ["claude", "--verbose", "--output-format", "stream-json", "--print", prompt],
        env=env, capture_output=True, text=True, timeout=120
    )
    elapsed = time.time() - t0
    out = proc.stdout

    # Parse events to find tool calls and final answer
    tools_called = []
    final_text = ""
    for line in out.splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
            if e.get("type") == "assistant":
                for b in (e.get("message", {}).get("content") or []):
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tools_called.append(b["name"])
            if e.get("type") == "result":
                final_text = e.get("result", "")
        except Exception:
            pass

    check("T9a: CC exited successfully", proc.returncode == 0,
          f"rc={proc.returncode}\nstderr={proc.stderr[:300]}")
    check("T9b: completed in time", elapsed < 120, f"{elapsed:.1f}s")
    ws_called = "WebSearch" in tools_called or "web_search" in tools_called
    check("T9c: a web search tool was called", ws_called, f"tools={tools_called}")
    has_result = len(final_text) > 20
    check("T9d: got a non-empty final answer", has_result, final_text[:150])

    # Report which fix path fired
    log_text = err_log.read_text() if err_log.exists() else ""
    ddg_inject = "CC returned empty" in log_text
    ddg_sc = "web_search short-circuit" in log_text
    print(f"         Proxy log: inject={ddg_inject}, short-circuit={ddg_sc}")
    if not ddg_inject and not ddg_sc:
        print("         (neither fix needed — WebSearch worked natively!)")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
passed = sum(results)
total = len(results)
print(f"Result: {passed}/{total} checks passed")
if passed < total:
    sys.exit(1)
