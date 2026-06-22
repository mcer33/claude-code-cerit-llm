"""
Verify that claude-cerit-* interactive sessions can use WebSearch and OneNote COM.

Run from this machine (NTB) with:
    python tests/verify_cerit_tools.py

Requires:
    - ~/dev/cerit-rewrite-proxy.py
    - ~/dev/cerit_prompts.py
    - ~/.config/cerit/token
    - ddgs package (pip install ddgs)
    - win32com package for OneNote checks
"""
import json, os, pathlib, subprocess, sys, time, socket

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"
results = []

def check(name, condition, detail="", skip=False):
    if skip:
        print(f"  [{SKIP}] {name}" + (f": {detail}" if detail else ""))
        return
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    results.append(condition)

def header(title):
    print(f"\n{'='*56}")
    print(f"  {title}")
    print(f"{'='*56}")


# ═══════════════════════════════════════════════════════════
# P1 — Prerequisites
# ═══════════════════════════════════════════════════════════
header("P: Prerequisites")

home = pathlib.Path.home()
token_path = home / ".config/cerit/token"
proxy_path = home / "dev/cerit-rewrite-proxy.py"
prompts_path = home / "dev/cerit_prompts.py"

check("P1: CERIT token exists", token_path.exists(), str(token_path))
check("P2: proxy script exists", proxy_path.exists(), str(proxy_path))
check("P3: cerit_prompts.py exists", prompts_path.exists(), str(prompts_path))
import importlib.util as _ilu
check("P4: ddgs available", _ilu.find_spec("ddgs") is not None)
check("P5: win32com available", _ilu.find_spec("win32com") is not None)

token = token_path.read_text().strip() if token_path.exists() else ""
if not token:
    print("FATAL: no CERIT token — aborting")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
# P2 — Start local proxy
# ═══════════════════════════════════════════════════════════
header("Proxy startup on localhost:9999")

def proxy_alive():
    try:
        s = socket.create_connection(("127.0.0.1", 9999), timeout=1)
        s.close()
        return True
    except Exception:
        return False

if proxy_alive():
    print("  Proxy already running on :9999")
    check("proxy alive before start", True)
else:
    print("  Starting proxy ...")
    env_proxy = {**os.environ, "PYTHONPATH": str(home / "dev")}
    proc = subprocess.Popen(
        [sys.executable, str(proxy_path)],
        stdout=subprocess.DEVNULL,
        stderr=open("/tmp/proxy_err_ntb.log", "w"),
        env=env_proxy,
    )
    for _ in range(8):
        time.sleep(0.5)
        if proxy_alive():
            break
    check("proxy started on :9999", proxy_alive(), f"pid={proc.pid}")


# ═══════════════════════════════════════════════════════════
# WS — WebSearch via claude-cerit-rich (CC --print)
# ═══════════════════════════════════════════════════════════
header("WS: WebSearch in claude-cerit-rich session")

cc_env = {
    **os.environ,
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999/",
    "ANTHROPIC_AUTH_TOKEN": token,
    "ANTHROPIC_API_KEY": token,
    "ANTHROPIC_MODEL": "claude-sonnet-4-6",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-sonnet-4-6",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-6",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "110000",
    "MAX_THINKING_TOKENS": "0",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "PYTHONPATH": str(home / "dev"),
}
settings = home / "dev/cerit-subagents/cerit-claude-settings.json"

ws_prompt = (
    "Use the WebSearch tool to search for 'GROMACS 2024 release notes'. "
    "Report the top result title and URL. Do not use Bash or Read."
)

t0 = time.time()
ws_args = [
    "claude", "--verbose",
    "--allowedTools", "WebSearch",
    "--output-format", "stream-json", "--print", ws_prompt,
]
if settings.exists():
    ws_args = [
        "claude", "--verbose",
        "--settings", str(settings),
        "--allowedTools", "WebSearch",
        "--output-format", "stream-json", "--print", ws_prompt,
    ]

try:
    proc_ws = subprocess.run(
        ws_args, env=cc_env, capture_output=True, text=True, timeout=120
    )
    ws_elapsed = time.time() - t0
    ws_out = proc_ws.stdout
    ws_err = proc_ws.stderr

    # Parse stream-json events
    tools_called = []
    final_answer = ""
    for line in ws_out.splitlines():
        try:
            e = json.loads(line)
            if e.get("type") == "assistant":
                for b in e.get("message", {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tools_called.append(b["name"])
            if e.get("type") == "result":
                final_answer = e.get("result", "")
        except Exception:
            pass

    check("WS1: CC exited cleanly", proc_ws.returncode == 0, f"rc={proc_ws.returncode}")
    check("WS2: finished within 120s", ws_elapsed < 120, f"{ws_elapsed:.1f}s")
    ws_tool_called = any(t in ("WebSearch", "web_search") for t in tools_called)
    check("WS3: WebSearch tool was called", ws_tool_called, f"tools={tools_called}")
    check("WS4: non-empty final answer", len(final_answer) > 30, final_answer[:120])

    # Read proxy log to see which fix path fired
    err_log = pathlib.Path("/tmp/proxy_err_ntb.log")
    log_text = err_log.read_text() if err_log.exists() else ""
    sc_fired = "web_search short-circuit" in log_text
    inj_fired = "CC returned empty" in log_text
    tc_cleared = "cleared tool_choice" in log_text
    check("WS5: no 400 error from CERIT", "400" not in ws_err or "tool_choice" not in ws_err,
          "check proxy log if failing")
    print(f"  [info] proxy: short-circuit={sc_fired}, inject={inj_fired}, tool_choice_cleared={tc_cleared}")
    if not ws_tool_called or not final_answer:
        print(f"  [debug] CC stderr: {ws_err[:400]}")
        print(f"  [debug] proxy log tail: {log_text[-400:]}")

except subprocess.TimeoutExpired:
    check("WS1: timed out", False, "120s timeout reached")
except Exception as e:
    check("WS1: exception", False, str(e))


# ═══════════════════════════════════════════════════════════
# ON — OneNote COM access
# ═══════════════════════════════════════════════════════════
header("ON: OneNote COM access")

try:
    import win32com.client, win32com.client.gencache, win32com.client.makepy
    import pywintypes, pythoncom
    import traceback

    # OneNote's IDispatch.GetTypeInfo() returns E_FAIL, so EnsureDispatch can't
    # auto-generate stubs from the live object.  We generate them from the typelib
    # directly using makepy.GenerateFromTypeLibSpec, then Dispatch picks them up.

    ONENOTE_TYPELIB = ("{0EA692EE-BB50-4E3C-AEF0-356D91732725}", 0, 1, 1, 0)
    ONENOTE_TYPELIB_GUID = "{0EA692EE-BB50-4E3C-AEF0-356D91732725}"

    # Step 0: verify typelib is reachable
    print(f"  Step 0: LoadRegTypeLib({ONENOTE_TYPELIB_GUID}, 1.1) ...")
    try:
        tlb = pythoncom.LoadRegTypeLib(ONENOTE_TYPELIB_GUID, 1, 1, 0)
        check("ON0: typelib loads OK (no TYPE_E_LIBNOTREGISTERED)", True)
    except Exception as e:
        check("ON0: typelib load FAILED", False, repr(e))
        print("       This is the root cause of 0x8002801F errors — see OneNote memory.")
        raise  # can't proceed

    # Step 1: generate stubs from typelib (not from live object — GetTypeInfo fails)
    print("  Step 1: makepy.GenerateFromTypeLibSpec for OneNote ...")
    try:
        win32com.client.makepy.GenerateFromTypeLibSpec(ONENOTE_TYPELIB, verboseLevel=0)
        check("ON1: stubs generated from typelib", True)
    except Exception as e:
        check("ON1: stub generation failed", False, repr(e)[:200])
        print(f"  [debug] {traceback.format_exc()}")

    # Step 2: create COM object — now Dispatch will find the generated stubs
    print("  Step 2: Dispatch('OneNote.Application') ...")
    on = None
    try:
        on = win32com.client.Dispatch("OneNote.Application")
        check("ON2: Dispatch succeeded", True)
    except Exception as e:
        check("ON2: Dispatch failed", False, repr(e)[:200])
        print(f"  [debug] {traceback.format_exc()}")

    if on is not None:
        # Step 3: call GetHierarchy(bstrStartNodeID, hsScope) -> bstrHierarchyXML
        # hsScope: 4 = hsNotebooks
        print("  Step 3: GetHierarchy('', 4) ...")
        try:
            xml = on.GetHierarchy("", 4)
            check("ON3: GetHierarchy returned data", bool(xml), f"len={len(xml)}")
            check("ON4: returned XML (contains '<')", "<" in str(xml), str(xml)[:80])
            print(f"  [info] hierarchy snippet: {str(xml)[:200]}")
        except Exception as e:
            check("ON3: GetHierarchy failed", False, repr(e)[:200])
            print(f"  [debug] {traceback.format_exc()}")

except ImportError:
    check("ON: win32com not available", False, "pip install pywin32", skip=True)
except Exception as e:
    if not str(e).startswith("  root cause"):
        check("ON: unexpected error", False, str(e)[:200])


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*56}")
passed = sum(results)
total = len(results)
print(f"  Result: {passed}/{total} checks passed")
print(f"{'='*56}")
if passed < total:
    sys.exit(1)
