"""Anthropic-shaped proxy for CERIT.

Listens on localhost:9999. Accepts requests as if it were the Anthropic API.
Interventions (in order of application):

1. Tool-definition sanitizer: convert/strip Anthropic proprietary tool types
   (web_search_20250305, computer_20241022, etc.); patch missing input_schema.
2. Tool-argument sanitizer: strip spurious outer quotes (gemma4 bug).
3. Agentic continuation injection: append CERIT_CONTINUATION to system prompt
   of every tool-using request. Force tool_choice={'type':'any'} to prevent
   idle text turns.
4. Model routing: rewrite claude-* model names to best CERIT model for the task.
   Gemma4 is EXCLUDED from tool-using sessions (unreliable tool-call output).
5. Context-overflow fallback: on 400 "exceeds max length", retry with next
   model in FALLBACK_CHAIN.

CERIT server config (confirmed by Lukáš Hejtmánek, 2026-06-19):
- kimi         → admin-recommended alias → kimi-k2.6, sglang, --tool-call-parser kimi_k2, tools ✓
- qwen3.5-122b → sglang, --tool-call-parser qwen3_coder, tools ✓
  (qwen3-coder / agentic / coder are aliases for qwen3.5-122b)
- deepseek-v4-pro-thinking → vLLM, --tool-call-parser deepseek_v4, --enable-auto-tool-choice,
  tools ✓, context 1M tokens (guaranteed tier)
- glm-5.2      → experimental, tools ✓, perf ≈ Opus 4.8; thinking ON by default
  (disable with chat_template_kwargs: {"enable_thinking": false})
- gpt-oss-120b → alias "mini", 128K ctx, general tasks (guaranteed tier)
All deployed models have tools and parsers configured. Idle-stop (33%) is
therefore model-side behaviour, not server config — mitigated by CERIT_CONTINUATION.
Note: kimi's thinking phase without tool_choice:any takes 500+ s; empirically
qwen3.5-122b (qwen3-coder alias) is the best DEFAULT_TOOL_TARGET for Claude Code tasks.

Run: python cerit-rewrite-proxy.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urlreq
from urllib.error import HTTPError

UPSTREAM = "https://llm.ai.e-infra.cz"

# Text-only / no-tools default. gemma4 is fast.
DEFAULT_TARGET = "gemma4"

# Default for tool-using sessions. glm-5.2 wins all benchmarks with thinking disabled:
# 494s total / 7/7 / 0% idle / 0 upstream errors / 0 context inflation (run9, 2026-06-19).
# Proxy injects chat_template_kwargs enable_thinking=false for glm-5.2 targets.
# Fallback: glm-5.2 → kimi → qwen3.5-122b (glm is experimental tier).
# Prior default qwen3-coder (qwen3.5-122b) was reliable but slower (883s total, run6).
DEFAULT_TOOL_TARGET = "glm-5.2"

# Explicit overrides: incoming spoofed model name → CERIT model.
# All claude-cerit-* bash functions use spoofed names so Claude Code's TUI
# shows a spinner; the proxy rewrites them to the real CERIT model.
MODEL_OVERRIDES = {
    "claude-cerit-medium":  "qwen3.5-122b",                    # sglang, 122B MoE, tools ✓
    "claude-cerit-agentic": "agentic",                         # CERIT alias → qwen3.5-122b
    "claude-cerit-long":    "llama-4-scout-17b-16e-instruct",  # long context
    "claude-cerit-kimi":    "kimi",                            # admin alias → kimi-k2.6 sglang
    "claude-cerit-deep":    "deepseek-v4-pro-thinking",        # vLLM, 1M ctx, tools ✓
    "claude-cerit-glm":     "glm-5.2",                         # ≈ Opus 4.8; thinking disabled by proxy
}

# "qwen3-coder" / "agentic" / "coder" are aliases for "qwen3.5-122b" on CERIT.
# "kimi" is admin-recommended alias → kimi-k2.6 (use "kimi", not "kimi-k2.6" directly).

FALLBACK_CHAIN = {
    "gemma4":                            ["kimi", "qwen3.5-122b"],
    "kimi":                              ["qwen3.5-122b", "glm-5.2"],
    "qwen3.5-122b":                      ["kimi", "glm-5.2"],
    "qwen3-coder":                       ["kimi", "qwen3.5-122b"],            # alias fallback
    "agentic":                           ["kimi", "qwen3.5-122b"],            # alias fallback
    "glm-5.2":                           ["kimi", "qwen3.5-122b"],
    "deepseek-v4-pro-thinking":          ["kimi", "qwen3.5-122b"],
    "llama-4-scout-17b-16e-instruct":    ["qwen3.5-122b", "kimi"],
}

# Models confirmed to lack tool support. llama-4-scout tool status unconfirmed;
# kept here until verified. All other CERIT models have tools + parsers enabled
# (confirmed by admin 2026-06-19).
NO_TOOL_SUPPORT_TARGETS = {"llama-4-scout-17b-16e-instruct"}

TOKEN_FILE = os.path.expanduser("~/.config/cerit/token")
with open(TOKEN_FILE, "r", encoding="utf-8") as f:
    CERIT_TOKEN = f.read().strip()
if not CERIT_TOKEN:
    sys.exit(f"[proxy] ERROR: token file {TOKEN_FILE} is empty — add your CERIT API key")

# ── Agentic continuation rule ─────────────────────────────────────────────────
# Injected at the END of the system prompt of every tool-using request.
# Imperative form: research showed polite phrasings have lower compliance than
# direct imperatives on RLHF-trained models.
CERIT_CONTINUATION = (
    "\n\n"
    "────────────────────────────────────────────────────────────\n"
    "AGENTIC EXECUTION — NON-NEGOTIABLE\n"
    "────────────────────────────────────────────────────────────\n"
    "When tools are available: call a tool immediately. Do not explain. Do not narrate.\n"
    "FORBIDDEN before a tool call: 'I will now' / 'Let me' / 'I'll first' / 'Next I'll'\n"
    "  / 'Shall I continue?' / 'Would you like me to proceed?' / 'Please confirm'\n"
    "BAD: \"I will search the filesystem for X.\"  GOOD: [call Glob immediately]\n"
    "end_turn is ONLY valid: (a) task 100% done — write your FULL final answer now,\n"
    "  OR (b) you need information only the human can provide.\n"
    "All other turns: CALL A TOOL. No exceptions.\n"
    "If uncertain what to do: call delegate_explorer or the most relevant\n"
    "  inspection tool to gather facts, then proceed.\n"
    "────────────────────────────────────────────────────────────\n"
)

# Turn-count and repetition guard — appended as additional system text when triggered.
# TURN_WARN_SOFT fires at turn 12: gentle nudge toward synthesis.
# TURN_WARN_HARD fires at turn 20: imperative stop.
# REPEAT_GUARD fires when any single tool is called ≥3 times in the last 30 messages.
# Threshold is 3 (not 4): GLM-style models call 9+ tools per turn, so 4 fires too late.
TURN_WARN_SOFT_TMPL = (
    "\n\n[PROGRESS CHECK — turn {n}] You have taken {n} turns. "
    "If the core task is substantially covered, stop gathering and write your FULL "
    "synthesis now as a text response. Every turn past this point must move toward "
    "closure, not additional exploration.\n"
)
TURN_WARN_HARD_TMPL = (
    "\n\n[TURN LIMIT — turn {n}] STOP CALLING TOOLS. "
    "Write your complete final answer RIGHT NOW as a text response. "
    "You have enough information. Do not make any more tool calls.\n"
)
REPEAT_GUARD_TMPL = (
    "\n\n[REPETITION GUARD] You have called '{tool}' {count} times with similar inputs. "
    "You are in a loop. STOP. Write your synthesis from what you already know.\n"
)

MIN_MAX_TOKENS = 8192  # floor — CERIT models sometimes stop mid-sentence at 4096

OVERFLOW_RE = re.compile(
    r"(exceeds the maximum allowed length"
    r"|exceeds the model's maximum context length"
    r"|is longer than the model's context"
    r"|ContextWindowExceededError)",
    re.IGNORECASE,
)

# Claude Code /compact: long history + summarization request. Route to a strong
# reasoning model with large context (qwen3.5-122b preferred over gemma4).
COMPACT_RE = re.compile(
    r"(Your task is to create a detailed summary"
    r"|please generate a detailed, structured summary"
    r"|create a comprehensive summary of the conversation so it can be used"
    r"|summarize the conversation so far so that a new instance"
    r"|generate.*summary.*continue the conversation"
    r"|Your task is to.*summary.*conversation)",
    re.IGNORECASE,
)


# ── Smart task router ────────────────────────────────────────────────────────
CODE_TASK_RE = re.compile(
    r"(implement|refactor|fix\s+bug|write\s+a\s+(function|class|script|test)"
    r"|```[\w]*\n|\.py\b|\.ts\b|\.sh\b|def |class )",
    re.IGNORECASE,
)
CODE_TOOLS = {"Write", "Edit", "Bash", "NotebookEdit"}

TASK_ROUTE = {
    "code":      "qwen3-coder",
    "long_read": "llama-4-scout-17b-16e-instruct",
    "fast":      "gemma4",
    "default":   "gemma4",
}


def _last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            return str(content)
    return ""


def classify_task(body: dict) -> str:
    """Return routing bucket for text-only (no-tools) requests."""
    messages = body.get("messages", [])
    last_user = _last_user_text(messages)
    est_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
    tool_names = {t.get("name", "") for t in (body.get("tools") or [])}

    if tool_names & CODE_TOOLS and CODE_TASK_RE.search(last_user):
        return "code"
    if est_tokens > 80_000:
        return "long_read"
    if est_tokens < 20_000:
        return "fast"
    return "default"


# ── Tool-definition sanitizer ────────────────────────────────────────────────
# Three classes of problems in tool definitions Claude Code sends:
# 1. Anthropic proprietary types (computer_20241022, web_search_20250305, etc.)
#    — no input_schema; CERIT vLLM rejects them with 400.
#    → convert web_search_* to standard tool (model can call it, CC handles result).
#    → strip computer/text_editor/bash variants (CERIT has no GUI; CC's own tools cover these).
# 2. Any tool missing input_schema → inject permissive default.
# 3. tool_choice referencing a stripped tool → reset to "auto".

_DEFAULT_INPUT_SCHEMA = {"type": "object", "properties": {}}

_ANTHROPIC_TYPE_MAP: dict[str, tuple | None] = {
    "web_search_20250305": ("web_search", "Search the web for current information.", {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query."}},
        "required": ["query"],
    }),
    "web_search_20241022": ("web_search", "Search the web for current information.", {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query."}},
        "required": ["query"],
    }),
    "computer_20241022": None,
    "computer_20250124": None,
    "text_editor_20241022": None,
    "text_editor_20250124": None,
    "bash_20241022": None,
    "bash_20250124": None,
}


def sanitize_tool_definitions(body_obj: dict) -> dict:
    tools = body_obj.get("tools")
    if not tools:
        return body_obj
    patched: list = []
    stripped_names: set = set()
    changed = False
    for tool in tools:
        if not isinstance(tool, dict):
            patched.append(tool)
            continue
        tool_type = tool.get("type", "")
        if tool_type in _ANTHROPIC_TYPE_MAP:
            override = _ANTHROPIC_TYPE_MAP[tool_type]
            if override is None:
                stripped_names.add(tool.get("name", ""))
                sys.stderr.write(
                    f"[proxy] stripped proprietary tool {tool_type!r} "
                    f"({tool.get('name', '?')!r})\n"
                )
                changed = True
                continue
            new_name, new_desc, new_schema = override
            t = {k: v for k, v in tool.items() if k != "type"}
            t["name"] = new_name
            t.setdefault("description", new_desc)
            t["input_schema"] = new_schema
            sys.stderr.write(f"[proxy] converted {tool_type!r} -> {new_name!r}\n")
            patched.append(t)
            changed = True
            continue
        if "input_schema" not in tool:
            t = dict(tool)
            t["input_schema"] = _DEFAULT_INPUT_SCHEMA
            sys.stderr.write(f"[proxy] patched missing input_schema on {t.get('name', '?')!r}\n")
            patched.append(t)
            changed = True
        else:
            patched.append(tool)
    if not changed:
        return body_obj
    body_obj = dict(body_obj)
    body_obj["tools"] = patched
    tc = body_obj.get("tool_choice")
    if (isinstance(tc, dict) and tc.get("type") == "tool"
            and tc.get("name") in stripped_names):
        sys.stderr.write(f"[proxy] tool_choice referenced stripped tool — resetting to auto\n")
        body_obj["tool_choice"] = {"type": "auto"}
    return body_obj


# ── Tool-argument sanitizer ───────────────────────────────────────────────────
# gemma4 sometimes wraps string tool arguments in extra quotes.
_OUTER_QUOTES_RE = re.compile(r'^(["\'])(.*)\1$', re.DOTALL)


def sanitize_tool_inputs(body_obj: dict) -> dict:
    messages = body_obj.get("messages")
    if not messages:
        return body_obj
    changed = False
    new_messages = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        new_content = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                new_content.append(block)
                continue
            inp = block.get("input")
            if not isinstance(inp, dict):
                new_content.append(block)
                continue
            new_inp = {}
            block_changed = False
            for k, v in inp.items():
                if isinstance(v, str):
                    m = _OUTER_QUOTES_RE.match(v)
                    if m:
                        new_v = m.group(2)
                        sys.stderr.write(
                            f"[proxy] sanitize tool_use {block.get('name')}.{k}: "
                            f"{v!r} -> {new_v!r}\n"
                        )
                        new_inp[k] = new_v
                        block_changed = True
                    else:
                        new_inp[k] = v
                else:
                    new_inp[k] = v
            if block_changed:
                block = dict(block)
                block["input"] = new_inp
                changed = True
            new_content.append(block)
        if changed:
            msg = dict(msg)
            msg["content"] = new_content
        new_messages.append(msg)
    if changed:
        body_obj = dict(body_obj)
        body_obj["messages"] = new_messages
    return body_obj


def looks_like_compact(body_obj: dict) -> bool:
    msgs = body_obj.get("messages", [])
    if len(msgs) < 3:
        return False
    last = msgs[-1]
    content = last.get("content", "")
    if isinstance(content, list):
        text = " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        text = str(content)
    return bool(COMPACT_RE.search(text))


def pick_target(body_obj: dict) -> str | None:
    model = body_obj.get("model", "")
    if not isinstance(model, str):
        return None
    if model in MODEL_OVERRIDES:
        return MODEL_OVERRIDES[model]
    if model.startswith("claude"):
        return DEFAULT_TARGET
    return None


def is_overflow(data: bytes) -> bool:
    try:
        return bool(OVERFLOW_RE.search(data.decode("utf-8", errors="replace")))
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[proxy] " + (fmt % args) + "\n")

    def _build_headers(self) -> dict:
        h = {}
        for name in ("content-type", "anthropic-version", "anthropic-beta", "user-agent", "accept"):
            v = self.headers.get(name)
            if v:
                h[name] = v
        h["Authorization"] = f"Bearer {CERIT_TOKEN}"
        h["x-api-key"] = CERIT_TOKEN
        h.setdefault("anthropic-version", "2023-06-01")
        return h

    def _send_request(self, method: str, body_obj: dict | None, headers: dict):
        upstream_url = UPSTREAM + self.path
        if body_obj is not None:
            body_bytes = json.dumps(body_obj).encode("utf-8")
            headers["Content-Length"] = str(len(body_bytes))
        else:
            body_bytes = None
        req = urlreq.Request(upstream_url, data=body_bytes, method=method, headers=headers)
        try:
            return urlreq.urlopen(req, timeout=600), None
        except HTTPError as e:
            data = e.read()
            ct = e.headers.get("Content-Type", "application/json") if e.headers else "application/json"
            return None, (e.code, data, ct)

    def _proxy(self, method: str):
        body = b""
        clen = int(self.headers.get("Content-Length", "0") or "0")
        if clen:
            body = self.rfile.read(clen)

        body_obj = None
        if body:
            try:
                body_obj = json.loads(body)
            except Exception:
                body_obj = None

        if isinstance(body_obj, dict):
            body_obj = sanitize_tool_inputs(body_obj)
            body_obj = sanitize_tool_definitions(body_obj)

        request_has_tools = isinstance(body_obj, dict) and bool(body_obj.get("tools"))

        if isinstance(body_obj, dict):
            # Lift max_tokens floor.
            mt = body_obj.get("max_tokens") or 0
            if mt < MIN_MAX_TOKENS:
                body_obj = dict(body_obj)
                body_obj["max_tokens"] = MIN_MAX_TOKENS
                sys.stderr.write(f"[proxy] boosted max_tokens {mt} -> {MIN_MAX_TOKENS}\n")

            if request_has_tools:
                body_obj = dict(body_obj)
                messages = body_obj.get("messages") or []

                # Count assistant turns (each user+assistant pair = 1 real turn).
                n_asst = sum(1 for m in messages if m.get("role") == "assistant")

                # Repetition detection: scan last 30 messages for tool_use blocks.
                tool_call_counts: dict = {}
                for msg in messages[-30:]:
                    content = msg.get("content") or []
                    if isinstance(content, list):
                        for blk in content:
                            if isinstance(blk, dict) and blk.get("type") == "tool_use":
                                nm = blk.get("name", "")
                                tool_call_counts[nm] = tool_call_counts.get(nm, 0) + 1
                repeat_tool = max(tool_call_counts, key=lambda k: tool_call_counts[k], default="")
                repeat_count = tool_call_counts.get(repeat_tool, 0)

                # Build injections.
                extra = ""
                if n_asst >= 20:
                    extra += TURN_WARN_HARD_TMPL.format(n=n_asst)
                    sys.stderr.write(f"[proxy] TURN_WARN_HARD at turn {n_asst}\n")
                elif n_asst >= 12:
                    extra += TURN_WARN_SOFT_TMPL.format(n=n_asst)
                    sys.stderr.write(f"[proxy] TURN_WARN_SOFT at turn {n_asst}\n")

                if repeat_count >= 3 and repeat_tool:
                    extra += REPEAT_GUARD_TMPL.format(tool=repeat_tool, count=repeat_count)
                    sys.stderr.write(f"[proxy] REPEAT_GUARD '{repeat_tool}' x{repeat_count}\n")

                injection = CERIT_CONTINUATION + extra

                # Inject agentic continuation rule (+ turn/repeat guards if triggered).
                existing_sys = body_obj.get("system", "")
                if isinstance(existing_sys, list):
                    existing_sys = list(existing_sys)
                    existing_sys.append({"type": "text", "text": injection})
                    body_obj["system"] = existing_sys
                else:
                    body_obj["system"] = (existing_sys or "") + injection
                sys.stderr.write("[proxy] injected continuation rule\n")
                # tool_choice:any deliberately NOT injected — causes cat-loop:
                # model wants to output final text after completing work but is
                # forced to keep calling tools (repeating `cat file.pbs` 20+ times).
                # CERIT_CONTINUATION (imperative + BAD/GOOD example) is sufficient.

        # Resolve routing
        attempts: list[str] = []
        is_compact = isinstance(body_obj, dict) and looks_like_compact(body_obj)
        if isinstance(body_obj, dict):
            initial = pick_target(body_obj)
            if initial is not None:
                if initial == DEFAULT_TARGET:
                    if is_compact:
                        # Compact: needs strong reasoning + large context.
                        initial = "qwen3.5-122b"
                        sys.stderr.write("[proxy] compact → qwen3.5-122b\n")
                    elif request_has_tools:
                        # Gemma4 excluded from tool sessions (unreliable in benchmarks).
                        # Route to kimi (admin-recommended alias → kimi-k2.7 on sglang).
                        initial = DEFAULT_TOOL_TARGET
                        sys.stderr.write(
                            f"[proxy] tool request → gemma4 excluded → {initial}\n"
                        )
                    else:
                        bucket = classify_task(body_obj)
                        routed = TASK_ROUTE[bucket]
                        if routed != DEFAULT_TARGET:
                            sys.stderr.write(f"[proxy] smart-route {bucket!r} -> {routed!r}\n")
                            initial = routed
                attempts.append(initial)
                attempts.extend(FALLBACK_CHAIN.get(initial, []))

        if request_has_tools:
            attempts = [t for t in attempts if t not in NO_TOOL_SUPPORT_TARGETS]

        if not attempts:
            attempts = [None]

        headers = self._build_headers()
        last_err = None
        for idx, target in enumerate(attempts):
            send_obj = body_obj
            if isinstance(body_obj, dict) and target is not None:
                send_obj = dict(body_obj)
                send_obj["model"] = target
                if idx == 0:
                    sys.stderr.write(
                        f"[proxy] {body_obj.get('model')!r} -> {target!r}\n"
                    )
                else:
                    sys.stderr.write(f"[proxy] fallback -> {target!r}\n")
                # GLM thinking mode is ON by default — disable for tool sessions to
                # prevent per-turn thinking storms that inflate context and wall-time.
                if target == "glm-5.2":
                    send_obj["chat_template_kwargs"] = {"enable_thinking": False}
                    sys.stderr.write("[proxy] GLM: thinking disabled\n")

            resp, err = self._send_request(method, send_obj, dict(headers))
            if resp is not None:
                self._relay_success(resp)
                return

            code, data, ct = err
            last_err = err
            if code == 400 and is_overflow(data) and idx + 1 < len(attempts):
                continue
            self._relay_error(code, data, ct)
            return

        if last_err:
            code, data, ct = last_err
            self._relay_error(code, data, ct)

    def _relay_success(self, resp):
        self.send_response(resp.status)
        skip = {"transfer-encoding", "content-encoding", "connection"}
        for k, v in resp.headers.items():
            if k.lower() in skip:
                continue
            self.send_header(k, v)
        is_sse = "text/event-stream" in (resp.headers.get("Content-Type") or "")
        if is_sse:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                # Buffer trailing incomplete UTF-8 bytes across chunk boundaries.
                # Fixed-size reads can split multi-byte chars (e.g. emoji = 3 bytes);
                # accumulate the remainder and prepend it to the next chunk.
                leftover = b""
                while True:
                    raw = resp.read(1024)
                    if not raw:
                        break
                    chunk = leftover + raw
                    # Find last valid UTF-8 boundary by decoding with error='ignore'
                    # and encoding back — difference is the incomplete tail.
                    decoded = chunk.decode("utf-8", errors="ignore")
                    safe = decoded.encode("utf-8")
                    leftover = chunk[len(safe):]
                    if not safe:
                        continue
                    self.wfile.write(("%X\r\n" % len(safe)).encode())
                    self.wfile.write(safe)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                if leftover:  # flush any remaining bytes as-is
                    self.wfile.write(("%X\r\n" % len(leftover)).encode())
                    self.wfile.write(leftover)
                    self.wfile.write(b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            except Exception as e:
                sys.stderr.write(f"[proxy] stream error: {e}\n")
        else:
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def _relay_error(self, code: int, data: bytes, ct: str):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        sys.stderr.write(f"[proxy] upstream {code}: {data[:300]!r}\n")

    def do_POST(self): self._proxy("POST")
    def do_GET(self):  self._proxy("GET")
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()


if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", "9999"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    sys.stderr.write(f"[proxy] listening on http://127.0.0.1:{port}/  ->  {UPSTREAM}\n")
    sys.stderr.write(f"[proxy] text-only default: {DEFAULT_TARGET}\n")
    sys.stderr.write(f"[proxy] tool-session default: {DEFAULT_TOOL_TARGET} "
                     f"(thinking disabled by proxy; fallback: kimi → qwen3.5-122b)\n")
    for src, dst in MODEL_OVERRIDES.items():
        sys.stderr.write(f"[proxy] override: {src} -> {dst}\n")
    for src, chain in FALLBACK_CHAIN.items():
        sys.stderr.write(f"[proxy] fallback: {src} -> {' -> '.join(chain)}\n")
    srv.serve_forever()
