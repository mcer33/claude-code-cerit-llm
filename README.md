# claude-code-cerit-llm

Route [Claude Code](https://github.com/anthropics/claude-code) to the free Czech academic LLM gateway at [llm.ai.e-infra.cz](https://llm.ai.e-infra.cz/) (operated by CERIT-SC / e-INFRA CZ).

Includes a local rewrite proxy, shell functions, and a 160-test benchmark suite. N-suite winner: **GLM-5.2 — avg quality 8.79/10 across 43 tasks, 908 s total, 0 idle stops** (beats native Sonnet 4.6, qwen3.5-122b, and DeepSeek V4 Pro).

## What's included

| File | Purpose |
|---|---|
| `cerit-rewrite-proxy.py` | Local HTTP proxy on :9999 — model rewriting, continuation injection, turn guards, GLM thinking disable, 429 retry, context-overflow fallback |
| `cerit_prompts.py` | Prompt constants used by the proxy (continuation rule, turn/repeat guards, task-complete sentinel) |
| `cerit-bashrc.snippet` | Shell functions: `claude-cerit-rich`, `claude-cerit-medium`, `claude-cerit-long`, `claude-cerit-ping` |
| `install.sh` | One-shot installer — copies files, appends to `~/.bashrc`, creates venv for MCP delegation |
| `run_tests.py` | 117-test benchmark suite with LLM quality judge; captures turns/tools/idle-stop/quality via `--stream-json` |
| `cerit_idle_stop_reproducer.py` | Idle-stop baseline measurement — N trials with/without continuation injection, JSON output |
| `subagents/` | Optional MCP delegation layer (routes heavy Read/Grep/Glob work to isolated subagents) |

## Prerequisites

- MetaCentrum or MUNI account → [get CERIT API token at llm.ai.e-infra.cz](https://llm.ai.e-infra.cz/)
- Claude Code installed: `npm i -g @anthropic-ai/claude-code`
- Python 3.10+

## Quick start

```bash
# 1. Store your CERIT token
mkdir -p ~/.config/cerit
echo 'YOUR_TOKEN' > ~/.config/cerit/token && chmod 600 ~/.config/cerit/token

# 2. Clone and install
git clone https://github.com/mcer33/claude-code-cerit-llm.git ~/dev/claude-code-cerit-llm
cd ~/dev/claude-code-cerit-llm && bash install.sh

# 3. Reload shell and verify
source ~/.bashrc && claude-cerit-ping

# 4. Launch
claude-cerit-rich
```

## Shell commands

| Command | Model (on wire) | Context | Use when |
|---|---|---|---|
| `claude-cerit-rich` | glm-5.2 (default) | 110 K | Normal sessions — recommended |
| `claude-cerit-medium` | qwen3.5-122b | 230 K | 80 K–220 K sessions |
| `claude-cerit-long` | llama-4-scout | 270 K | Very long text-only sessions |
| `claude-cerit` | gemma4 (direct) | 110 K | Fast chat, no proxy |
| `claude-cerit-ping` | — | — | Health check |

## Benchmark results

### Core benchmark (run9, June 2026 — 7 tasks)

| | qwen3.5-122b | DeepSeek V4 Pro | GLM-5.2 ON | **GLM-5.2 OFF** |
|---|---|---|---|---|
| Total | 883 s · 7/7 | 755 s · 7/7 | 1865 s · 6/7 | **494 s · 7/7** |
| Idle-stop | 0% | 0% | 0% | **0%** |
| Upstream errors | 0 | 68 | 68 | **0** |
| Context inflation | 1.0× | 2.2–2.4× | 1.1–2.6× | **1.0×** |
| Proxy injections | 56 | 188 | 255 | **0** |

### Extended benchmark (run10, June 2026 — 117 tests, GLM-5.2)

103/117 (88%) · **0 idle stops** · 947 tool calls · avg quality 5.37/10

| Category | Tests | Completion | Avg quality |
|---|---|---|---|
| D Bash/shell | 15 | 15/15 100% | **9.0** |
| C Edit/refactor | 15 | 15/15 100% | **8.0** |
| A Read/analyze | 13 | 13/13 100% | 6.2 |
| I Verify/test | 5 | 5/5 100% | 6.0 |
| J Web research | 5 | 5/5 100% | 5.3 |
| F Multi-tool | 18 | 13/18 72% | 4.4 |
| H Vibe styles | 10 | 7/10 70% | 4.3 |
| B Code gen | 16 | 14/16 88% | 3.4 |
| G Stress | 5 | 4/5 80% | 3.5 |
| E Search/audit | 10 | 8/10 80% | 2.0 |
| K Autonomous | 5 | 4/5 80% | 2.0 |

### N-suite head-to-head (43 tasks, June 2026 — 4 models)

43-task benchmark run against the public `cerit-comparison-bench` repo (identical working directory for all models). Quality judged by qwen3.5-122b (1–10 scale). Native Sonnet 4.6 included as reference baseline.

| Category (n tasks) | GLM-5.2 | DeepSeek V4 | qwen3.5-122b | Sonnet 4.6 *(native)* |
|---|---|---|---|---|
| **[N] Self-referential (10)** | 6.8 | 5.6 | 7.2 | 5.3 |
| **[A] Read/analyze (6)** | **10.0** | 9.0 | 6.8 | 7.7 |
| **[B] Code generation (4)** | 8.0 | **10.0** | 8.3 | 9.5 |
| **[C] Edit/patch (3)** | **10.0** | **10.0** | **10.0** | — |
| **[D] Bash/shell (4)** | **10.0** | **10.0** | 8.2 | 9.8 |
| **[E] Search/audit (4)** | 9.3 | 9.2 | 8.7 | **10.0** |
| **[F] Multi-tool (6)** | **10.0** | 9.2 | 6.5 | 8.3 |
| **[G] Stress (2)** | **10.0** | 9.5 | **10.0** | **10.0** |
| **[H] Explain (2)** | 2.0 | **10.0** | **10.0** | **10.0** |
| **[I] Verify (2)** | **10.0** | **10.0** | **10.0** | 9.5 |
| **Overall avg quality** | **8.79** | 8.57 | 7.97 | 8.21 |
| **Total wall time** | **908 s** | 1751 s | 1291 s | 772 s |
| **Proxy injections** | 165 | 302 | 173 | — |

GLM-5.2 wins on quality (8.79) and is the fastest CERIT model — only 17% slower than native Sonnet 4.6. DeepSeek V4 scores second (8.57) but needs 2× the wall time and nearly 2× the continuation nudges. GLM's one weak spot is [H] explanation tasks (2.0), where it briefly narrated instead of using tools; DeepSeek and qwen3.5 handle those flawlessly.

Full blog post: [michalcifra.com/blogs/BL2605-claude-via-cerit-llms/](https://michalcifra.com/blogs/BL2605-claude-via-cerit-llms/)

## How the proxy works

Six interventions per request:

1. **Tool sanitizer** — strips/converts Anthropic proprietary tool types (`computer_20241022`, `web_search_20250305`, etc.) that CERIT's vLLM rejects
2. **Continuation injection** — appends an imperative system-prompt block to prevent idle-stop; model ends its final response with `TASK_COMPLETE` when done
3. **Turn + repetition guards** — at turn 12: synthesis nudge; at turn 20: hard stop; if any tool called ≥3× in last 30 messages: loop-break injection
4. **GLM thinking disable** — injects `chat_template_kwargs: {"enable_thinking": false}` for `glm-5.2` targets (thinking ON by default causes 900 s timeouts)
5. **Fallback chain** — on HTTP 400 context-overflow: `glm-5.2 → kimi → qwen3.5-122b`
6. **HTTP 429 retry** — rate-limit retry (3×, 5 s backoff) before falling through to the fallback chain; thread-safe request counter logs every request

### Smart task routing (text-only requests)

For non-tool requests, `classify_task()` routes by content:
- Code tasks → `qwen3-coder`
- Long-context reads (>80 K tokens) → `llama-4-scout`
- Fast/short → `gemma4`

Tool-using requests always route to `DEFAULT_TOOL_TARGET` (`glm-5.2`).

## Running the benchmark

```bash
# Quick check (7 base tasks only)
python3 run_tests.py --suite base --no-judge

# Full 117-test suite with quality scoring
python3 run_tests.py --suite all

# Specific categories
python3 run_tests.py --category C,D --suite ext

# Overnight run (2× timeouts)
python3 run_tests.py --suite all --timeout-scale 2.0 --no-judge

# Measure idle-stop baseline vs. mitigated
python3 cerit_idle_stop_reproducer.py --n 10
```

Before running, edit the path variables at the top of `run_tests.py` to point to your own project directories (or set the corresponding environment variables).

## CERIT model aliases

```
agentic   → qwen3.5-122b    (256 K, guaranteed)
thinker   → kimi             (256 K, experimental)
mini      → gpt-oss-120b    (128 K, guaranteed)
```

## Access

Access requires a [MetaCentrum](https://metavo.metacentrum.cz/) or Masaryk University account.
Apply at [metacentrum.cz](https://metavo.metacentrum.cz/cs/application/index.html).

## License

MIT
