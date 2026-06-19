# claude-code-cerit-llm

Route [Claude Code](https://github.com/anthropics/claude-code) to the free Czech academic LLM gateway at [llm.ai.e-infra.cz](https://llm.ai.e-infra.cz/) (operated by CERIT-SC / e-INFRA CZ).

Includes a local rewrite proxy, shell functions, and a benchmark suite. Benchmark winner: **GLM-5.2 with thinking disabled — 494 s / 7/7 tasks / 0% idle-stop / 0 upstream errors**.

## What's included

| File | Purpose |
|---|---|
| `cerit-rewrite-proxy.py` | Local HTTP proxy on :9999 — model rewriting, continuation injection, turn guards, GLM thinking disable, context-overflow fallback |
| `cerit-bashrc.snippet` | Shell functions: `claude-cerit-rich`, `claude-cerit-medium`, `claude-cerit-long`, `claude-cerit-ping` |
| `install.sh` | One-shot installer — copies files, appends to `~/.bashrc`, creates venv for MCP delegation |
| `run_tests.py` | Automated benchmark suite — 7 tasks, captures turns/tools/idle-stop/context-growth via `--stream-json` |
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
| `claude-cerit-rich` | glm-5.2 (default) | 110 K | Normal sessions |
| `claude-cerit-medium` | qwen3.5-122b | 230 K | 80 K–220 K sessions |
| `claude-cerit-long` | llama-4-scout | 270 K | Very long, text-only |
| `claude-cerit` | gemma4 (direct) | 110 K | Fast chat, no proxy |
| `claude-cerit-ping` | — | — | Health check |

## Benchmark results (June 2026, 7 tasks)

| | qwen3.5-122b | DeepSeek V4 Pro | GLM-5.2 ON | **GLM-5.2 OFF** |
|---|---|---|---|---|
| Total | 883 s · 7/7 | 755 s · 7/7 | 1865 s · 6/7 | **494 s · 7/7** |
| Idle-stop | 0% | 0% | 0% | **0%** |
| Upstream errors | 0 | 68 | 68 | **0** |
| Context inflation | 1.0× | 2.2–2.4× | 1.1–2.6× | **1.0×** |
| Proxy injections | 56 | 188 | 255 | **0** |

Full blog post: [michalcifra.com/blogs/BL2605-claude-via-cerit-llms/](https://michalcifra.com/blogs/BL2605-claude-via-cerit-llms/)

## How the proxy works

Five interventions per request:

1. **Tool sanitizer** — strips/converts Anthropic proprietary tool types (`computer_20241022`, `web_search_20250305`, etc.) that CERIT's vLLM rejects
2. **Continuation injection** — appends an imperative system-prompt block to prevent idle-stop (model outputting text instead of calling a tool)
3. **Turn + repetition guards** — at turn 12: synthesis nudge; at turn 20: hard stop; if any tool called ≥3× in last 30 messages: loop-break injection
4. **GLM thinking disable** — injects `chat_template_kwargs: {"enable_thinking": false}` for `glm-5.2` targets (thinking ON by default causes 900 s timeouts)
5. **Fallback chain** — on HTTP 400 context-overflow: `glm-5.2 → kimi → qwen3.5-122b`

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
