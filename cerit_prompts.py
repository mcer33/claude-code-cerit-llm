"""Prompt templates injected by cerit-rewrite-proxy.py.

Extracted to keep the proxy module focused on routing/sanitization. These
strings are appended to the system prompt of tool-using requests.
"""
from __future__ import annotations

# Sentinel written by the model at the end of its FINAL response when the task
# is fully complete. Detected by run_tests.py to disambiguate genuine completion
# from success_re mismatch.
TASK_COMPLETE_MARKER = "TASK_COMPLETE"

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
    "NEVER ask for clarification. NEVER ask 'what would you like me to search for?'.\n"
    "FORBIDDEN before a tool call: 'I will now' / 'Let me' / 'I'll first' / 'Next I'll'\n"
    "  / 'Shall I continue?' / 'Would you like me to proceed?' / 'Please confirm'\n"
    "BAD: \"I will search the filesystem for X.\"  GOOD: [call delegate_explorer immediately]\n"
    "end_turn is ONLY valid: (a) task 100% done — write your FULL final answer now,\n"
    "  OR (b) you need information only the human can provide.\n"
    "All other turns: CALL A TOOL. No exceptions.\n"
    "WEB SEARCH RULE (HIGHEST PRIORITY): If the task asks you to search the web,\n"
    "  find current/latest information online, or says 'WebSearch' — call WebSearch\n"
    "  IMMEDIATELY. Do NOT call delegate_explorer for internet searches.\n"
    "LOCAL FILE RULE: For any grep/find/read/cat task on LOCAL files, call\n"
    "  mcp__cerit-delegate__delegate_explorer FIRST — do NOT attempt Bash(grep),\n"
    "  Bash(find), Bash(cat), Grep, Glob, or Read first. They are blocked.\n"
    "  Do NOT waste your first tool call on Bash(ls), Bash(pwd), or Bash(echo) —\n"
    "  these reveal nothing useful. Start with delegate_explorer.\n"
    "FILE DENIED RULE: If any of those blocked tools return permission-denied\n"
    "  anyway → call mcp__cerit-delegate__delegate_explorer IMMEDIATELY.\n"
    "  Never retry the denied tool.\n"
    "If uncertain what to do: call mcp__cerit-delegate__delegate_explorer or the\n"
    "  most relevant inspection tool to gather facts, then proceed.\n"
    "Completion signal: when the task is 100% done and you are writing your\n"
    "  final text response, end it with the single line: TASK_COMPLETE\n"
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
