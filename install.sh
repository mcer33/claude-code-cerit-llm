#!/usr/bin/env bash
# Install CERIT Claude Code setup on the current machine.
# Run from: ecosystem/infra/client-setup/cerit-llm/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV="$HOME/dev"
CERIT_SUBAGENTS="$DEV/cerit-subagents"

echo "[cerit-install] Installing to $DEV ..."

mkdir -p "$DEV" "$CERIT_SUBAGENTS"

cp "$SCRIPT_DIR/cerit-rewrite-proxy.py" "$DEV/"
cp "$SCRIPT_DIR/subagents/cerit-system-prompt-append.txt" "$CERIT_SUBAGENTS/"
cp "$SCRIPT_DIR/subagents/cerit-claude-settings.json"     "$CERIT_SUBAGENTS/"
cp "$SCRIPT_DIR/subagents/cerit_subagents.py"             "$CERIT_SUBAGENTS/"
cp "$SCRIPT_DIR/subagents/mcp_delegate.py"                "$CERIT_SUBAGENTS/"

# Append bashrc functions if not already present
if ! grep -q 'claude-cerit-rich' ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    cat "$SCRIPT_DIR/cerit-bashrc.snippet" >> ~/.bashrc
    echo "[cerit-install] Appended cerit functions to ~/.bashrc"
else
    echo "[cerit-install] ~/.bashrc already has cerit functions — skipping append"
fi

# CERIT token
TOKEN_FILE="$HOME/.config/cerit/token"
if [ ! -f "$TOKEN_FILE" ]; then
    mkdir -p "$(dirname "$TOKEN_FILE")"
    echo "[cerit-install] Create your CERIT token file:"
    echo "    echo 'YOUR_TOKEN' > $TOKEN_FILE && chmod 600 $TOKEN_FILE"
else
    echo "[cerit-install] Token file already exists at $TOKEN_FILE"
fi

# Python deps for MCP delegation (optional)
PY="$(command -v python3 || command -v python)"
if [ -n "$PY" ]; then
    if [ ! -d "$CERIT_SUBAGENTS/venv" ]; then
        echo "[cerit-install] Creating venv for MCP delegation layer..."
        "$PY" -m venv "$CERIT_SUBAGENTS/venv"
        "$CERIT_SUBAGENTS/venv/bin/pip" install --quiet anthropic mcp
        echo "[cerit-install] venv ready"
    else
        echo "[cerit-install] venv already exists — skipping"
    fi
fi

echo "[cerit-install] Done. Run: source ~/.bashrc && claude-cerit-ping"
