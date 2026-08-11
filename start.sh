#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$project_dir"

if [ ! -x "venv/bin/python" ] || ! venv/bin/python -c "import sys" >/dev/null 2>&1; then
    python3 -m venv --clear venv
    venv/bin/python -m pip install -r requirements.txt
fi

exec venv/bin/python bot.py
