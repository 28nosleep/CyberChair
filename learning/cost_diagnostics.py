"""Developer-only local LLM cost report.

Usage: python -m learning.cost_diagnostics --chat-id -100123 --data-dir data
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from .repository import ChatRepository


TICKS_PER_USD = 10_000_000_000


def _cost(ticks):
    return "n/a" if ticks is None else f"${ticks / TICKS_PER_USD:.6f}"


def format_report(report):
    lines = []
    for name in ("reply", "summary", "autonomous", "meme"):
        item = report["groups"][name]
        lines.append(
            f"{name}: calls {item['calls']} | input {item['input_tokens']} | "
            f"cached {item['cached_input_tokens']} | output {item['output_tokens']} | "
            f"reasoning {item['reasoning_tokens']} | cost {_cost(item['cost_usd_ticks'])}"
        )
    total = report["total"]
    lines.extend((
        f"TOTAL: calls {total['calls']} | input {total['input_tokens']} | "
        f"cached {total['cached_input_tokens']} | output {total['output_tokens']} | "
        f"reasoning {total['reasoning_tokens']} | cost {_cost(total['cost_usd_ticks'])}",
        f"AVG/CALL: {_cost(total['avg_cost_usd_ticks'])}",
        f"AVG/CHAT/DAY: {_cost(total['avg_cost_per_chat_day_usd_ticks'])}",
    ))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CyberChair local LLM cost diagnostics")
    parser.add_argument("--chat-id", type=int, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    since = (datetime.now(timezone.utc) - timedelta(hours=args.hours)).isoformat()
    repository = ChatRepository(args.data_dir, args.chat_id)
    print(format_report(repository.llm_usage_report(since)))


if __name__ == "__main__":
    main()
