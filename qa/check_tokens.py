"""Сколько токенов съел запуск агента и сколько это стоит.

    python qa/check_tokens.py <папка задачи> [цена_руб_за_1М_токенов]

Читает .agent-*/events.jsonl, который агент пишет рядом с task.md.
"""
import json
from pathlib import Path
import sys

task_dir = Path(sys.argv[1])
price = float(sys.argv[2]) if len(sys.argv) > 2 else None
for state in sorted(task_dir.glob(".agent-*")):
    events = [json.loads(line) for line in (state / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    api = [e for e in events if e["event"] == "api"]
    errors = [e for e in events if e["event"] == "api_error"]
    tokens = api[-1]["charged_tokens"] if api else 0
    last = events[-1]
    print(f"{state.name}: вызовов LLM {len(api)}, ошибок API {len(errors)}, токенов {tokens:,}, "
          f"минут {last['elapsed'] / 60:.1f}, итог: {last['event']} ({last.get('reason', '')})")
    if price:
        print(f"  ≈ {tokens / 1e6 * price:.1f} руб. при цене {price} руб/1М")
