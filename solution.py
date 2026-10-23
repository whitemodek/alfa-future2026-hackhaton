"""Platform entry point (как baseline.py): python3 solution.py [task.md]."""
import json
import os
import sys
from pathlib import Path

from agent import Agent
from agent_config import Config


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 1:
        print('Usage: python3 solution.py [task.md]', file=sys.stderr)
        return 2
    task = Path(args[0] if args else 'task.md')  # baseline.py по умолчанию тоже берёт task.md
    try:
        config = Config.from_env()
        from openai import OpenAI
        # Как в baseline: base_url берётся из env, если задан. Без ключа агент не падает на старте,
        # а завершится с понятной причиной после первого запроса к API.
        client_args = {'api_key': os.environ.get('OPENAI_API_KEY') or 'missing', 'max_retries': 0}
        if os.environ.get('OPENAI_BASE_URL'):
            client_args['base_url'] = os.environ['OPENAI_BASE_URL']
        with OpenAI(**client_args) as client:
            result = Agent(task, client, config).run()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result['success'] else 1
    except Exception as exc:
        # SDK exceptions can contain auth headers/request bodies. Print only class name.
        print(f'Agent could not start or publish: {type(exc).__name__}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())