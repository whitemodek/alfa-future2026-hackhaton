"""Platform entry point: python3 solution.py task.md."""
import json
import os
import sys
from pathlib import Path

from agent import Agent
from agent_config import Config


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print('Usage: python3 solution.py task.md', file=sys.stderr)
        return 2
    try:
        config = Config.from_env()
        if not os.environ.get('OPENAI_API_KEY') or not os.environ.get('OPENAI_BASE_URL'):
            raise ValueError('OPENAI_API_KEY and OPENAI_BASE_URL are required')
        from openai import OpenAI
        with OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_BASE_URL'],
                    max_retries=0) as client:
            result = Agent(Path(args[0]), client, config).run()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result['success'] else 1
    except Exception as exc:
        # SDK exceptions can contain auth headers/request bodies. Print only class name.
        print(f'Agent could not start or publish: {type(exc).__name__}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
