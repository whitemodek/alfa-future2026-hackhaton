"""Локальный прогон «как на платформе»: распаковать ZIP в чистую папку, запустить solution.py
по каждой задаче отдельно с жёстким лимитом, проверить сабмит и метрику.

    python pack_solution.py dist/solution.zip
    python qa/gen_tasks.py qa_tasks --small
    # ключи LLM — в переменных окружения, как требует solution.py
    python qa/run_local.py dist/solution.zip qa_tasks --limit-min 20

--limit-min 180 воспроизводит настоящий лимит. Для быстрых прогонов задайте также
AGENT_TOTAL_SECONDS / AGENT_RESERVE_SECONDS, иначе агент рассчитывает на 175 минут.
ВНИМАНИЕ: команда запуска должна совпадать с baseline.py — проверьте COMMAND ниже.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
# Сверено с baseline.py: "uv run baseline.py task.md", ключи в OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
COMMAND = ["{python}", "solution.py", "{task_md}"]


def run_task(zip_path, task_dir, limit_seconds):
    work = Path(tempfile.mkdtemp(prefix="platform-"))
    code_dir, data_dir = work / "code", work / "data"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(code_dir)
    shutil.copytree(task_dir, data_dir)
    command = [part.format(python=sys.executable, task_md=str(data_dir / "task.md")) for part in COMMAND]
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=code_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        out, err = process.communicate(timeout=limit_seconds)
        killed = False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)  # платформа не ждёт
        out, err = process.communicate()
        killed = True
    elapsed = time.monotonic() - started
    # check_submission ищет ответы в ../_answers/<имя папки>
    answers_src = task_dir.parent / "_answers" / f"{task_dir.name}.csv"
    if answers_src.exists():
        (work / "_answers").mkdir(exist_ok=True)
        shutil.copy(answers_src, work / "_answers" / "data.csv")
        (work / "_answers" / "data.json").write_text(json.dumps({"kind": task_dir.name}))
    check = subprocess.run([sys.executable, str(HERE / "check_submission.py"), str(data_dir)],
                           capture_output=True, text=True)
    return {"task": task_dir.name, "exit_code": process.returncode, "killed_by_limit": killed,
            "minutes": round(elapsed / 60, 1), "check": json.loads(check.stdout or "{}"),
            "stderr_tail": err[-800:], "work_dir": str(work)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip")
    parser.add_argument("tasks_root")
    parser.add_argument("--limit-min", type=float, default=180)
    args = parser.parse_args()
    tasks = sorted(p for p in Path(args.tasks_root).iterdir() if (p / "task.md").exists())
    report = [run_task(Path(args.zip).resolve(), t.resolve(), args.limit_min * 60) for t in tasks]
    print(json.dumps(report, ensure_ascii=False, indent=2))
