"""간단한 TODO 관리 CLI 앱.

할 일을 추가/조회/완료/삭제할 수 있으며, 데이터는 JSON 파일로 저장됩니다.

사용 예시:
    python todo.py add "우유 사기"
    python todo.py list
    python todo.py done 1
    python todo.py remove 1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_STORE = Path.home() / ".todo.json"


@dataclass
class Task:
    id: int
    title: str
    done: bool = False


class TodoStore:
    """JSON 파일에 할 일을 저장하는 간단한 저장소."""

    def __init__(self, path: Path = DEFAULT_STORE):
        self.path = path

    def load(self) -> list[Task]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [Task(**item) for item in raw]

    def save(self, tasks: list[Task]) -> None:
        data = [asdict(t) for t in tasks]
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, title: str) -> Task:
        tasks = self.load()
        next_id = max((t.id for t in tasks), default=0) + 1
        task = Task(id=next_id, title=title)
        tasks.append(task)
        self.save(tasks)
        return task

    def mark_done(self, task_id: int) -> Task | None:
        tasks = self.load()
        for task in tasks:
            if task.id == task_id:
                task.done = True
                self.save(tasks)
                return task
        return None

    def remove(self, task_id: int) -> bool:
        tasks = self.load()
        new_tasks = [t for t in tasks if t.id != task_id]
        if len(new_tasks) == len(tasks):
            return False
        self.save(new_tasks)
        return True


def format_task(task: Task) -> str:
    mark = "✓" if task.done else " "
    return f"[{mark}] {task.id}. {task.title}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="간단한 TODO 관리 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    add_p = sub.add_parser("add", help="할 일 추가")
    add_p.add_argument("title", help="할 일 내용")

    sub.add_parser("list", help="할 일 목록 조회")

    done_p = sub.add_parser("done", help="할 일 완료 처리")
    done_p.add_argument("id", type=int, help="완료할 할 일의 ID")

    rm_p = sub.add_parser("remove", help="할 일 삭제")
    rm_p.add_argument("id", type=int, help="삭제할 할 일의 ID")

    return parser


def main(argv: list[str] | None = None, store: TodoStore | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = store or TodoStore()

    if args.command == "add":
        task = store.add(args.title)
        print(f"추가됨: {format_task(task)}")
    elif args.command == "list":
        tasks = store.load()
        if not tasks:
            print("할 일이 없습니다.")
        else:
            for task in tasks:
                print(format_task(task))
    elif args.command == "done":
        task = store.mark_done(args.id)
        if task:
            print(f"완료: {format_task(task)}")
        else:
            print(f"ID {args.id} 를 찾을 수 없습니다.", file=sys.stderr)
            return 1
    elif args.command == "remove":
        if store.remove(args.id):
            print(f"삭제됨: ID {args.id}")
        else:
            print(f"ID {args.id} 를 찾을 수 없습니다.", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
