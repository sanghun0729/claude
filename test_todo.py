"""todo.py 에 대한 단위 테스트."""

from pathlib import Path

import pytest

from todo import TodoStore, format_task, main


@pytest.fixture
def store(tmp_path: Path) -> TodoStore:
    return TodoStore(path=tmp_path / "todo.json")


def test_add_assigns_incrementing_ids(store: TodoStore):
    t1 = store.add("첫 번째")
    t2 = store.add("두 번째")
    assert t1.id == 1
    assert t2.id == 2
    assert t2.title == "두 번째"
    assert t2.done is False


def test_load_returns_saved_tasks(store: TodoStore):
    store.add("일")
    store.add("이")
    tasks = store.load()
    assert [t.title for t in tasks] == ["일", "이"]


def test_mark_done(store: TodoStore):
    store.add("할 일")
    task = store.mark_done(1)
    assert task is not None
    assert task.done is True
    assert store.load()[0].done is True


def test_mark_done_missing_id_returns_none(store: TodoStore):
    assert store.mark_done(999) is None


def test_remove_existing(store: TodoStore):
    store.add("삭제 대상")
    assert store.remove(1) is True
    assert store.load() == []


def test_remove_missing_returns_false(store: TodoStore):
    assert store.remove(42) is False


def test_format_task():
    from todo import Task

    assert format_task(Task(id=1, title="x", done=False)) == "[ ] 1. x"
    assert format_task(Task(id=2, title="y", done=True)) == "[✓] 2. y"


def test_main_done_missing_returns_error_code(store: TodoStore):
    assert main(["done", "5"], store=store) == 1


def test_main_add_and_list(capsys, store: TodoStore):
    assert main(["add", "테스트"], store=store) == 0
    assert main(["list"], store=store) == 0
    out = capsys.readouterr().out
    assert "테스트" in out
