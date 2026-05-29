# TODO CLI

Claude Code 능력 테스트용으로 만든 간단한 명령줄 TODO 관리 앱입니다.
순수 Python 표준 라이브러리만 사용하며, 데이터는 JSON 파일에 저장됩니다.

## 요구 사항

- Python 3.10 이상

## 사용법

```bash
# 할 일 추가
python todo.py add "우유 사기"

# 목록 조회
python todo.py list

# 완료 처리 (ID 지정)
python todo.py done 1

# 삭제 (ID 지정)
python todo.py remove 1
```

목록 출력 예시:

```
[ ] 1. 우유 사기
[✓] 2. 빨래하기
```

데이터는 기본적으로 홈 디렉터리의 `~/.todo.json` 에 저장됩니다.

## 테스트

```bash
pip install pytest
pytest
```

## 구조

| 파일 | 설명 |
| --- | --- |
| `todo.py` | CLI 진입점 및 `TodoStore` 저장소 로직 |
| `test_todo.py` | pytest 단위 테스트 |
