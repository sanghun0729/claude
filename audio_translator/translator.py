"""Google 번역(deep-translator) 기반 한국어 번역.

- 이미 한국어인 텍스트는 그대로 반환한다.
- 같은 가사/문장이 반복될 때 불필요한 API 호출을 막기 위해 캐시를 둔다.
- 테스트를 위해 `translate_fn`을 주입할 수 있다(네트워크 없이 검증 가능).
"""

from __future__ import annotations

from typing import Callable, Optional

TARGET_LANG = "ko"

# (text, source_lang) -> 번역 결과
_cache: dict[tuple[str, Optional[str]], str] = {}


def _default_translate(text: str, source_lang: Optional[str]) -> str:
    """deep-translator GoogleTranslator를 사용한 실제 번역."""
    from deep_translator import GoogleTranslator

    source = source_lang or "auto"
    try:
        return GoogleTranslator(source=source, target=TARGET_LANG).translate(text)
    except Exception:
        # 언어 코드가 지원되지 않는 등 실패 시 자동 감지로 재시도
        return GoogleTranslator(source="auto", target=TARGET_LANG).translate(text)


def to_korean(
    text: str,
    source_lang: Optional[str] = None,
    translate_fn: Callable[[str, Optional[str]], str] = _default_translate,
) -> str:
    """텍스트를 한국어로 번역한다. 이미 한국어면 그대로 반환."""
    text = (text or "").strip()
    if not text:
        return ""
    if source_lang == TARGET_LANG:
        return text

    key = (text, source_lang)
    if key in _cache:
        return _cache[key]

    result = translate_fn(text, source_lang) or text
    _cache[key] = result
    return result


def clear_cache() -> None:
    _cache.clear()
