"""한국어 번역 (모두 무료, API 키 불필요).

폴백 체인:
    1) Google 번역 (deep-translator, 무료 엔드포인트) — 품질 우수
    2) Google(auto 감지) 재시도
    3) MyMemory (무료) — Google이 차단/실패했을 때의 대비책

- 이미 한국어인 텍스트는 그대로 반환한다.
- 같은 문장 반복 시 캐시로 중복 호출을 막는다.
- 테스트를 위해 `translate_fn`을 주입할 수 있다(네트워크 없이 검증 가능).
"""

from __future__ import annotations

from typing import Callable, Optional

TARGET_LANG = "ko"
_MYMEMORY_TARGET = "ko-KR"

# Whisper가 내놓는 ISO 639-1 코드 → MyMemory가 요구하는 지역 코드.
# 매핑에 없으면 영어로 가정하지 않고 Google(auto)에 맡긴다.
_MYMEMORY_SOURCE = {
    "en": "en-GB", "ja": "ja-JP", "zh": "zh-CN", "es": "es-ES",
    "fr": "fr-FR", "de": "de-DE", "ru": "ru-RU", "it": "it-IT",
    "pt": "pt-PT", "vi": "vi-VN", "th": "th-TH", "id": "id-ID",
    "ar": "ar-SA", "hi": "hi-IN",
}

# (text, source_lang) -> 번역 결과
_cache: dict[tuple[str, Optional[str]], str] = {}


def _google(text: str, source: str) -> str:
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source=source, target=TARGET_LANG).translate(text)


def _mymemory(text: str, source_lang: Optional[str]) -> str:
    from deep_translator import MyMemoryTranslator

    src = _MYMEMORY_SOURCE.get(source_lang or "", "en-GB")
    return MyMemoryTranslator(source=src, target=_MYMEMORY_TARGET).translate(text)


def _default_translate(text: str, source_lang: Optional[str]) -> str:
    """무료 엔진들을 순서대로 시도한다. 모두 실패하면 원문을 반환."""
    source = source_lang or "auto"
    for attempt in (
        lambda: _google(text, source),
        lambda: _google(text, "auto"),
        lambda: _mymemory(text, source_lang),
    ):
        try:
            result = attempt()
            if result:
                return result
        except Exception:
            continue
    return text  # 모든 무료 엔진 실패 시 원문 유지


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
