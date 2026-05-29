"""콘솔 실시간 자막 출력.

원문과 한국어 번역을 보기 좋게 출력하며, 직전과 동일한 줄은
중복 출력하지 않는다(가사 반복 등으로 인한 잡음 방지).
colorama가 있으면 색을 입히고, 없으면 평문으로 출력한다.
"""

from __future__ import annotations

import datetime
from typing import Optional

try:  # 색상은 선택 사항
    from colorama import Fore, Style, init as _color_init

    _color_init()
    _C_DIM = Style.DIM
    _C_RESET = Style.RESET_ALL
    _C_SRC = Fore.CYAN
    _C_KO = Fore.GREEN
except Exception:  # pragma: no cover - colorama 미설치 환경
    _C_DIM = _C_RESET = _C_SRC = _C_KO = ""


def format_subtitle(
    original: str,
    translated: str,
    language: str,
    probability: Optional[float] = None,
    timestamp: Optional[str] = None,
    colored: bool = True,
) -> str:
    """자막 한 줄(여러 줄 문자열)을 포맷한다."""
    ts = timestamp or datetime.datetime.now().strftime("%H:%M:%S")
    prob = f" {probability:.0%}" if probability is not None else ""
    header = f"[{ts}] ({language}{prob})"

    if language == "ko" or translated == original:
        body = original
    else:
        body = original

    if colored:
        lines = [f"{_C_DIM}{header}{_C_RESET}", f"{_C_SRC}{body}{_C_RESET}"]
        if translated and translated != original:
            lines.append(f"{_C_KO}→ {translated}{_C_RESET}")
    else:
        lines = [header, body]
        if translated and translated != original:
            lines.append(f"→ {translated}")
    return "\n".join(lines)


class SubtitlePrinter:
    """중복 제거 기능이 있는 콘솔 자막 출력기."""

    def __init__(self, colored: bool = True):
        self.colored = colored
        self._last: Optional[str] = None

    def show(
        self,
        original: str,
        translated: str,
        language: str,
        probability: Optional[float] = None,
    ) -> bool:
        """자막을 출력한다. 직전과 같으면 건너뛰고 False를 반환."""
        original = (original or "").strip()
        if not original:
            return False
        if original == self._last:
            return False
        self._last = original
        print(
            format_subtitle(
                original, translated, language, probability, colored=self.colored
            ),
            flush=True,
        )
        return True
