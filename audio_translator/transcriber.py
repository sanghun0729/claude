"""faster-whisper 기반 음성인식 + 언어 자동감지 (무료·로컬).

정확도 향상 포인트:
- GPU가 있으면 `large-v3`(최고 정확도)를 자동 선택, 없으면 CPU에서
  실시간에 가까운 모델을 자동 선택한다(`model_size="auto"`).
- `beam_size`를 키워 탐색 품질을 높인다(기본 5).
- 직전 인식 결과를 다음 청크의 `initial_prompt`로 넘겨 문맥(고유명사·
  말투)을 유지한다.
- VAD 필터로 무음/잡음 구간을 제거해 환각(hallucination)을 줄인다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float


def detect_device() -> tuple[str, str]:
    """사용 가능한 장치를 감지해 (device, compute_type)를 반환한다."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def recommend_model(device: str, language: Optional[str] = "en") -> str:
    """장치/언어에 맞는 정확도·속도 균형 모델을 추천한다."""
    if language == "en":
        # 영어 전용: distil-large-v3 = large급 정확도 + 최고속(영어 특화).
        return "distil-large-v3"
    # 다국어: GPU면 large급+고속 turbo, CPU면 실시간성 우선 small.
    return "large-v3-turbo" if device == "cuda" else "small"


class Transcriber:
    def __init__(
        self,
        model_size: str = "auto",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        context_chars: int = 200,
        language: Optional[str] = "en",
    ):
        if device == "auto":
            device, auto_compute = detect_device()
            if compute_type == "auto":
                compute_type = auto_compute
        elif compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        # language="auto"는 자동 감지(None)로 처리한다.
        self.language = None if language in (None, "auto") else language

        self.device = device
        self.compute_type = compute_type
        self.model_size = (
            recommend_model(device, self.language)
            if model_size == "auto"
            else model_size
        )
        self.beam_size = beam_size
        self.context_chars = context_chars
        self._model = None
        self._context = ""  # 직전 인식 결과(문맥 유지용)

    def _ensure_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - 환경 의존
                raise RuntimeError(
                    "faster-whisper가 필요합니다. `pip install faster-whisper`"
                ) from exc
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """오디오 청크를 텍스트로 변환하고 언어를 감지한다."""
        model = self._ensure_model()
        segments, info = model.transcribe(
            np.asarray(audio, dtype=np.float32),
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=True,
            initial_prompt=self._context or None,
            language=self.language,  # 고정 시 감지 생략(더 빠르고 정확)
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            # 최근 문맥만 유지(프롬프트가 너무 길어지지 않도록)
            self._context = (self._context + " " + text)[-self.context_chars :]
        return Transcript(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
        )
