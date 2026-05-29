"""faster-whisper 기반 음성인식 + 언어 자동감지.

오디오(16kHz 모노 float32 numpy 배열)를 받아 텍스트와 감지된 언어를
돌려준다. 모델은 최초 사용 시 한 번만 로드한다(지연 로딩).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float


class Transcriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

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
            beam_size=1,
            vad_filter=True,  # 무음/잡음 구간 제거로 환각(hallucination) 완화
            language=None,    # 자동 감지
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return Transcript(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
        )
