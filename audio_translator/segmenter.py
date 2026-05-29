"""VAD(음성 활동 감지) 기반 발화 분할기.

고정 길이(예: 5초)로 자르면 문장 중간이 잘려 인식/번역 정확도가
떨어진다. 이 모듈은 **말이 시작되는 지점부터 일정 시간 침묵이 올
때까지**를 하나의 발화(utterance)로 묶어 내보낸다.

핵심 로직(`VadSegmenter`)은 numpy만 의존하므로 실제 마이크 없이도
프레임 시퀀스를 넣어 단위 테스트할 수 있다. 음성/비음성 판정은
`is_speech_fn`으로 교체 가능하다(기본: 에너지 기반, 선택: webrtcvad).
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import numpy as np

from audio_source import rms


def energy_is_speech(threshold: float = 0.01) -> Callable[[np.ndarray], bool]:
    """RMS 에너지로 음성 여부를 판정하는 기본 함수를 만든다."""

    def fn(frame: np.ndarray) -> bool:
        return rms(frame) >= threshold

    return fn


def webrtcvad_is_speech(
    sample_rate: int = 16_000, aggressiveness: int = 2
) -> Callable[[np.ndarray], bool]:
    """webrtcvad 기반 판정 함수(무료). 프레임은 10/20/30ms여야 한다.

    webrtcvad 미설치/프레임 크기 불일치 시 에너지 기반으로 폴백한다.
    """
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    fallback = energy_is_speech()

    def fn(frame: np.ndarray) -> bool:
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        try:
            return vad.is_speech(pcm, sample_rate)
        except Exception:
            return fallback(frame)

    return fn


class VadSegmenter:
    """프레임을 받아 완성된 발화 단위 오디오를 내보낸다.

    push()에 프레임을 하나씩 넣는다. 발화가 끝났다고 판단되면 그 발화의
    오디오(np.ndarray)를 반환하고, 아직 진행 중이면 None을 반환한다.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_seconds: float = 0.03,
        silence_ms: float = 700.0,
        min_speech_ms: float = 300.0,
        max_utterance_s: float = 20.0,
        speech_pad_ms: float = 200.0,
        is_speech_fn: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        self.sample_rate = sample_rate
        self.frame_ms = frame_seconds * 1000.0
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_s = max_utterance_s
        self.is_speech_fn = is_speech_fn or energy_is_speech()

        pad_frames = max(1, int(round(speech_pad_ms / self.frame_ms)))
        self._preroll: deque = deque(maxlen=pad_frames)
        self._reset()

    def _reset(self) -> None:
        self._buf: list[np.ndarray] = []
        self._in_speech = False
        self._speech_ms = 0.0
        self._silence_run_ms = 0.0
        self._preroll.clear()

    def _buffered_seconds(self) -> float:
        return sum(f.shape[0] for f in self._buf) / self.sample_rate

    def _emit(self) -> Optional[np.ndarray]:
        # 누적된 '실제 음성' 길이가 너무 짧으면 잡음으로 보고 버린다.
        if self._speech_ms < self.min_speech_ms or not self._buf:
            self._reset()
            return None
        audio = np.concatenate(self._buf).astype(np.float32)
        self._reset()
        return audio

    def push(self, frame: np.ndarray) -> Optional[np.ndarray]:
        frame = np.asarray(frame, dtype=np.float32)
        speech = self.is_speech_fn(frame)

        if not self._in_speech:
            if speech:
                # 발화 시작: 직전 프리롤(말머리 보존)을 포함해 누적 시작
                self._buf = list(self._preroll)
                self._preroll.clear()
                self._buf.append(frame)
                self._in_speech = True
                self._speech_ms = self.frame_ms
                self._silence_run_ms = 0.0
            else:
                self._preroll.append(frame)  # 말이 시작되기 전 짧은 여유 보관
            return None

        # 발화 진행 중: 현재 프레임을 누적(침묵이어도 말꼬리 보존)
        self._buf.append(frame)
        if speech:
            self._speech_ms += self.frame_ms
            self._silence_run_ms = 0.0
        else:
            self._silence_run_ms += self.frame_ms

        # 충분히 침묵하면 한 발화 종료
        if self._silence_run_ms >= self.silence_ms:
            return self._emit()
        # 너무 길어지면(끊임없는 말/음악) 강제로 끊어 실시간성 유지
        if self._buffered_seconds() >= self.max_utterance_s:
            return self._emit()
        return None

    def flush(self) -> Optional[np.ndarray]:
        """스트림 종료 시 남아있는 발화를 내보낸다."""
        if self._in_speech and self._speech_ms >= self.min_speech_ms and self._buf:
            audio = np.concatenate(self._buf).astype(np.float32)
            self._reset()
            return audio
        self._reset()
        return None
