"""시스템 출력음(스피커 소리)을 캡처하는 모듈.

Windows에서는 WASAPI 루프백을 통해 별도 가상 장치 없이도 현재
재생 중인 소리(유튜브, 음악 등)를 그대로 녹음할 수 있습니다.
`soundcard` 라이브러리의 loopback 마이크 기능을 사용합니다.

오디오 신호 처리 유틸(`to_mono`, `resample`, `rms`)은 numpy만
있으면 동작하므로 단위 테스트가 가능합니다. 실제 캡처를 담당하는
`LoopbackRecorder`만 `soundcard` 의존성을 갖습니다(지연 임포트).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

TARGET_SR = 16_000  # faster-whisper가 기대하는 샘플레이트


def to_mono(audio: np.ndarray) -> np.ndarray:
    """(frames, channels) 또는 (frames,) 배열을 모노 1차원으로 변환."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        if audio.shape[1] == 1:
            return audio[:, 0]
        return audio.mean(axis=1).astype(np.float32)
    return audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """선형 보간으로 샘플레이트를 변환한다(음성 인식에는 충분한 품질)."""
    audio = np.asarray(audio, dtype=np.float32)
    if orig_sr == target_sr or audio.size == 0:
        return audio
    target_len = int(round(audio.shape[0] * target_sr / orig_sr))
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    """오디오의 RMS(에너지) 값. 무음 구간 판별에 사용."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def is_silent(audio: np.ndarray, threshold: float = 0.005) -> bool:
    """RMS가 임계값보다 작으면 무음으로 간주한다."""
    return rms(audio) < threshold


class LoopbackRecorder:
    """시스템 기본 스피커의 출력음을 일정 길이 청크로 내보낸다.

    `soundcard` 라이브러리를 지연 임포트하므로, 이 모듈을 임포트하는
    것만으로는 의존성이 강제되지 않는다(테스트 친화적).
    """

    def __init__(self, chunk_seconds: float = 5.0, target_sr: int = TARGET_SR):
        self.chunk_seconds = chunk_seconds
        self.target_sr = target_sr

    def _open_loopback(self):
        try:
            import soundcard as sc
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise RuntimeError(
                "soundcard 라이브러리가 필요합니다. `pip install soundcard`"
            ) from exc

        speaker = sc.default_speaker()
        # 기본 스피커와 같은 이름의 루프백 마이크를 찾는다.
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        return mic, 48_000  # 대부분의 Windows 장치 기본 샘플레이트

    def chunks(self) -> Iterator[np.ndarray]:
        """기본 스피커 루프백에서 모노 16kHz float32 청크를 무한히 생성."""
        mic, device_sr = self._open_loopback()
        frames = int(device_sr * self.chunk_seconds)
        with mic.recorder(samplerate=device_sr, channels=None) as rec:
            while True:
                data = rec.record(numframes=frames)  # (frames, channels)
                mono = to_mono(data)
                yield resample(mono, device_sr, self.target_sr)

    def frames(self, frame_seconds: float = 0.03) -> Iterator[np.ndarray]:
        """짧은 프레임(기본 30ms)을 연속 생성한다(VAD 분할용)."""
        mic, device_sr = self._open_loopback()
        nframes = int(device_sr * frame_seconds)
        with mic.recorder(samplerate=device_sr, channels=None) as rec:
            while True:
                data = rec.record(numframes=nframes)
                mono = to_mono(data)
                yield resample(mono, device_sr, self.target_sr)
