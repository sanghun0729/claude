"""실시간 시스템 음성 → 한국어 자막 변환기 (메인 진입점).

흐름:
    [캡처 스레드] 스피커 출력음을 짧은 프레임(30ms)으로 연속 녹음
                  → VAD 세그먼터로 '말 시작~말 끝(침묵)' 단위 발화 추출 → 큐
    [메인 스레드] 큐에서 발화를 꺼내 STT(영어 기본) → 한국어 번역 → 자막

고정 길이로 자르지 않고 발화 단위로 처리하므로 문장 중간이 잘리지 않아
인식/번역 정확도가 높다. 인식이 느려도 최신 발화를 우선 처리하도록
큐가 가득 차면 오래된 발화를 버린다(실시간성 유지).
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading

from audio_source import LoopbackRecorder
from segmenter import VadSegmenter, energy_is_speech, webrtcvad_is_speech
from subtitle import SubtitlePrinter
from transcriber import Transcriber
from translator import to_korean


def _enqueue(q: "queue.Queue", item) -> None:
    if q.full():
        try:
            q.get_nowait()  # 오래된 발화 폐기
        except queue.Empty:
            pass
    q.put(item)


def _capture_loop(
    recorder: LoopbackRecorder,
    segmenter: VadSegmenter,
    frame_seconds: float,
    q: "queue.Queue",
    stop: threading.Event,
):
    for frame in recorder.frames(frame_seconds=frame_seconds):
        if stop.is_set():
            break
        utterance = segmenter.push(frame)
        if utterance is not None:
            _enqueue(q, utterance)
    rest = segmenter.flush()
    if rest is not None:
        _enqueue(q, rest)


def _make_vad(args):
    if args.vad == "webrtcvad":
        try:
            return webrtcvad_is_speech(aggressiveness=args.vad_aggressiveness)
        except Exception:
            print("webrtcvad 사용 불가 → 에너지 기반으로 대체합니다.", file=sys.stderr)
    return energy_is_speech(threshold=args.speech_threshold)


def run(args) -> int:
    recorder = LoopbackRecorder()
    transcriber = Transcriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        language=args.language,
    )
    printer = SubtitlePrinter(colored=not args.no_color)
    segmenter = VadSegmenter(
        frame_seconds=args.frame_ms / 1000.0,
        silence_ms=args.silence_ms,
        min_speech_ms=args.min_speech_ms,
        max_utterance_s=args.max_utterance,
        is_speech_fn=_make_vad(args),
    )

    q: "queue.Queue" = queue.Queue(maxsize=args.queue_size)
    stop = threading.Event()
    cap = threading.Thread(
        target=_capture_loop,
        args=(recorder, segmenter, args.frame_ms / 1000.0, q, stop),
        daemon=True,
    )

    lang_label = args.language if args.language not in (None, "auto") else "자동감지"
    print("🎧 시스템 출력음을 듣는 중입니다... (Ctrl+C 로 종료)", file=sys.stderr)
    print(
        f"   모델={transcriber.model_size}, 장치={transcriber.device}"
        f"({transcriber.compute_type}), 인식언어={lang_label}, "
        f"분할=VAD(침묵 {args.silence_ms:.0f}ms), 대상=한국어\n",
        file=sys.stderr,
    )
    cap.start()

    try:
        while True:
            utterance = q.get()
            result = transcriber.transcribe(utterance)
            if not result.text:
                continue
            translated = to_korean(result.text, source_lang=result.language)
            printer.show(
                result.text,
                translated,
                result.language,
                result.language_probability,
            )
    except KeyboardInterrupt:
        print("\n종료합니다.", file=sys.stderr)
    finally:
        stop.set()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="시스템 출력음을 실시간으로 한국어 자막으로 변환합니다."
    )
    # --- 인식 ---
    p.add_argument(
        "--language", default="en",
        help="인식 언어. en=영어 전용(기본, 더 빠르고 정확), auto=자동감지",
    )
    p.add_argument(
        "--model", default="auto",
        help="모델 (auto/tiny/base/small/medium/large-v3/large-v3-turbo/"
             "distil-large-v3). auto=영어면 distil-large-v3, 다국어는 GPU "
             "large-v3-turbo·CPU small. 기본 auto",
    )
    p.add_argument(
        "--device", default="auto",
        help="auto/cpu/cuda. auto=GPU 자동 감지. 기본 auto",
    )
    p.add_argument(
        "--compute-type", default="auto",
        help="연산 타입 auto/int8/float16. 기본 auto",
    )
    p.add_argument(
        "--beam-size", type=int, default=5,
        help="빔 서치 크기. 클수록 정확↑/속도↓. 기본 5",
    )
    # --- VAD 발화 분할 ---
    p.add_argument(
        "--vad", default="energy", choices=["energy", "webrtcvad"],
        help="음성 활동 감지 방식. energy(무의존)/webrtcvad(무료, 더 정확). 기본 energy",
    )
    p.add_argument(
        "--vad-aggressiveness", type=int, default=2,
        help="webrtcvad 민감도 0~3 (클수록 엄격). 기본 2",
    )
    p.add_argument(
        "--frame-ms", type=float, default=30.0,
        help="프레임 길이(ms). webrtcvad는 10/20/30만 허용. 기본 30",
    )
    p.add_argument(
        "--silence-ms", type=float, default=700.0,
        help="이만큼 침묵하면 한 발화 종료로 판단(ms). 기본 700",
    )
    p.add_argument(
        "--min-speech-ms", type=float, default=300.0,
        help="이보다 짧은 소리는 잡음으로 무시(ms). 기본 300",
    )
    p.add_argument(
        "--max-utterance", type=float, default=20.0,
        help="발화가 길어질 때 강제로 끊는 최대 길이(초). 기본 20",
    )
    p.add_argument(
        "--speech-threshold", type=float, default=0.01,
        help="energy VAD의 음성 판정 RMS 임계값. 기본 0.01",
    )
    # --- 기타 ---
    p.add_argument("--queue-size", type=int, default=8, help="발화 버퍼 큐 크기")
    p.add_argument("--no-color", action="store_true", help="색상 출력 비활성화")
    return p


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
