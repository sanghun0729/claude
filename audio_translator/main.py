"""실시간 시스템 음성 → 한국어 자막 변환기 (메인 진입점).

흐름:
    [캡처 스레드] 스피커 출력음을 chunk_seconds 단위로 녹음 → 큐
    [메인 스레드] 큐에서 청크를 꺼내 무음 검사 → STT/언어감지
                  → (한국어가 아니면) 한국어 번역 → 콘솔 자막 출력

캡처와 인식을 분리해, 인식이 느려도 최신 오디오를 우선 처리하도록
큐가 가득 차면 오래된 청크를 버린다(실시간성 유지).
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading

from audio_source import LoopbackRecorder, is_silent
from subtitle import SubtitlePrinter
from transcriber import Transcriber
from translator import to_korean


def _capture_loop(recorder: LoopbackRecorder, q: "queue.Queue", stop: threading.Event):
    for chunk in recorder.chunks():
        if stop.is_set():
            break
        if q.full():
            try:
                q.get_nowait()  # 오래된 청크 폐기
            except queue.Empty:
                pass
        q.put(chunk)


def run(args) -> int:
    recorder = LoopbackRecorder(chunk_seconds=args.chunk_seconds)
    transcriber = Transcriber(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
    )
    printer = SubtitlePrinter(colored=not args.no_color)

    q: "queue.Queue" = queue.Queue(maxsize=args.queue_size)
    stop = threading.Event()
    cap = threading.Thread(
        target=_capture_loop, args=(recorder, q, stop), daemon=True
    )

    print("🎧 시스템 출력음을 듣는 중입니다... (Ctrl+C 로 종료)", file=sys.stderr)
    print(
        f"   모델={transcriber.model_size}, 장치={transcriber.device}"
        f"({transcriber.compute_type}), 청크={args.chunk_seconds}s, 대상=한국어\n",
        file=sys.stderr,
    )
    cap.start()

    try:
        while True:
            chunk = q.get()
            if is_silent(chunk, threshold=args.silence):
                continue
            result = transcriber.transcribe(chunk)
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
    p.add_argument(
        "--model", default="auto",
        help="모델 (auto/tiny/base/small/medium/large-v3/large-v3-turbo). "
             "auto=GPU면 large-v3-turbo, CPU면 small. 기본 auto",
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
    p.add_argument(
        "--chunk-seconds", type=float, default=5.0,
        help="한 번에 녹음/인식할 길이(초). 기본 5",
    )
    p.add_argument(
        "--silence", type=float, default=0.005,
        help="무음 판정 RMS 임계값. 기본 0.005",
    )
    p.add_argument("--queue-size", type=int, default=4, help="오디오 버퍼 큐 크기")
    p.add_argument("--no-color", action="store_true", help="색상 출력 비활성화")
    return p


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
