# 실시간 시스템 음성 → 한국어 자막 (Windows)

PC에서 **재생되는 소리**(유튜브 영상, 음악, 영어 강의 등)를 실시간으로
캡처해 음성인식하고, 한국어가 아니면 **한국어로 번역해 콘솔에 자막처럼
출력**하는 프로그램입니다.

```
시스템 출력음(WASAPI 루프백) → 5초 단위 캡처 → 16kHz 변환
   → faster-whisper STT + 언어 자동감지
   → 한국어면 원문만 / 아니면 Google 번역 → 콘솔 자막
```

## 동작 환경

- **Windows 10/11** (WASAPI 루프백으로 가상 오디오 장치 없이 캡처)
- Python 3.10 이상
- 번역은 온라인(Google 번역)을 사용하므로 인터넷 연결 필요

## 설치

```powershell
pip install -r requirements.txt
```

> faster-whisper는 첫 실행 시 모델을 자동 다운로드합니다.
> CPU만 있어도 동작하며(`--compute-type int8`), NVIDIA GPU가 있으면
> `--device cuda --compute-type float16` 으로 훨씬 빠릅니다.

## 실행

```powershell
python main.py
```

예시 출력:

```
🎧 시스템 출력음을 듣는 중입니다... (Ctrl+C 로 종료)
   모델=small, 청크=5.0s, 대상=한국어

[14:03:21] (en 98%)
Never gonna give you up, never gonna let you down
→ 절대 너를 포기하지 않을 거야, 절대 너를 실망시키지 않을 거야
```

### 자주 쓰는 옵션

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `--model` | 모델 크기 `tiny/base/small/medium/large-v3` | `small` |
| `--device` | `cpu` 또는 `cuda` | `cpu` |
| `--compute-type` | `int8`(CPU) / `float16`(GPU) | `int8` |
| `--chunk-seconds` | 한 번에 인식할 길이(초) | `5.0` |
| `--silence` | 무음 판정 임계값(RMS) | `0.005` |
| `--no-color` | 색상 출력 끄기 | off |

빠른 반응이 필요하면 `--model base --chunk-seconds 3`,
정확도가 중요하면 `--model medium` 을 권장합니다.

## 구조

| 파일 | 역할 |
| --- | --- |
| `audio_source.py` | 스피커 루프백 캡처 + 모노/리샘플/무음 판정 |
| `transcriber.py` | faster-whisper STT + 언어 자동감지 |
| `translator.py` | Google 번역(한국어) + 중복 캐시 |
| `subtitle.py` | 콘솔 자막 포맷 + 중복 줄 제거 |
| `main.py` | 캡처/인식 스레드 오케스트레이션 |
| `test_pipeline.py` | 오디오·모델 없이 가능한 단위 테스트 |

## 테스트

```powershell
pip install pytest
pytest
```

신호 처리(리샘플·무음감지), 번역(캐시·한국어 스킵), 자막(포맷·중복 제거)
로직은 마이크나 모델 없이 검증됩니다.

## 한계 / 참고

- **노래**는 배경음악·화음 때문에 일반 음성보다 인식 정확도가 떨어질 수
  있습니다. 가사 자막용으로는 `--model medium` 이상을 권장합니다.
- 5초 고정 창으로 처리하므로 문장이 창 경계에서 잘릴 수 있습니다.
- Google 번역은 비공식 무료 엔드포인트를 사용하므로 과도한 사용 시
  일시적으로 제한될 수 있습니다.
- macOS/Linux에서 쓰려면 `audio_source.py`의 캡처 부분만 해당 OS의
  루프백 장치(BlackHole / PulseAudio monitor)에 맞게 바꾸면 됩니다.
