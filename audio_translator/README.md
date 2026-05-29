# 실시간 시스템 음성 → 한국어 자막 (Windows)

PC에서 **재생되는 소리**(유튜브 영상, 음악, 영어 강의 등)를 실시간으로
캡처해 음성인식하고, 한국어가 아니면 **한국어로 번역해 콘솔에 자막처럼
출력**하는 프로그램입니다.

```
시스템 출력음(WASAPI 루프백) → 5초 단위 캡처 → 16kHz 변환
   → faster-whisper STT + 언어 자동감지 (문맥 유지·beam search)
   → 한국어면 원문만 / 아니면 무료 번역(Google→MyMemory 폴백) → 콘솔 자막
```

## 비용 — 전부 무료

| 구성요소 | 비용 | 비고 |
| --- | --- | --- |
| faster-whisper (음성인식) | 무료 | 오픈소스, 로컬 실행, 모델 자동 다운로드 |
| soundcard (소리 캡처) | 무료 | — |
| deep-translator (번역) | 무료 | **API 키 불필요**. Google 무료 + MyMemory 무료 |
| colorama (색상) | 무료 | — |

어떤 단계에도 유료 API나 키가 필요하지 않습니다.

## 정확도를 높이는 설계

- **장치/모델 자동 선택**: GPU가 있으면 최고 정확도 모델 `large-v3`를,
  CPU면 실시간에 가까운 모델을 자동 적용합니다 (`--model auto`).
- **문맥 유지 인식**: 직전 인식 결과를 다음 청크의 힌트로 넘겨 고유명사·
  말투의 일관성을 높입니다. `beam_size=5`로 탐색 품질도 올립니다.
- **VAD 필터**: 무음/잡음 구간을 잘라 환각(없는 말 생성)을 줄입니다.
- **번역 폴백**: Google이 일시 차단돼도 MyMemory(무료)로 자동 전환되어
  끊기지 않습니다.

> 정확도의 가장 큰 변수는 **모델 크기**입니다. 정확도를 최우선으로 한다면
> `--model medium` 또는 `--model large-v3` 를 쓰세요(아래 표 참고).

## 동작 환경

- **Windows 10/11** (WASAPI 루프백으로 가상 오디오 장치 없이 캡처)
- Python 3.10 이상
- 번역에 무료 온라인 엔진을 쓰므로 인터넷 연결 필요(번역 단계만)

## 설치

```powershell
pip install -r requirements.txt
```

> faster-whisper는 첫 실행 시 모델을 자동 다운로드합니다(무료).
> CPU만 있어도 동작하며, NVIDIA GPU가 있으면 자동으로 GPU(`float16`)와
> `large-v3` 모델을 사용해 정확도·속도가 크게 향상됩니다.

## 실행

```powershell
python main.py
```

예시 출력:

```
🎧 시스템 출력음을 듣는 중입니다... (Ctrl+C 로 종료)
   모델=large-v3, 장치=cuda(float16), 청크=5.0s, 대상=한국어

[14:03:21] (en 98%)
Never gonna give you up, never gonna let you down
→ 절대 너를 포기하지 않을 거야, 절대 너를 실망시키지 않을 거야
```

### 자주 쓰는 옵션

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `--model` | `auto/tiny/base/small/medium/large-v3` (auto=GPU면 large-v3) | `auto` |
| `--device` | `auto/cpu/cuda` (auto=GPU 자동 감지) | `auto` |
| `--compute-type` | `auto/int8/float16` | `auto` |
| `--beam-size` | 빔 서치 크기 (클수록 정확↑/속도↓) | `5` |
| `--chunk-seconds` | 한 번에 인식할 길이(초) | `5.0` |
| `--silence` | 무음 판정 임계값(RMS) | `0.005` |
| `--no-color` | 색상 출력 끄기 | off |

기본값(`auto`)이 하드웨어에 맞춰 최적 모델을 고릅니다. 빠른 반응이
필요하면 `--model base --chunk-seconds 3`, **정확도를 최우선**으로 하면
`--model large-v3`(권장, GPU 필요) 또는 CPU에서는 `--model medium` 을
쓰세요.

## 구조

| 파일 | 역할 |
| --- | --- |
| `audio_source.py` | 스피커 루프백 캡처 + 모노/리샘플/무음 판정 |
| `transcriber.py` | faster-whisper STT + 언어감지 + 장치/모델 자동선택 |
| `translator.py` | 무료 번역(Google→MyMemory 폴백) + 중복 캐시 |
| `subtitle.py` | 콘솔 자막 포맷 + 중복 줄 제거 |
| `main.py` | 캡처/인식 스레드 오케스트레이션 |
| `test_pipeline.py` | 오디오·모델 없이 가능한 단위 테스트 |

## 테스트

```powershell
pip install pytest
pytest
```

신호 처리(리샘플·무음감지), 번역(캐시·한국어 스킵·엔진 폴백), 자막
(포맷·중복 제거), 장치/모델 자동선택 로직은 마이크나 모델 없이 검증됩니다.

## 한계 / 참고

- **노래**는 배경음악·화음 때문에 일반 음성보다 인식 정확도가 떨어질 수
  있습니다. 가사 자막용으로는 `--model large-v3`(GPU) 또는 `medium`(CPU)을
  권장합니다.
- 5초 고정 창으로 처리하므로 문장이 창 경계에서 잘릴 수 있습니다.
- Google 무료 번역은 비공식 엔드포인트라 과도한 사용 시 일시 제한될 수
  있는데, 이때 자동으로 무료 MyMemory 엔진으로 전환되어 끊기지 않습니다
  (MyMemory는 품질이 다소 낮고 하루 사용량 제한이 있습니다).
- CPU만 있는 경우 `large-v3`는 실시간 처리가 어려울 수 있습니다. 이때는
  정확도와 속도의 균형점인 `medium`/`small`을 쓰세요.
- macOS/Linux에서 쓰려면 `audio_source.py`의 캡처 부분만 해당 OS의
  루프백 장치(BlackHole / PulseAudio monitor)에 맞게 바꾸면 됩니다.
