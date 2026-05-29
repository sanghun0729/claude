# 실시간 시스템 음성 → 한국어 자막 (Windows)

PC에서 **재생되는 소리**(유튜브 영상, 음악, 영어 강의 등)를 실시간으로
캡처해 음성인식하고, 한국어가 아니면 **한국어로 번역해 콘솔에 자막처럼
출력**하는 프로그램입니다.

```
시스템 출력음(WASAPI 루프백) → 30ms 프레임 연속 캡처 → 16kHz 변환
   → VAD 발화 분할(말 시작~침묵까지를 한 문장으로)
   → faster-whisper STT (영어 전용 기본·문맥 유지·beam search)
   → 무료 번역(Google→MyMemory 폴백) → 콘솔 자막
```

5초 고정 분할이 아니라 **VAD(음성 활동 감지)로 문장 단위**를 잡으므로
문장 중간이 잘리지 않아 인식·번역 정확도가 높습니다. 기본은 **영어 전용**
으로, 언어 감지를 생략해 더 빠르고 정확합니다(다국어는 `--language auto`).

## 비용 — 전부 무료

| 구성요소 | 비용 | 비고 |
| --- | --- | --- |
| faster-whisper (음성인식) | 무료 | 오픈소스, 로컬 실행, 모델 자동 다운로드 |
| soundcard (소리 캡처) | 무료 | — |
| deep-translator (번역) | 무료 | **API 키 불필요**. Google 무료 + MyMemory 무료 |
| colorama (색상) | 무료 | — |

어떤 단계에도 유료 API나 키가 필요하지 않습니다.

## 정확도를 높이는 설계

- **VAD 문장 단위 분할**: 고정 길이로 자르지 않고 말이 시작된 지점부터
  일정 시간(기본 700ms) 침묵할 때까지를 한 문장으로 묶어 처리합니다.
  문장이 중간에 끊기지 않아 인식·번역 품질이 크게 좋아집니다.
- **영어 전용 최적화**: 기본 `--language en` 으로 언어 감지를 건너뛰고
  영어 특화 모델 `distil-large-v3`(large급 정확도 + 최고속)를 자동 적용
  합니다. → 영어 기준 **정확도·속도 모두 최상**.
- **장치/모델 자동 선택**: 다국어(`--language auto`)에서는 GPU면
  `large-v3-turbo`, CPU면 실시간에 가까운 모델을 자동 적용합니다.
- **문맥 유지 인식**: 직전 인식 결과를 다음 청크의 힌트로 넘겨 고유명사·
  말투의 일관성을 높입니다. `beam_size=5`로 탐색 품질도 올립니다.
- **VAD 필터**: 무음/잡음 구간을 잘라 환각(없는 말 생성)을 줄입니다.
- **번역 폴백**: Google이 일시 차단돼도 MyMemory(무료)로 자동 전환되어
  끊기지 않습니다.

> **영어 콘텐츠라면 기본값(distil-large-v3) 그대로가 정확도·속도 최적**입니다.
> 다국어를 다룬다면 `--language auto` + `--model large-v3-turbo`(GPU)를 쓰세요.

### 모델별 속도/정확도 (5초 오디오 기준, 대략)

| 모델 | CPU 속도 | GPU 속도 | 정확도 | 비고 |
| --- | --- | --- | --- | --- |
| `small` | ~실시간 | 매우 빠름 | 보통 | CPU 기본값 |
| `medium` | 느림 | 빠름 | 좋음 | CPU 정확도용 |
| `large-v3` | 매우 느림 | ~실시간 | 최고 | CPU 비권장 |
| `large-v3-turbo` | 느림 | 매우 빠름 | large급 | **GPU 기본값(추천)** |
| `distil-large-v3` | 보통 | 매우 빠름 | large급(영어) | **영어 전용**·최고속 |

> `large-v3`는 CPU에서 5초 소리 처리에 수십 초가 걸려 실시간이 불가능합니다.
> GPU가 있다면 `large-v3` 대신 거의 같은 정확도에 훨씬 빠른
> `large-v3-turbo` 를 쓰는 것이 좋습니다.
>
> **영어 콘텐츠만** 다룬다면 `--model distil-large-v3` 가 가장 빠르면서도
> large급 정확도를 냅니다(다국어 미지원이라 영어 외에는 부정확).

## 동작 환경

- **Windows 10/11** (WASAPI 루프백으로 가상 오디오 장치 없이 캡처)
- Python 3.10 이상
- 번역에 무료 온라인 엔진을 쓰므로 인터넷 연결 필요(번역 단계만)

## 설치

```powershell
pip install -r requirements.txt
```

> faster-whisper는 첫 실행 시 모델을 자동 다운로드합니다(무료).
> CPU만 있어도 동작하며, NVIDIA GPU가 있으면 자동으로 GPU(`float16`)를
> 사용해 속도가 크게 향상됩니다.

## 실행

```powershell
python main.py                 # 영어 인식 → 한국어 자막 (기본)
python main.py --language auto  # 다국어 자동 감지
```

예시 출력:

```
🎧 시스템 출력음을 듣는 중입니다... (Ctrl+C 로 종료)
   모델=distil-large-v3, 장치=cuda(float16), 인식언어=en, 분할=VAD(침묵 700ms), 대상=한국어

[14:03:21] (en 100%)
Never gonna give you up, never gonna let you down
→ 절대 너를 포기하지 않을 거야, 절대 너를 실망시키지 않을 거야
```

### 자주 쓰는 옵션

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `--language` | `en`(영어 전용·빠름) / `auto`(자동감지) | `en` |
| `--model` | `auto/.../large-v3-turbo/distil-large-v3` | `auto` |
| `--device` | `auto/cpu/cuda` | `auto` |
| `--compute-type` | `auto/int8/float16` | `auto` |
| `--beam-size` | 빔 서치 크기 (클수록 정확↑/속도↓) | `5` |
| `--vad` | 발화 감지 `energy`(무의존)/`webrtcvad`(더 정확) | `energy` |
| `--silence-ms` | 이만큼 침묵하면 문장 종료(ms) | `700` |
| `--min-speech-ms` | 이보다 짧은 소리는 잡음으로 무시(ms) | `300` |
| `--max-utterance` | 발화 강제 종료 최대 길이(초) | `20` |
| `--frame-ms` | 프레임 길이(ms, webrtcvad는 10/20/30) | `30` |
| `--speech-threshold` | energy VAD 음성 판정 임계값(RMS) | `0.01` |
| `--no-color` | 색상 출력 끄기 | off |

기본값(`auto`)이 언어·하드웨어에 맞춰 최적 모델을 고릅니다(영어면
`distil-large-v3`). 더 정확한 발화 분할이 필요하면 `--vad webrtcvad`
(`pip install webrtcvad`)를, 반응을 더 빨리 보려면 `--silence-ms 400` 처럼
침묵 기준을 줄이세요. 다국어가 필요하면 `--language auto` 를 쓰세요.

## 구조

| 파일 | 역할 |
| --- | --- |
| `audio_source.py` | 스피커 루프백 캡처(프레임/청크) + 모노/리샘플 |
| `segmenter.py` | VAD 발화 분할(말 시작~침묵 단위) + 음성 판정 |
| `transcriber.py` | faster-whisper STT + 영어전용/장치/모델 자동선택 |
| `translator.py` | 무료 번역(Google→MyMemory 폴백) + 중복 캐시 |
| `subtitle.py` | 콘솔 자막 포맷 + 중복 줄 제거 |
| `main.py` | 캡처+VAD 스레드 / STT+번역 스레드 오케스트레이션 |
| `test_pipeline.py` | 오디오·모델 없이 가능한 단위 테스트(22개) |

## 테스트

```powershell
pip install pytest
pytest
```

신호 처리(리샘플), VAD 발화 분할(문장 경계·잡음 폐기·강제 종료·flush),
번역(캐시·한국어 스킵·엔진 폴백), 자막(포맷·중복 제거), 영어전용/장치/모델
자동선택 로직은 마이크나 모델 없이 검증됩니다(테스트 22개).

## 한계 / 참고

- **노래**는 배경음악·화음 때문에 일반 음성보다 인식 정확도가 떨어질 수
  있습니다. energy VAD가 음악을 음성으로 오인할 수 있으니, 가사 자막은
  `--vad webrtcvad` 와 함께 쓰는 것을 권장합니다.
- VAD 특성상 자막은 **말이 잠시 멈춘 뒤**에 나옵니다(문장 단위 처리). 더
  빠른 반응이 필요하면 `--silence-ms` 를 줄이세요(짧으면 문장이 더 잘게
  쪼개집니다 — 정확도와 지연의 트레이드오프).
- Google 무료 번역은 비공식 엔드포인트라 과도한 사용 시 일시 제한될 수
  있는데, 이때 자동으로 무료 MyMemory 엔진으로 전환되어 끊기지 않습니다
  (MyMemory는 품질이 다소 낮고 하루 사용량 제한이 있습니다).
- CPU만 있는 경우 `large-v3`는 실시간 처리가 어려울 수 있습니다. 이때는
  정확도와 속도의 균형점인 `medium`/`small`을 쓰세요.
- macOS/Linux에서 쓰려면 `audio_source.py`의 캡처 부분만 해당 OS의
  루프백 장치(BlackHole / PulseAudio monitor)에 맞게 바꾸면 됩니다.
