"""오디오/모델 없이 검증 가능한 순수 로직 테스트.

audio_source(신호 처리), translator(번역 캐시/스킵), subtitle(포맷/중복)을
실제 마이크나 무거운 모델 없이 검증한다.
"""

import numpy as np
import pytest

import audio_source as a
import translator as t
import transcriber as tr
from subtitle import SubtitlePrinter, format_subtitle


# --- audio_source ---------------------------------------------------------

def test_to_mono_from_stereo():
    stereo = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    mono = a.to_mono(stereo)
    assert mono.shape == (2,)
    assert np.allclose(mono, [0.5, 1.0])


def test_to_mono_passthrough_1d():
    mono = np.array([0.1, 0.2], dtype=np.float32)
    assert np.array_equal(a.to_mono(mono), mono)


def test_resample_changes_length():
    audio = np.ones(48_000, dtype=np.float32)
    out = a.resample(audio, 48_000, 16_000)
    assert out.shape[0] == 16_000


def test_resample_noop_same_rate():
    audio = np.arange(10, dtype=np.float32)
    assert np.array_equal(a.resample(audio, 16_000, 16_000), audio)


def test_rms_and_silence():
    silent = np.zeros(1000, dtype=np.float32)
    loud = np.ones(1000, dtype=np.float32)
    assert a.rms(silent) == 0.0
    assert a.is_silent(silent)
    assert not a.is_silent(loud)


# --- translator -----------------------------------------------------------

def test_korean_is_not_translated():
    calls = []

    def fake(text, src):
        calls.append(text)
        return "X"

    out = t.to_korean("안녕하세요", source_lang="ko", translate_fn=fake)
    assert out == "안녕하세요"
    assert calls == []  # 번역 함수가 호출되지 않아야 함


def test_translation_uses_cache():
    t.clear_cache()
    calls = []

    def fake(text, src):
        calls.append(text)
        return "번역됨"

    assert t.to_korean("hello", "en", fake) == "번역됨"
    assert t.to_korean("hello", "en", fake) == "번역됨"
    assert calls == ["hello"]  # 두 번째는 캐시 사용


def test_empty_text_returns_empty():
    assert t.to_korean("   ", "en") == ""


def test_falls_back_when_first_engine_fails(monkeypatch):
    # _default_translate 가 첫 엔진 실패 시 다음 무료 엔진으로 넘어가는지 검증.
    calls = []

    def boom(text, source):
        calls.append("google")
        raise RuntimeError("blocked")

    def ok_mymemory(text, source_lang):
        calls.append("mymemory")
        return "폴백번역"

    monkeypatch.setattr(t, "_google", boom)
    monkeypatch.setattr(t, "_mymemory", ok_mymemory)
    assert t._default_translate("hello", "en") == "폴백번역"
    # google 두 번 시도(source, auto) 후 mymemory 성공
    assert calls == ["google", "google", "mymemory"]


def test_returns_original_when_all_engines_fail(monkeypatch):
    monkeypatch.setattr(t, "_google", lambda *a: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(t, "_mymemory", lambda *a: (_ for _ in ()).throw(RuntimeError()))
    assert t._default_translate("hello", "en") == "hello"


# --- transcriber (auto 설정 로직) -----------------------------------------

def test_recommend_model_english_uses_distil():
    # 영어 전용이면 장치와 무관하게 distil-large-v3(영어 특화 고속·고정밀).
    assert tr.recommend_model("cuda", "en") == "distil-large-v3"
    assert tr.recommend_model("cpu", "en") == "distil-large-v3"


def test_recommend_model_multilingual_by_device():
    assert tr.recommend_model("cuda", None) == "large-v3-turbo"
    assert tr.recommend_model("cpu", None) == "small"


def test_auto_resolves_english_without_loading_model():
    # 기본(영어)에서 device/compute_type/model_size가 로드 없이 해석되는지 확인.
    t_ = tr.Transcriber(model_size="auto", device="cpu", compute_type="auto")
    assert t_.device == "cpu"
    assert t_.compute_type == "int8"
    assert t_.language == "en"
    assert t_.model_size == "distil-large-v3"
    assert t_._model is None


def test_language_auto_is_normalized_to_none():
    t_ = tr.Transcriber(model_size="auto", device="cpu", language="auto")
    assert t_.language is None
    assert t_.model_size == "small"  # 다국어 CPU 기본


# --- subtitle -------------------------------------------------------------

def test_format_includes_translation_arrow():
    s = format_subtitle("hello", "안녕", "en", 0.99, timestamp="12:00:00",
                        colored=False)
    assert "hello" in s
    assert "→ 안녕" in s
    assert "(en 99%)" in s


def test_format_korean_has_no_arrow():
    s = format_subtitle("안녕", "안녕", "ko", timestamp="12:00:00", colored=False)
    assert "→" not in s


def test_printer_skips_duplicate(capsys):
    p = SubtitlePrinter(colored=False)
    assert p.show("hello", "안녕", "en") is True
    assert p.show("hello", "안녕", "en") is False  # 중복 스킵
    assert p.show("world", "세계", "en") is True
    out = capsys.readouterr().out
    assert out.count("→") == 2


def test_printer_skips_empty():
    p = SubtitlePrinter(colored=False)
    assert p.show("   ", "x", "en") is False


# --- segmenter (VAD 발화 분할) --------------------------------------------

from segmenter import VadSegmenter, energy_is_speech

SR = 16_000
FRAME = 0.1  # 100ms 프레임으로 계산하기 쉽게


def _frame(speech: bool):
    n = int(SR * FRAME)
    return np.ones(n, dtype=np.float32) if speech else np.zeros(n, dtype=np.float32)


def _seg():
    return VadSegmenter(
        sample_rate=SR, frame_seconds=FRAME, silence_ms=300,
        min_speech_ms=200, speech_pad_ms=200,
        is_speech_fn=energy_is_speech(threshold=0.5),
    )


def test_segmenter_emits_one_utterance_after_silence():
    seg = _seg()
    # 침묵1, 말3, 침묵3 → 침묵 300ms에서 발화 종료
    pattern = [False, True, True, True, False, False, False]
    outputs = [seg.push(_frame(s)) for s in pattern]
    emitted = [o for o in outputs if o is not None]
    assert len(emitted) == 1
    # 프리롤(직전 침묵1) + 말3 + 말꼬리 침묵3 = 7프레임
    assert emitted[0].shape[0] == 7 * int(SR * FRAME)
    # 마지막(7번째) 프레임에서만 방출
    assert outputs[-1] is not None
    assert all(o is None for o in outputs[:-1])


def test_segmenter_discards_short_blip():
    seg = _seg()
    # 말 1프레임(100ms < 최소 200ms) 후 침묵 → 잡음으로 폐기
    pattern = [True, False, False, False]
    emitted = [o for o in (seg.push(_frame(s)) for s in pattern) if o is not None]
    assert emitted == []


def test_segmenter_force_emits_on_max_length():
    seg = VadSegmenter(
        sample_rate=SR, frame_seconds=FRAME, silence_ms=10_000,
        min_speech_ms=100, max_utterance_s=0.5,
        is_speech_fn=energy_is_speech(threshold=0.5),
    )
    # 침묵 없이 계속 말하면 max_utterance(0.5s=5프레임)에서 강제 종료
    outputs = [seg.push(_frame(True)) for _ in range(6)]
    assert any(o is not None for o in outputs)


def test_segmenter_flush_returns_pending_speech():
    seg = _seg()
    seg.push(_frame(True))
    seg.push(_frame(True))  # 200ms 누적, 아직 침묵 없음
    out = seg.flush()
    assert out is not None
    assert out.shape[0] == 2 * int(SR * FRAME)
