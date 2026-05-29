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

def test_recommend_model_by_device():
    assert tr.recommend_model("cuda") == "large-v3"
    assert tr.recommend_model("cpu") == "small"


def test_auto_resolves_without_loading_model():
    # 모델을 실제로 로드하지 않고 device/compute_type/model_size만 해석되는지 확인.
    t_ = tr.Transcriber(model_size="auto", device="cpu", compute_type="auto")
    assert t_.device == "cpu"
    assert t_.compute_type == "int8"
    assert t_.model_size == "small"
    assert t_._model is None


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
