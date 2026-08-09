from __future__ import annotations

from tools.multilingual_routing import analyze_multilingual_query


def test_latin_query_does_not_require_multilingual_model():
    signal = analyze_multilingual_query("retrieval evidence graph")
    assert signal.scripts == ("latin",)
    assert signal.code_switched is False
    assert signal.multilingual_model_required is False


def test_devanagari_query_requires_multilingual_model():
    signal = analyze_multilingual_query("साक्ष्य पुनर्प्राप्ति")
    assert signal.scripts == ("devanagari",)
    assert signal.multilingual_model_required is True


def test_hinglish_code_switch_detects_latin_and_devanagari_without_claiming_language():
    signal = analyze_multilingual_query("retrieval में evidence चाहिए")
    assert {"latin", "devanagari"}.issubset(set(signal.scripts))
    assert signal.code_switched is True
    assert signal.multilingual_model_required is True
    assert signal.lexical_fallback_recommended is True


def test_symbol_only_query_recommends_lexical_fallback():
    signal = analyze_multilingual_query("12345 !!!")
    assert signal.scripts == ()
    assert signal.lexical_fallback_recommended is True
