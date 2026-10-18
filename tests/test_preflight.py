from __future__ import annotations

from types import SimpleNamespace

import pytest

from distillkit import preflight


def _config(model_type: str, vocab_size: int):
    text = SimpleNamespace(model_type=f"{model_type}_text", vocab_size=vocab_size)
    return SimpleNamespace(model_type=model_type, text_config=text)


def _tokenizer(path: str, vocab_size: int):
    return {
        "path": path,
        "vocab_size": vocab_size,
        "vocabulary_sha256": "same",
    }


def test_verify_pair_accepts_matching_target_family(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        preflight,
        "_model_identity",
        lambda path: preflight.ModelIdentity(
            path=path,
            model_type="qwen3_5_text",
            architecture_family="qwen3_5",
            vocab_size=248_320,
        ),
    )
    monkeypatch.setattr(
        preflight,
        "_tokenizer_identity",
        lambda path: _tokenizer(path, 248_077),
    )

    report = preflight.verify_pair(
        student_path="student",
        teacher_path="teacher",
        target_family="qwen3_5",
        expected_model_vocab_size=248_320,
    )

    assert report["student_model"]["architecture_family"] == "qwen3_5"
    assert report["teacher_model"]["architecture_family"] == "qwen3_5"


def test_verify_pair_rejects_wrong_family(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        preflight,
        "_model_identity",
        lambda path: preflight.ModelIdentity(
            path=path,
            model_type="qwen3_text",
            architecture_family="qwen3",
            vocab_size=151_936,
        ),
    )
    monkeypatch.setattr(
        preflight,
        "_tokenizer_identity",
        lambda path: _tokenizer(path, 151_936),
    )

    with pytest.raises(ValueError, match="not the requested qwen3_5 family"):
        preflight.verify_pair(
            student_path="student",
            teacher_path="teacher",
            target_family="qwen3_5",
            expected_model_vocab_size=248_320,
        )


def test_verify_pair_rejects_tokenizer_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        preflight,
        "_model_identity",
        lambda path: preflight.ModelIdentity(
            path=path,
            model_type="qwen3_5_text",
            architecture_family="qwen3_5",
            vocab_size=248_320,
        ),
    )

    def identity(path: str):
        result = _tokenizer(path, 248_077)
        result["vocabulary_sha256"] = path
        return result

    monkeypatch.setattr(preflight, "_tokenizer_identity", identity)

    with pytest.raises(ValueError, match="different vocabularies"):
        preflight.verify_pair(
            student_path="student",
            teacher_path="teacher",
            target_family="qwen3_5",
            expected_model_vocab_size=248_320,
        )
