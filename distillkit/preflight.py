"""Architecture and tokenizer checks for a distillation run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click


@dataclass(frozen=True)
class ModelIdentity:
    path: str
    model_type: str
    architecture_family: str
    vocab_size: int


def _model_identity(path: str) -> ModelIdentity:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    text_config = getattr(config, "text_config", None)
    source_config = text_config if text_config is not None else config
    model_type = str(source_config.model_type)
    architecture_family = model_type.removesuffix("_text")
    vocab_size = int(getattr(source_config, "vocab_size", 0) or 0)
    return ModelIdentity(
        path=path,
        model_type=model_type,
        architecture_family=architecture_family,
        vocab_size=vocab_size,
    )


def _tokenizer_identity(path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    vocabulary = tokenizer.get_vocab()
    return {
        "path": path,
        "vocab_size": max(len(vocabulary), max(vocabulary.values()) + 1),
        "vocabulary_sha256": _stable_vocabulary_hash(vocabulary),
    }


def _stable_vocabulary_hash(vocabulary: dict[str, int]) -> str:
    import hashlib
    import json

    normalized = sorted(vocabulary.items(), key=lambda item: (item[1], item[0]))
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_pair(
    *,
    student_path: str,
    teacher_path: str | None,
    target_family: str,
    expected_model_vocab_size: int,
) -> dict[str, Any]:
    """Verify model families, vocab sizes, and tokenizers before training."""
    student_model = _model_identity(student_path)
    student_tokenizer = _tokenizer_identity(student_path)
    teacher_model = _model_identity(teacher_path) if teacher_path else None
    teacher_tokenizer = _tokenizer_identity(teacher_path) if teacher_path else None

    model_identities = [student_model]
    if teacher_model is not None:
        model_identities.append(teacher_model)
    for identity in model_identities:
        if identity.architecture_family != target_family:
            raise ValueError(
                f"{identity.path} is {identity.architecture_family}, "
                f"not the requested {target_family} family"
            )
        if identity.vocab_size != expected_model_vocab_size:
            raise ValueError(
                f"{identity.path} has vocab size {identity.vocab_size}, "
                f"expected {expected_model_vocab_size}"
            )

    tokenizer_identities = [student_tokenizer]
    if teacher_tokenizer is not None:
        tokenizer_identities.append(teacher_tokenizer)
    for identity in tokenizer_identities:
        if not 0 < identity["vocab_size"] <= expected_model_vocab_size:
            raise ValueError(
                f"{identity['path']} tokenizer vocab size "
                f"{identity['vocab_size']} exceeds model size "
                f"{expected_model_vocab_size}"
            )
    if teacher_tokenizer is not None and (
        student_tokenizer["vocabulary_sha256"] != teacher_tokenizer["vocabulary_sha256"]
    ):
        raise ValueError("student and teacher tokenizers have different vocabularies")

    return {
        "target_family": target_family,
        "expected_model_vocab_size": expected_model_vocab_size,
        "student_model": asdict(student_model),
        "student_tokenizer": student_tokenizer,
        "teacher_model": asdict(teacher_model) if teacher_model else None,
        "teacher_tokenizer": teacher_tokenizer,
    }


@click.command("distillkit-preflight")
@click.option("--student", required=True)
@click.option("--teacher")
@click.option("--target-family", required=True)
@click.option("--expected-model-vocab-size", required=True, type=int)
@click.option("--output", type=click.Path(path_type=Path))
def main(
    student: str,
    teacher: str | None,
    target_family: str,
    expected_model_vocab_size: int,
    output: Path | None,
) -> None:
    """Fail unless the requested teacher/student pair passes the gate."""
    import json

    report = verify_pair(
        student_path=student,
        teacher_path=teacher,
        target_family=target_family,
        expected_model_vocab_size=expected_model_vocab_size,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    click.echo(payload)


if __name__ == "__main__":
    main()
