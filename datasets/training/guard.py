"""Fail-closed path guard for future fine-tuning entrypoints."""

from pathlib import Path


DATASETS_ROOT = Path(__file__).resolve().parents[1]


def assert_safe_training_input(path, role="train"):
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(DATASETS_ROOT)
    except ValueError as exc:
        raise ValueError("Training inputs must be inside this repository's datasets directory.") from exc

    parts = tuple(part.casefold() for part in relative.parts)
    name = candidate.name.casefold()

    if parts.intersection({"test", "evaluation", "eval", "review_queue"}):
        raise ValueError("Test/evaluation data must never be used for training.")

    if "eval_v1" in name or "review_queue" in name or name == "test.jsonl":
        raise ValueError("Held-out evaluation data must never be used for training.")

    if len(parts) != 3 or parts[0] not in {"generated", "cleaned"} or parts[1] != "sft":
        raise ValueError(
            "Training inputs must be datasets/generated/sft or datasets/cleaned/sft."
        )

    if role == "train" and name != "train.jsonl":
        raise ValueError("Fine-tuning may consume only sft/train.jsonl.")
    if role == "validation" and name != "validation.jsonl":
        raise ValueError("Development validation may consume only sft/validation.jsonl.")
    if role not in {"train", "validation"}:
        raise ValueError("role must be 'train' or 'validation'.")

    return resolved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--role", choices=["train", "validation"], default="train")
    args = parser.parse_args()
    print(assert_safe_training_input(args.path, args.role))
