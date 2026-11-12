"""Fail-closed path guard for future fine-tuning entrypoints."""

from pathlib import Path


def assert_safe_training_input(path, role="train"):
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    parts = {part.casefold() for part in candidate.parts}
    name = candidate.name.casefold()

    if parts.intersection({"test", "evaluation", "eval", "review_queue"}):
        raise ValueError("Test/evaluation data must never be used for training.")

    if "eval_v1" in name or "review_queue" in name or "test" in name:
        raise ValueError("Held-out evaluation data must never be used for training.")

    if "sft" not in parts:
        raise ValueError("Training inputs must come from a generated SFT directory.")

    if role == "train" and name != "train.jsonl":
        raise ValueError("Fine-tuning may consume only sft/train.jsonl.")
    if role == "validation" and name != "validation.jsonl":
        raise ValueError("Development validation may consume only sft/validation.jsonl.")
    if role not in {"train", "validation"}:
        raise ValueError("role must be 'train' or 'validation'.")

    return candidate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--role", choices=["train", "validation"], default="train")
    args = parser.parse_args()
    print(assert_safe_training_input(args.path, args.role))
