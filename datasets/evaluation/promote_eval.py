"""Promote a fully reviewed queue to an immutable versioned benchmark."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from validation import load_validator, require_valid_case


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def promote(input_path, output_path, manifest_path, expected_split):
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Benchmark versions are immutable. Choose a new versioned output path."
        )

    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError("Cannot promote an empty evaluation queue.")

    seen = set()
    reviewers = set()
    schema_validator = load_validator()
    for row in rows:
      case_id = row.get('case_id', '<missing>')
      require_valid_case(row, case_id, schema_validator)
        case_id = row.get("case_id")
        if not case_id or case_id in seen:
            raise ValueError(f"Missing or duplicate case_id: {case_id}")
        seen.add(case_id)

        review = row.get("review", {})
        if review.get("status") != "approved":
            raise ValueError(f"Case {case_id} is not approved.")
        if not review.get("reviewer") or not review.get("reviewed_at"):
            raise ValueError(f"Case {case_id} lacks reviewer metadata.")
        if row.get("provenance", {}).get("source_split") != expected_split:
            raise ValueError(
                f"Case {case_id} does not belong to {expected_split} split."
            )
        if not row.get("gold", {}).get("labels"):
            raise ValueError(f"Case {case_id} has no reviewed gold label.")
        reviewers.add(review["reviewer"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents accidental overwrite of a released benchmark.
    with output_path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "version": output_path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_review_queue": str(input_path),
        "source_split": expected_split,
        "case_count": len(rows),
        "reviewers": sorted(reviewers),
        "sha256": digest,
        "immutable_release": True
    }
    with manifest_path.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["validation", "test"], required=True
    )
    args = parser.parse_args()
    print(json.dumps(
        promote(args.input, args.output, args.manifest, args.split), indent=2
    ))


if __name__ == "__main__":
    main()
