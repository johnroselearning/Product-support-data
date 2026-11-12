"""Validate canonical normalized output and reject unredacted PII."""
import argparse
import json
from collections import Counter
from pathlib import Path

from quality.privacy import detect_flags, ensure_person_ner_available

SPLITS = ("train", "validation", "test")
REQUIRED = {
    "schema_version", "case_id", "source", "task", "ticket", "labels",
    "evidence", "actions", "root_cause", "resolution", "quality",
    "fingerprint", "source_group", "source_group_type", "split_group"
}


def validate(root, allow_regex_only=False):
    if not allow_regex_only:
        ensure_person_ner_available()

    counts = Counter()
    seen_ids = set()
    fingerprint_splits = {}
    group_splits = {}
    errors = []

    for split in SPLITS:
        path = root / "master" / (split + ".jsonl")
        if not path.is_file():
            errors.append(f"Missing {path}")
            continue

        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                try:
                    rec = json.loads(line)
                    missing = REQUIRED - rec.keys()
                    if missing:
                        raise ValueError("missing: " + ",".join(sorted(missing)))

                    description = rec["ticket"].get("description")
                    if not isinstance(description, str) or not description.strip():
                        raise ValueError("empty ticket description")
                    if not isinstance(rec["labels"], dict) or not rec["labels"]:
                        raise ValueError("missing labels")
                    if rec["root_cause"] != "unknown" or rec["resolution"] != "unknown":
                        raise ValueError("unverified outcomes in canonical record")

                    pii_flags = detect_flags(description, use_ner=not allow_regex_only)
                    if pii_flags:
                        raise ValueError("unredacted PII categories: " + ",".join(pii_flags))

                    case_id = rec["case_id"]
                    if case_id in seen_ids:
                        raise ValueError("duplicate case ID")
                    seen_ids.add(case_id)

                    fingerprint_key = rec["fingerprint"]
                    previous_split = fingerprint_splits.get(fingerprint_key)
                    if previous_split is not None and previous_split != split:
                        raise ValueError("identical text crosses split boundary")
                    fingerprint_splits[fingerprint_key] = split

                    group = rec["split_group"]
                    if group in group_splits and group_splits[group] != split:
                        raise ValueError("exact-text split group crosses split boundary")
                    group_splits[group] = split

                    counts[split] += 1
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    errors.append(f"{path}:{lineno}: {exc}")

    return counts, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent / "generated"
    )
    parser.add_argument(
        "--allow-regex-only", action="store_true",
        help="Explicitly bypass person/location NER; validation is not production-complete"
    )
    args = parser.parse_args()
    counts, errors = validate(args.input, args.allow_regex_only)
    print(json.dumps({
        "counts": dict(counts),
        "errors": errors[:50],
        "error_count": len(errors),
        "privacy_mode": (
            "regex_only_not_production_ready"
            if args.allow_regex_only else "regex_plus_spacy_ner"
        )
    }, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
