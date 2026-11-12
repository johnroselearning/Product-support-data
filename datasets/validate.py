"""Validate canonical normalized output without needing source downloads."""
import argparse
import json
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test")
REQUIRED = {"schema_version", "case_id", "source", "task", "ticket",
            "labels", "evidence", "actions", "root_cause", "resolution",
            "quality", "fingerprint", "source_group"}
def validate(root):
    counts, seen_ids, seen_fingerprints, groups, errors = Counter(), set(), {}, {}, []
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
                    if not isinstance(rec["ticket"].get("description"), str) or not rec["ticket"]["description"].strip():
                        raise ValueError("empty ticket description")
                    if not isinstance(rec["labels"], dict) or not rec["labels"]:
                        raise ValueError("missing labels")
                    if rec["root_cause"] != "unknown" or rec["resolution"] != "unknown":
                        raise ValueError("unverified outcomes in canonical record")
                    case_id, fp = rec["case_id"], rec["fingerprint"]
                    if case_id in seen_ids:
                        raise ValueError("duplicate case ID")
                    seen_ids.add(case_id)
                    if fp in seen_fingerprints:
                        raise ValueError("duplicate text across splits")
                    seen_fingerprints[fp] = split
                    group = (rec["source"], rec["source_group"])
                    if group in groups and groups[group] != split:
                        raise ValueError("source group crosses split boundary")
                    groups[group] = split
                    counts[split] += 1
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    errors.append(f"{path}:{lineno}: {exc}")
    return counts, errors
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "generated")
    args = parser.parse_args()
    counts, errors = validate(args.input)
    print(json.dumps({"counts": dict(counts), "errors": errors[:50], "error_count": len(errors)}, indent=2))
    if errors:
        raise SystemExit(1)
if __name__ == "__main__":
    main()
