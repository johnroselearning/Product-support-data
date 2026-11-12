"""Conservative, auditable cleaning. Redacts PII and never edits raw/master inputs."""
import argparse
import sys
import json
import hashlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exporters.jsonl import write_jsonl
from quality.privacy import redact_text, ensure_person_ner_available, privacy_capabilities
from normalization.sft import to_sft

SPLITS = ("train", "validation", "test")


def clean_text(value):
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def clean_record(record, allow_regex_only=False):
    out = dict(record)
    ticket = dict(record.get("ticket", {}))
    text = ticket.get("description")
    if not isinstance(text, str):
        return None, ["invalid_description_type"], {}

    text = clean_text(text)
    if not text:
        return None, ["empty_after_trim"], {}

    safe_text, flags, redaction_counts = redact_text(
        text, require_ner=not allow_regex_only,
        use_ner=not allow_regex_only
    )
    safe_text = clean_text(safe_text)
    if not safe_text:
        return None, ["empty_after_redaction"], redaction_counts

    ticket["description"] = safe_text
    out["ticket"] = ticket

    labels = record.get("labels")
    if not isinstance(labels, dict) or not labels:
        return None, ["missing_labels"], redaction_counts

    cleaned_labels = {}
    for key, value in labels.items():
        if value is None or not str(value).strip():
            return None, ["empty_label"], redaction_counts
        label = str(value).strip()
        _, label_flags, _ = redact_text(
            label, require_ner=not allow_regex_only,
            use_ner=not allow_regex_only
        )
        if label_flags:
            return None, ["privacy_in_label"], redaction_counts
        cleaned_labels[key] = label
    out["labels"] = cleaned_labels

    fingerprint = hashlib.sha256(
        " ".join(safe_text.casefold().split()).encode("utf-8")
    ).hexdigest()
    out["fingerprint"] = fingerprint
    out["split_group"] = fingerprint
    if out.get("source_group_type") == "text_fingerprint":
        out["source_group"] = fingerprint
    out["case_id"] = (
        f'{out["source"]}-{out.get("upstream_split", "unknown")}-'
        f'{out.get("upstream_row", 0)}-{fingerprint[:12]}'
    )

    quality = dict(out.get("quality", {}))
    prior_flags = set(quality.get("privacy_flags", []))
    quality["privacy_flags"] = sorted(prior_flags | set(flags))
    prior_counts = Counter(quality.get("privacy_redactions", {}))
    prior_counts.update(redaction_counts)
    quality["privacy_redactions"] = dict(prior_counts)
    quality["privacy_mode"] = (
        "regex_only_not_production_ready"
        if allow_regex_only else "regex_plus_spacy_ner"
    )
    out["quality"] = quality
    return out, [], redaction_counts


def run(input_dir, output_dir, allow_regex_only=False):
    if not allow_regex_only:
        ensure_person_ner_available()

    output_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    accepted = {s: [] for s in SPLITS}
    quarantine = []
    report = {
        "input_records": 0,
        "accepted_records": 0,
        "excluded_records": 0,
        "exclusion_reasons": {},
        "by_split": {},
        "pii_redaction_counts": {},
        "privacy_capabilities": privacy_capabilities(),
        "privacy_mode": (
            "regex_only_not_production_ready"
            if allow_regex_only else "regex_plus_spacy_ner"
        ),
        "actions": [
            "Normalize line endings and remove NUL bytes",
            "Redact detected PII before writing cleaned canonical records",
            "Hash incident identifiers and exact-text split groups",
            "Trim description and label edges",
            "Preserve label values and spelling",
            "Do not remove records solely for unusual length",
            "Do not synthesize or infer labels, root causes, evidence or resolutions",
            "Collapse exact repeats only within the same source incident identity"
        ]
    }
    reasons = Counter()
    redaction_totals = Counter()

    # Read test first so same-identity exact repeats retain the publisher test copy.
    for split in ("test", "validation", "train"):
        path = input_dir / "master" / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                report["input_records"] += 1
                record = json.loads(line)
                cleaned, errors, redaction_counts = clean_record(
                    record, allow_regex_only
                )
                redaction_totals.update(redaction_counts)

                if errors:
                    for reason in errors:
                        reasons[reason] += 1
                    quarantine.append({
                        "case_id": record.get("case_id"),
                        "source": record.get("source"),
                        "original_split": split,
                        "line": line_no,
                        "reasons": errors
                    })
                    continue

                key = (
                    cleaned["source"],
                    cleaned["fingerprint"],
                    cleaned.get("source_group", "unknown")
                )
                if key in seen:
                    reasons["duplicate_within_same_incident_identity"] += 1
                    quarantine.append({
                        "case_id": cleaned["case_id"],
                        "source": cleaned["source"],
                        "original_split": split,
                        "line": line_no,
                        "reasons": ["duplicate_within_same_incident_identity"],
                        "retained_case_id": seen[key]
                    })
                    continue

                seen[key] = cleaned["case_id"]
                accepted[split].append(cleaned)

    for split, records in accepted.items():
        write_jsonl(output_dir / "master" / f"{split}.jsonl", records)
        if split in ("train", "validation"):
            sft_records = [to_sft(record) for record in records]
            write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_records)
        report["by_split"][split] = len(records)

    stale_test_sft = output_dir / "sft" / "test.jsonl"
    if stale_test_sft.exists():
        stale_test_sft.unlink()

    report["accepted_records"] = sum(map(len, accepted.values()))
    report["excluded_records"] = len(quarantine)
    report["exclusion_reasons"] = dict(reasons)
    report["pii_redaction_counts"] = dict(redaction_totals)
    write_jsonl(output_dir / "quarantine.jsonl", quarantine)
    (output_dir / "cleaning_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parents[1] / "generated"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "cleaned"
    )
    parser.add_argument(
        "--allow-regex-only", action="store_true",
        help="Explicitly bypass person/location NER; output is not production-ready"
    )
    args = parser.parse_args()
    print(json.dumps(
        run(args.input, args.output, args.allow_regex_only), indent=2
    ))


if __name__ == "__main__":
    main()
