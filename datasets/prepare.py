"""Orchestrate download, PII redaction, identity-aware deduplication, splitting and export."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from downloaders.huggingface import download_source
from adapters.registry import get_adapter
from adapters.common import digest
from quality.privacy import (
    redact_text, ensure_person_ner_available, privacy_capabilities
)
from splitting.group_split import assign_records
from normalization.sft import to_sft
from exporters.jsonl import write_jsonl

ROOT = Path(__file__).resolve().parent


def deduplicate_records(records):
    """Collapse repeats within a known identity; preserve distinct incidents.

    Conflicting labels for identical model inputs are quarantined because the
    current SFT prompt has no additional feature to distinguish those labels.
    """
    groups = defaultdict(list)
    for record in records:
        groups[(record["source"], record["fingerprint"])].append(record)

    chosen, duplicate_audit, conflict_quarantine = [], [], []
    stats = Counter()

    for (source, fingerprint), candidates in groups.items():
        labels = {json.dumps(r["labels"], sort_keys=True) for r in candidates}
        ordered = sorted(
            candidates,
            key=lambda r: (r["upstream_split"] != "test", r["upstream_row"])
        )

        if len(labels) > 1:
            stats["duplicate_label_conflict_groups"] += 1
            stats["conflict_records_quarantined"] += len(ordered)
            duplicate_audit.append({
                "source": source,
                "fingerprint": fingerprint,
                "label_conflict": True,
                "decision": "quarantine_all_candidates_pending_review",
                "candidate_count": len(ordered),
                "candidates": [{
                    "case_id": r["case_id"],
                    "upstream_split": r["upstream_split"],
                    "upstream_row": r["upstream_row"],
                    "source_group_type": r.get("source_group_type", "unknown"),
                    "labels": r["labels"]
                } for r in ordered]
            })
            for record in ordered:
                conflict_quarantine.append({
                    "case_id": record["case_id"],
                    "source": source,
                    "original_split": record["upstream_split"],
                    "reasons": ["duplicate_text_with_conflicting_labels"],
                    "fingerprint": fingerprint
                })
            continue

        by_identity = defaultdict(list)
        for record in ordered:
            by_identity[record["source_group"]].append(record)

        retained_for_text = []
        for identity, identity_rows in by_identity.items():
            identity_rows.sort(
                key=lambda r: (r["upstream_split"] != "test", r["upstream_row"])
            )
            retained = identity_rows[0]
            retained_for_text.append(retained)
            chosen.append(retained)

            for excluded in identity_rows[1:]:
                stats["duplicate_records_collapsed"] += 1
                duplicate_audit.append({
                    "source": source,
                    "fingerprint": fingerprint,
                    "excluded_case_id": excluded["case_id"],
                    "retained_case_id": retained["case_id"],
                    "excluded_upstream_split": excluded["upstream_split"],
                    "retained_upstream_split": retained["upstream_split"],
                    "decision": "collapse_repeat_within_same_incident_identity",
                    "identity_type": retained.get("source_group_type", "unknown")
                })

        if len(retained_for_text) > 1:
            stats["same_text_distinct_incident_groups_preserved"] += len(retained_for_text)
            duplicate_audit.append({
                "source": source,
                "fingerprint": fingerprint,
                "decision": "preserve_same_text_distinct_incident_identities",
                "identity_count": len(retained_for_text),
                "incident_identity_verified": all(
                    r.get("source_group_type") == "incident_id"
                    for r in retained_for_text
                ),
                "case_ids": [r["case_id"] for r in retained_for_text]
            })

    return chosen, duplicate_audit, conflict_quarantine, stats


def run(output, cache_dir, force_download=False, include_bitext=False,
        allow_regex_only=False):
    if not allow_regex_only:
        ensure_person_ner_available()

    registry = json.loads((ROOT / "sources.json").read_text())
    raw_dir = ROOT / "raw"
    all_records, quarantined, stats, source_runs = [], [], Counter(), []

    for source in registry["sources"]:
        enabled = source["status"] == "enabled" or (
            include_bitext and source["id"] == "bitext"
        )
        if not enabled:
            continue

        adapter = get_adapter(source["id"])
        dataset, download_info = download_source(
            source, raw_dir, cache_dir, force_download
        )
        source_runs.append({"source": source["id"], **download_info})

        for upstream_split, rows in dataset.items():
            stats[f'{source["id"]}:downloaded:{upstream_split}'] += len(rows)
            for index, row in enumerate(rows):
                record = adapter(
                    source, row, rows.features, upstream_split, index
                )
                if record is None:
                    stats["invalid"] += 1
                    continue

                original_fingerprint = record["fingerprint"]
                description, flags, redaction_counts = redact_text(
                    record["ticket"]["description"],
                    require_ner=not allow_regex_only
                )
                label_text = " ".join(record["labels"].values())
                _, label_flags, _ = redact_text(
                    label_text, require_ner=not allow_regex_only
                )

                if label_flags:
                    quarantined.append({
                        "case_id": record["case_id"],
                        "source": source["id"],
                        "original_split": upstream_split,
                        "reasons": ["pii_detected_in_label"],
                        "flags": label_flags
                    })
                    stats["quarantined_label_pii"] += 1
                    continue

                record["ticket"]["description"] = description.strip()
                if not record["ticket"]["description"]:
                    stats["invalid_after_redaction"] += 1
                    continue

                # Recompute identifiers from redacted text; never export a hash
                # derived from the unredacted description.
                record["fingerprint"] = digest(record["ticket"]["description"])
                record["case_id"] = (
                    f'{source["id"]}-{upstream_split}-{index}-'
                    f'{record["fingerprint"][:12]}'
                )
                record["split_group"] = record["fingerprint"]
                if record.get("source_group_type") == "text_fingerprint":
                    record["source_group"] = record["fingerprint"]

                record["quality"]["privacy_flags"] = flags
                record["quality"]["privacy_redactions"] = redaction_counts
                if flags:
                    stats["records_with_pii_redacted"] += 1
                    for flag, count in redaction_counts.items():
                        stats[f"pii_redactions:{flag}"] += count

                if allow_regex_only:
                    record["quality"]["privacy_mode"] = "regex_only_not_production_ready"
                else:
                    record["quality"]["privacy_mode"] = "regex_plus_spacy_ner"

                all_records.append(record)

    chosen, duplicate_audit, conflict_quarantine, dedup_stats = (
        deduplicate_records(all_records)
    )
    stats.update(dedup_stats)
    quarantined.extend(conflict_quarantine)

    assignments = assign_records(chosen)
    master = {name: [] for name in ("train", "validation", "test")}
    sft = {name: [] for name in master}

    for record in chosen:
        split_key = (record["source"], record.get("split_group") or record["source_group"])
        split = assignments[split_key]
        master[split].append(record)
        sft[split].append(to_sft(record))
        stats[f'{record["source"]}:normalized:{split}'] += 1

    for split in master:
        write_jsonl(output / "master" / f"{split}.jsonl", master[split])
        write_jsonl(output / "sft" / f"{split}.jsonl", sft[split])

    write_jsonl(output / "quarantine.jsonl", quarantined)
    write_jsonl(output / "duplicate_audit.jsonl", duplicate_audit)

    manifest = {
        "schema_version": "1.4",
        "counts": dict(stats),
        "accepted": len(chosen),
        "quarantined": len(quarantined),
        "duplicate_audit_records": len(duplicate_audit),
        "duplicate_label_conflicts": stats["duplicate_label_conflict_groups"],
        "conflict_records_quarantined": stats["conflict_records_quarantined"],
        "privacy_capabilities": privacy_capabilities(),
        "privacy_mode": (
            "regex_only_not_production_ready"
            if allow_regex_only else "regex_plus_spacy_ner"
        ),
        "sources": source_runs,
        "policy": (
            "PII is redacted before canonical export. Exact-text groups are kept "
            "within one split. Repeats are collapsed only within the same source "
            "identity; distinct incident IDs are preserved. Identical model inputs "
            "with conflicting labels are quarantined."
        ),
        "warning": (
            "Automated PII detection is not complete anonymization. Raw snapshots "
            "remain unmodified and must be access-controlled. Human privacy and "
            "license review remain required."
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "generated")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--include-bitext", action="store_true",
        help="Only after license review"
    )
    parser.add_argument(
        "--allow-regex-only", action="store_true",
        help="Explicitly bypass person/location NER; output is not production-ready"
    )
    args = parser.parse_args()
    result = run(
        args.output, args.cache_dir, args.force_download,
        args.include_bitext, args.allow_regex_only
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
