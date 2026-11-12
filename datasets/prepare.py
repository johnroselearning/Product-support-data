"""Orchestrate download, adaptation, privacy screening, deduplication, splitting and export."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from downloaders.huggingface import download_source
from adapters.registry import get_adapter
from quality.privacy import detect_flags
from splitting.group_split import assign_records
from normalization.sft import to_sft
from exporters.jsonl import write_jsonl

ROOT = Path(__file__).resolve().parent

def deduplicate_records(records):
    """Deduplicate same-label text; quarantine every candidate when labels conflict."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["source"], record["fingerprint"])].append(record)
    chosen, duplicate_audit, conflict_quarantine = [], [], []
    stats = Counter()
    for (source, fingerprint), candidates in groups.items():
        labels = {json.dumps(r["labels"], sort_keys=True) for r in candidates}
        ordered = sorted(candidates, key=lambda r: (r["upstream_split"] != "test", r["upstream_row"]))
        if len(labels) > 1:
            stats["duplicate_label_conflict_groups"] += 1
            stats["conflict_records_quarantined"] += len(ordered)
            duplicate_audit.append({
                "source": source, "fingerprint": fingerprint,
                "label_conflict": True, "decision": "quarantine_all_candidates_pending_review",
                "candidate_count": len(ordered),
                "candidates": [{
                    "case_id": r["case_id"], "upstream_split": r["upstream_split"],
                    "upstream_row": r["upstream_row"], "labels": r["labels"]
                } for r in ordered]
            })
            for r in ordered:
                conflict_quarantine.append({
                    "case_id": r["case_id"], "source": source,
                    "original_split": r["upstream_split"],
                    "reasons": ["duplicate_text_with_conflicting_labels"],
                    "fingerprint": fingerprint
                })
            continue
        retained = ordered[0]
        chosen.append(retained)
        for excluded in ordered[1:]:
            stats["duplicates"] += 1
            duplicate_audit.append({
                "source": source, "fingerprint": fingerprint,
                "excluded_case_id": excluded["case_id"],
                "retained_case_id": retained["case_id"],
                "excluded_upstream_split": excluded["upstream_split"],
                "retained_upstream_split": retained["upstream_split"],
                "excluded_labels": excluded["labels"],
                "retained_labels": retained["labels"],
                "label_conflict": False,
                "decision": "deduplicate_same_text_same_label"
            })
    return chosen, duplicate_audit, conflict_quarantine, stats

def run(output, cache_dir, force_download=False, include_bitext=False):
    registry = json.loads((ROOT / "sources.json").read_text())
    raw_dir = ROOT / "raw"
    all_records, quarantined, stats, source_runs = [], [], Counter(), []
    for source in registry["sources"]:
        enabled = source["status"] == "enabled" or (include_bitext and source["id"] == "bitext")
        if not enabled:
            continue
        adapter = get_adapter(source["id"])
        dataset, download_info = download_source(source, raw_dir, cache_dir, force_download)
        source_runs.append({"source": source["id"], **download_info})
        for upstream_split, rows in dataset.items():
            stats[f'{source["id"]}:downloaded:{upstream_split}'] += len(rows)
            for index, row in enumerate(rows):
                record = adapter(source, row, rows.features, upstream_split, index)
                if record is None:
                    stats["invalid"] += 1
                    continue
                flags = detect_flags(record["ticket"]["description"] + "\n" + " ".join(record["labels"].values()))
                if flags:
                    record["quality"]["privacy_flags"] = flags
                    quarantined.append({"case_id": record["case_id"], "source": source["id"],
                                        "original_split": upstream_split, "flags": flags})
                    stats["quarantined"] += 1
                    continue
                all_records.append(record)

    chosen, duplicate_audit, conflict_quarantine, dedup_stats = deduplicate_records(all_records)
    stats.update(dedup_stats)
    quarantined.extend(conflict_quarantine)
    assignments = assign_records(chosen)
    master = {name: [] for name in ("train", "validation", "test")}
    sft = {name: [] for name in master}
    for record in chosen:
        split = assignments[(record["source"], record["source_group"])]
        master[split].append(record)
        sft[split].append(to_sft(record))
        stats[f'{record["source"]}:normalized:{split}'] += 1
    for split in master:
        write_jsonl(output / "master" / f"{split}.jsonl", master[split])
        write_jsonl(output / "sft" / f"{split}.jsonl", sft[split])
    write_jsonl(output / "quarantine.jsonl", quarantined)
    write_jsonl(output / "duplicate_audit.jsonl", duplicate_audit)
    manifest = {
        "schema_version": "1.3",
        "counts": dict(stats),
        "accepted": len(chosen),
        "quarantined": len(quarantined),
        "duplicate_audit_records": len(duplicate_audit),
        "duplicate_label_conflicts": stats["duplicate_label_conflict_groups"],
        "conflict_records_quarantined": stats["conflict_records_quarantined"],
        "sources": source_runs,
        "policy": "Same-text same-label duplicates prefer publisher test rows; all records in same-text conflicting-label groups are quarantined pending review.",
        "warning": "Heuristic PII detection is not complete anonymization. Human privacy and license review remain required."
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "generated")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--include-bitext", action="store_true", help="Only after license review")
    args = parser.parse_args()
    result = run(args.output, args.cache_dir, args.force_download, args.include_bitext)
    print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()
