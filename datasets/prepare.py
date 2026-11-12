"""Orchestrate download -> adapt -> privacy screen -> group split -> export."""
import argparse
import json
from collections import Counter
from pathlib import Path
from datasets.downloaders.huggingface import download_source
from datasets.adapters.registry import get_adapter
from datasets.quality.privacy import detect_flags
from datasets.splitting.group_split import assign_records
from datasets.normalization.sft import to_sft
from datasets.exporters.jsonl import write_jsonl

ROOT = Path(__file__).resolve().parent

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
                    quarantined.append({"case_id": record["case_id"], "source": source["id"], "flags": flags})
                    stats["quarantined"] += 1
                    continue
                all_records.append(record)
    # Deduplicate per source. Prefer publisher test rows when duplicate text straddles splits.
    chosen, seen = [], {}
    for record in sorted(all_records, key=lambda r: (r["upstream_split"] != "test", r["upstream_row"])):
        key = (record["source"], record["fingerprint"])
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen[key] = True
        chosen.append(record)
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
    manifest = {
        "schema_version": "1.2",
        "counts": dict(stats),
        "accepted": len(chosen),
        "quarantined": len(quarantined),
        "sources": source_runs,
        "policy": "Source test groups are held out as test; duplicate records prefer publisher test copy.",
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
