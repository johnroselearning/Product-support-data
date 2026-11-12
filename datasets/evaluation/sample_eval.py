"""Create a stratified human-review queue from the held-out cleaned test split.

This creates pending annotations, not a gold evaluation set. No RCA, resolution,
tool expectation, or answerability label is inferred.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_split_records(input_dir, split):
    path = input_dir / "master" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run ingestion and cleaning before sampling."
        )
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def allocate_strata(strata, target):
    keys = sorted(strata)
    allocation = {key: 0 for key in keys}
    target = min(target, sum(len(strata[key]) for key in keys))
    if target >= len(keys):
        for key in keys:
            allocation[key] = 1
    remaining = target - sum(allocation.values())

    while remaining > 0:
        capacities = {
            key: len(strata[key]) - allocation[key]
            for key in keys
            if len(strata[key]) > allocation[key]
        }
        if not capacities:
            break
        total_capacity = sum(capacities.values())
        quotas = {
            key: remaining * capacity / total_capacity
            for key, capacity in capacities.items()
        }
        additions = {
            key: min(capacities[key], int(quotas[key]))
            for key in capacities
        }
        added = sum(additions.values())
        for key, amount in additions.items():
            allocation[key] += amount
        remaining -= added
        if remaining <= 0:
            break

        order = sorted(
            capacities,
            key=lambda key: (quotas[key] - int(quotas[key]), capacities[key], key),
            reverse=True
        )
        for key in order:
            if remaining <= 0:
                break
            if allocation[key] < len(strata[key]):
                allocation[key] += 1
                remaining -= 1

    return allocation


def build_review_case(record, stratum, seed, split):
    return {
        "case_id": record["case_id"],
        "task": record["task"],
        "input": {
            "ticket": record["ticket"]["description"],
            "context": None
        },
        "gold": {
            # Source labels are prefilled for review, not treated as approved gold.
            "labels": dict(record["labels"]),
            "root_cause": None,
            "resolution": None,
            "must_include": [],
            "must_not": [],
            "expected_tools": [],
            "answerable": None
        },
        "tags": [
            "needs_human_review",
            "source:" + record["source"],
            "task:" + record["task"]
        ],
        "provenance": {
            "source": record["source"],
            "source_case_id": record["case_id"],
            "source_revision": str(record.get("source_revision", "unknown")),
            "license": str(record.get("license", "unknown")),
            "source_split": split
        },
        "selection": {
            "method": "stratified_by_task_and_source_label",
            "stratum": stratum,
            "seed": seed
        },
        "review": {
            "status": "pending",
            "reviewer": None,
            "reviewed_at": None,
            "notes": "Verify source label against ticket. Add hard-case tags and evidence-backed gold fields only when supported."
        }
    }


def run(input_dir, output_path, manifest_path, size=150, seed=20261112, split="test"):
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    records = read_split_records(input_dir, split)
    # Exact text duplicates across sources are not independent eval examples.
    by_fingerprint = defaultdict(list)
    for record in records:
        by_fingerprint[record.get("fingerprint", record["case_id"])].append(record)

    eligible = []
    excluded_conflicts = 0
    for fingerprint, candidates in by_fingerprint.items():
        label_signatures = {
            json.dumps(r.get("labels", {}), sort_keys=True) for r in candidates
        }
        if len(label_signatures) > 1:
            excluded_conflicts += len(candidates)
            continue
        # Keep a deterministic representative for cross-source exact text.
        eligible.append(sorted(candidates, key=lambda r: (r["source"], r["case_id"]))[0])

    strata = defaultdict(list)
    for record in eligible:
        label_key = sorted(record.get("labels", {}))[0]
        label_value = record["labels"][label_key]
        stratum = f'{record["task"]}|{label_key}|{label_value}'
        strata[stratum].append(record)

    if not eligible:
        raise ValueError("No eligible held-out test records available.")

    allocation = allocate_strata(strata, size)
    rng = random.Random(seed)
    selected = []
    for stratum in sorted(strata):
        rows = sorted(strata[stratum], key=lambda r: r["case_id"])
        count = allocation[stratum]
        selected.extend(
            (record, stratum)
            for record in rng.sample(rows, count)
        )

    cases = [
        build_review_case(record, stratum, seed, split)
        for record, stratum in sorted(selected, key=lambda item: item[0]["case_id"])
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    by_task = Counter(case["task"] for case in cases)
    by_stratum = Counter(case["selection"]["stratum"] for case in cases)
    manifest = {
        "version": "review_queue_v1",
        "status": "pending_human_review_not_gold_eval",
        "input_split": f"cleaned/master/{split}.jsonl",
        "requested_size": size,
        "actual_size": len(cases),
        "seed": seed,
        "eligible_test_records": len(eligible),
        "excluded_conflicting_exact_text_records": excluded_conflicts,
        "by_task": dict(by_task),
        "by_stratum": dict(by_stratum),
        "output": str(output_path),
        "policy": "Source labels are prefilled but unverified. RCA, resolution, answerability, required facts and expected tools are intentionally blank until human review."
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parents[1] / "cleaned"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "review_queue_v1.jsonl"
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).resolve().parent / "review_queue_manifest.json"
    )
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    print(json.dumps(
        run(args.input, args.output, args.manifest, args.size, args.seed, args.split),
        indent=2
    ))


if __name__ == "__main__":
    main()
