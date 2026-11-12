"""Detect near-duplicate ticket descriptions for human review; never auto-delete."""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize(text):
    return " ".join(str(text).casefold().split())


def shingles(text, width=5):
    text = normalize(text)
    if len(text) < width:
        return {text} if text else set()
    return {text[i:i + width] for i in range(len(text) - width + 1)}


def load_records(root):
    rows = []
    for split in ("train", "validation", "test"):
        path = root / "master" / (split + ".jsonl")
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("ticket", {}).get("description", "")
                if len(normalize(text)) < 30:
                    continue
                rows.append({
                    "case_id": record["case_id"],
                    "source": record["source"],
                    "split": split,
                    "fingerprint": record["fingerprint"],
                    "labels": record.get("labels", {}),
                    "text": normalize(text)
                })
    return rows


def find_near_duplicates(rows, threshold=0.82, max_candidates_per_row=20000):
    sets = [shingles(row["text"]) for row in rows]
    postings = defaultdict(list)
    max_document_frequency = max(20, math.ceil(len(rows) * 0.20))

    for index, grams in enumerate(sets):
        for gram in grams:
            postings[gram].append(index)

    pairs = set()
    candidate_truncated = 0
    for index, grams in enumerate(sets):
        candidates = set()
        for gram in grams:
            docs = postings[gram]
            if len(docs) > max_document_frequency:
                continue
            candidates.update(other for other in docs if other > index)
            if len(candidates) > max_candidates_per_row:
                candidate_truncated += 1
                candidates = set(sorted(candidates)[:max_candidates_per_row])
                break

        for other in candidates:
            if rows[index]["fingerprint"] == rows[other]["fingerprint"]:
                continue
            intersection = len(grams & sets[other])
            union = len(grams | sets[other])
            score = intersection / union if union else 0.0
            if score >= threshold:
                pairs.add((index, other, score))

    results = []
    for left, right, score in sorted(pairs, key=lambda x: (-x[2], x[0], x[1])):
        a, b = rows[left], rows[right]
        results.append({
            "case_id_a": a["case_id"],
            "case_id_b": b["case_id"],
            "source_a": a["source"],
            "source_b": b["source"],
            "split_a": a["split"],
            "split_b": b["split"],
            "label_a": a["labels"],
            "label_b": b["labels"],
            "jaccard_char_5gram": round(score, 4),
            "cross_split": a["split"] != b["split"],
            "cross_source": a["source"] != b["source"],
            "decision": "human_review_required_no_automatic_removal"
        })

    return results, candidate_truncated


def run(input_dir, output_path, summary_path, threshold=0.82):
    rows = load_records(input_dir)
    results, truncated = find_near_duplicates(rows, threshold=threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_split_pair = Counter(
        row["split_a"] + " -> " + row["split_b"] for row in results
    )
    summary = {
        "report_version": "1.0",
        "records_scanned": len(rows),
        "near_duplicate_pairs": len(results),
        "cross_split_pairs": sum(row["cross_split"] for row in results),
        "cross_source_pairs": sum(row["cross_source"] for row in results),
        "pairs_by_split": dict(by_split_pair),
        "threshold": threshold,
        "similarity": "Jaccard similarity over character 5-grams",
        "candidate_generation_truncated_rows": truncated,
        "policy": "Review-only report. No records are removed or relabeled automatically.",
        "limitations": [
            "Similarity is a heuristic, not proof that two tickets describe the same incident.",
            "Template collisions and different incidents with similar wording require human review.",
            "Short descriptions under 30 normalized characters are excluded."
        ],
        "pairs_file": str(output_path)
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parents[1] / "cleaned"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "quality_report" / "near_duplicate_pairs.jsonl"
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path(__file__).resolve().parents[1] / "quality_report" / "near_duplicate_summary.json"
    )
    parser.add_argument("--threshold", type=float, default=0.82)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.summary, args.threshold), indent=2))


if __name__ == "__main__":
    main()
