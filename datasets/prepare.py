"""Conservative support-data preparation; Python 3.10+."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = "You are a technical support engineer. Do not invent evidence or verified root causes."
PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){10,15}(?!\d)"),
    "secret": re.compile(r"(?i)\b(?:api[_-]?key|password|secret|bearer)\s*[:=]\s*\S+"),
}
def digest(s):
    return hashlib.sha256(" ".join(s.casefold().split()).encode()).hexdigest()
def pii(s):
    return [name for name, pattern in PATTERNS.items() if pattern.search(s)]
def split_for(key):
    bucket = int(key[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"
def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
def normalize(source, row, features, upstream, index):
    if source == "it_support":
        text = row.get("text")
        label = row.get("label")
        if isinstance(label, int) and hasattr(features["label"], "int2str"):
            label = features["label"].int2str(label)
        task = "classification"
        answer = json.dumps({"category": str(label)})
    else:
        text = row.get("instruction")
        task = "response"
        answer = row.get("response")
    if not isinstance(text, str) or not text.strip() or not isinstance(answer, str) or not answer.strip():
        return None
    key = digest(text)
    return {
        "case_id": source + "-" + key[:16],
        "source": source, "upstream_split": upstream, "upstream_row": index,
        "task": task, "ticket": {"description": text},
        "label_or_answer": answer,
        "evidence": [], "actions": [], "root_cause": "unknown",
        "resolution": "unknown", "quality": "unreviewed",
        "fingerprint": key,
    }
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--include-bitext", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "generated")
    args = p.parse_args()
    from datasets import load_dataset
    sources = json.loads((ROOT / "sources.json").read_text())["sources"]
    master = {s: [] for s in ("train", "validation", "test")}
    sft = {s: [] for s in master}
    seen, counts = set(), Counter()
    for src in sources:
        if src["status"] != "enabled" and not (args.include_bitext and src["id"] == "bitext"):
            continue
        dataset = load_dataset(src["repo"])
        for upstream, rows in dataset.items():
            for i, row in enumerate(rows):
                rec = normalize(src["id"], row, rows.features, upstream, i)
                if rec is None:
                    counts["invalid"] += 1
                    continue
                key = rec["fingerprint"]
                if key in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(key)
                flags = pii(rec["ticket"]["description"] + "\n" + rec["label_or_answer"])
                if flags:
                    counts["quarantined"] += 1
                    continue
                # Preserve upstream test; do not train on it.
                split = "test" if upstream == "test" else split_for(key)
                if src["id"] == "it_support" and upstream != "test" and split == "test":
                    split = "validation"
                master[split].append(rec)
                prompt = ("Classify this support ticket. Return JSON with category only:\n"
                          if rec["task"] == "classification" else "") + rec["ticket"]["description"]
                sft[split].append({
                    "case_id": rec["case_id"], "source": src["id"],
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": rec["label_or_answer"]}
                    ]
                })
                counts[src["id"] + ":" + split] += 1
    for split in master:
        write(args.output / "master" / (split + ".jsonl"), master[split])
        write(args.output / "sft" / (split + ".jsonl"), sft[split])
    (args.output / "manifest.json").write_text(json.dumps({
        "counts": dict(counts), "sources": [s for s in sources if s["status"] == "enabled" or (args.include_bitext and s["id"] == "bitext")],
        "warning": "PII checks are heuristic. Human privacy and licensing review required."
    }, indent=2) + "\n")
    print(dict(counts))
if __name__ == "__main__":
    main()
