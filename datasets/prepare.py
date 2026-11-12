"""Normalize permitted public support datasets into provenance-rich JSONL."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = "You are a technical support engineer. Never invent evidence or verified root causes."
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
        if set(row) != {"text", "label"}:
            raise ValueError("Unexpected IT support source schema")
        text = row["text"]
        label = row["label"]
        if isinstance(label, int) and hasattr(features["label"], "int2str"):
            label = features["label"].int2str(label)
        task, labels = "classification", {"category": str(label)}
        group = digest(text)
    elif source == "helpdesk":
        expected = {"issue_id", "text", "issue_priority"}
        if not expected.issubset(row):
            raise ValueError("Unexpected helpdesk source schema")
        text = row["text"]
        task, labels = "priority_label", {"priority_source": str(row["issue_priority"])}
        group = upstream + ":" + str(row["issue_id"])
    elif source == "bitext":
        expected = {"instruction", "response"}
        if not expected.issubset(row):
            raise ValueError("Unexpected Bitext source schema")
        text = row["instruction"]
        task, labels = "response", {"reference_response": row["response"]}
        group = digest(text)
    else:
        raise ValueError("Unsupported adapter: " + source)
    if not isinstance(text, str) or not text.strip() or any(not isinstance(v, str) or not v.strip() for v in labels.values()):
        return None
    key = digest(text)
    return {
        "schema_version": "1.1", "case_id": source + "-" + key[:16],
        "source": source, "upstream_split": upstream, "upstream_row": index,
        "source_group": group, "task": task, "ticket": {"description": text},
        "labels": labels, "evidence": [], "actions": [], "root_cause": "unknown",
        "resolution": "unknown", "quality": {"status": "unreviewed"},
        "fingerprint": key,
    }
def to_sft(rec):
    text, labels = rec["ticket"]["description"], rec["labels"]
    if rec["task"] == "classification":
        user = "Classify this support ticket. Return JSON with category only:\n" + text
        answer = json.dumps({"category": labels["category"]}, ensure_ascii=False)
    elif rec["task"] == "priority_label":
        user = "Return the original ticket priority label as JSON with priority_source only:\n" + text
        answer = json.dumps({"priority_source": labels["priority_source"]}, ensure_ascii=False)
    else:
        user, answer = text, labels["reference_response"]
    return {"case_id": rec["case_id"], "source": rec["source"],
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]}
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--include-bitext", action="store_true", help="Requires independent license review")
    p.add_argument("--output", type=Path, default=ROOT / "generated")
    args = p.parse_args()
    from datasets import load_dataset
    sources = json.loads((ROOT / "sources.json").read_text())["sources"]
    master = {s: [] for s in ("train", "validation", "test")}
    sft = {s: [] for s in master}
    quarantine, seen, counts = [], set(), Counter()
    for src in sources:
        if src["status"] != "enabled" and not (args.include_bitext and src["id"] == "bitext"):
            continue
        kwargs = {"path": src["repo"]}
        if src.get("config"):
            kwargs["name"] = src["config"]
        dataset = load_dataset(**kwargs)
        for upstream, rows in dataset.items():
            for i, row in enumerate(rows):
                rec = normalize(src["id"], row, rows.features, upstream, i)
                if rec is None:
                    counts["invalid"] += 1
                    continue
                key = (src["id"], rec["fingerprint"])
                if key in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(key)
                flags = pii(rec["ticket"]["description"] + "\n" + " ".join(rec["labels"].values()))
                if flags:
                    quarantine.append({"case_id": rec["case_id"], "source": src["id"], "flags": flags})
                    counts["quarantined"] += 1
                    continue
                # Preserve the publisher's test split. Validation is carved from train.
                if upstream == "test":
                    split = "test"
                elif src["id"] == "it_support":
                    split = "validation" if split_for(key) != "train" else "train"
                else:
                    split = split_for(digest(src["id"] + ":" + rec["source_group"]))
                master[split].append(rec)
                sft[split].append(to_sft(rec))
                counts[src["id"] + ":" + split] += 1
    for split in master:
        write(args.output / "master" / (split + ".jsonl"), master[split])
        write(args.output / "sft" / (split + ".jsonl"), sft[split])
    write(args.output / "quarantine.jsonl", quarantine)
    manifest = {"schema_version": "1.1", "counts": dict(counts),
                "accepted": sum(map(len, master.values())),
                "quarantined": len(quarantine),
                "sources": [s for s in sources if s["status"] == "enabled" or (args.include_bitext and s["id"] == "bitext")],
                "warning": "Heuristic PII checks and exact deduplication do not replace privacy, license and semantic leakage reviews."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))
if __name__ == "__main__":
    main()
