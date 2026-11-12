#!/usr/bin/env python3
"""Privacy-conscious profiling for locally downloaded customer-support CSVs.

Prints aggregate metadata only; never prints ticket text, names, email values or row contents.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

TEXT_CANDIDATES = [
    "ticket description", "description", "body", "subject",
    "ticket subject", "answer", "resolution", "ticket resolution"
]
LABEL_CANDIDATES = [
    "ticket type", "type", "queue", "priority", "ticket priority",
    "ticket status", "status", "ticket channel", "channel",
    "language", "customer satisfaction rating"
]
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")

def norm(value):
    return " ".join(str(value).casefold().split())

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def profile(path):
    df = pd.read_csv(path, low_memory=False)
    lower = {str(c).strip().casefold(): c for c in df.columns}
    result = {
        "file": str(path),
        "sha256": digest(path),
        "file_bytes": path.stat().st_size,
        "rows": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {
            str(c): int(df[c].isna().sum()) for c in df.columns
        },
        "unique_by_column": {
            str(c): int(df[c].nunique(dropna=True)) for c in df.columns
        },
        "label_distributions": {},
        "text_profiles": {},
        "text_label_conflicts": {}
    }
    for key, col in lower.items():
        if key in LABEL_CANDIDATES:
            counts = df[col].fillna("<MISSING>").astype(str).value_counts().head(100)
            result["label_distributions"][str(col)] = {
                str(k): int(v) for k, v in counts.items()
            }
    for key, col in lower.items():
        if key not in TEXT_CANDIDATES:
            continue
        values = df[col].fillna("").astype(str)
        lengths = values.str.len()
        normalized = values.map(norm)
        fingerprints = normalized.map(
            lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest() if x else ""
        )
        nonempty = normalized[normalized != ""]
        repeated = fingerprints[fingerprints != ""].value_counts()
        result["text_profiles"][str(col)] = {
            "nonempty": int((normalized != "").sum()),
            "empty": int((normalized == "").sum()),
            "unique_normalized_texts": int(nonempty.nunique()),
            "repeated_text_groups": int((repeated > 1).sum()),
            "rows_in_repeated_text_groups": int(repeated[repeated > 1].sum()),
            "duplicate_extra_rows": int((repeated[repeated > 1] - 1).sum()),
            "median_chars": float(lengths.median()) if len(lengths) else 0,
            "p95_chars": float(lengths.quantile(.95)) if len(lengths) else 0,
            "max_chars": int(lengths.max()) if len(lengths) else 0,
            "placeholder_rows": int(values.str.contains(PLACEHOLDER_RE, na=False).sum()),
            "email_pattern_rows": int(values.str.contains(EMAIL_RE, na=False).sum()),
            "phone_pattern_rows": int(values.str.contains(PHONE_RE, na=False).sum())
        }
        for label_key, label_col in lower.items():
            if label_key not in LABEL_CANDIDATES:
                continue
            pair = pd.DataFrame({"fp": fingerprints, "label": df[label_col].fillna("<MISSING>").astype(str)})
            pair = pair[pair["fp"] != ""]
            distinct = pair.groupby("fp")["label"].nunique()
            conflict_fps = set(distinct[distinct > 1].index)
            result["text_label_conflicts"][f"{col} -> {label_col}"] = {
                "conflicting_text_groups": int(len(conflict_fps)),
                "rows_in_conflicting_groups": int(pair[pair["fp"].isin(conflict_fps)].shape[0])
            }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="CSV files or directories containing CSV files")
    parser.add_argument("--output", type=Path, default=Path("datasets/research/customer_support_csv_profile.json"))
    args = parser.parse_args()
    files = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.csv")))
        elif p.is_file() and p.suffix.casefold() == ".csv":
            files.append(p)
        else:
            raise SystemExit(f"Not a CSV file or directory: {p}")
    files = list(dict.fromkeys(files))
    if not files:
        raise SystemExit("No CSV files found.")
    results = [profile(p) for p in files]
    by_hash = {}
    for item in results:
        by_hash.setdefault(item["sha256"], []).append(item["file"])
    summary = {
        "report_version": "1.0",
        "files_inspected": len(results),
        "identical_file_hash_groups": [paths for paths in by_hash.values() if len(paths) > 1],
        "files": results,
        "privacy_note": "Aggregate counts only; no row text or personal values are emitted. Regex counts are screening signals, not proof of anonymization.",
        "limitations": [
            "This is structural and heuristic profiling, not a human semantic-quality review.",
            "A matching hash proves byte identity; different hashes do not prove different underlying records.",
            "Template and phone-pattern counts require manual interpretation."
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "files_inspected": len(results),
        "report": str(args.output),
        "files": [{
            "file": x["file"], "rows": x["rows"], "columns": x["column_count"],
            "sha256": x["sha256"], "exact_duplicate_rows": x["exact_duplicate_rows"]
        } for x in results],
        "identical_file_hash_groups": summary["identical_file_hash_groups"]
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
