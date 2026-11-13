"""Dataset-agnostic exploratory analysis for product-support tables.

Reads CSV, JSONL/NDJSON, Parquet, or a directory of JSONL files. Writes
aggregate profiles, correlations, review hints, and charts to a separate folder.
It never changes input records or selects a training target.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ID_HINT = re.compile(r"(^|_)(id|uuid|email|phone|name|account|customer|user|token|secret)(_|$)", re.I)
TARGET_HINT = re.compile(r"(label|target|category|class|priority|severity|status|resolution|outcome|queue|team)", re.I)
TEXT_HINT = re.compile(r"(text|description|body|subject|message|comment|summary|title|resolution|answer)", re.I)
MAX_CORR_COLUMNS = 80
MAX_PROFILE_VALUES = 20


def load_frame(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise ValueError(f"No JSONL files found in {path}")
        frames = [load_frame(file).assign(_source_file=file.name) for file in files]
        return pd.concat(frames, ignore_index=True, sort=False)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Supported inputs: CSV, JSONL/NDJSON, JSON, Parquet, or a JSONL directory")


def clean_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def column_profile(df: pd.DataFrame) -> list[dict]:
    profiles = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        unique = int(s.nunique(dropna=True))
        item = {
            "column": str(col), "dtype": str(s.dtype), "rows": int(len(s)),
            "missing_count": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 3) if len(s) else 0,
            "distinct_count": unique,
            "distinct_pct_of_non_null": round(unique / len(non_null) * 100, 3) if len(non_null) else 0,
            "examples": [clean_scalar(v) for v in non_null.drop_duplicates().head(5)],
            "role_review_hints": [],
        }
        if ID_HINT.search(str(col)):
            item["role_review_hints"].append("Possible identifier or sensitive field; exclude or review before modeling.")
        if TARGET_HINT.search(str(col)):
            item["role_review_hints"].append("Possible outcome/label; confirm its meaning and prediction-time availability.")
        if unique > max(100, int(len(df) * .5)) and len(df) > 20:
            item["role_review_hints"].append("High cardinality; inspect for IDs, free text, or leakage.")
        if pd.api.types.is_numeric_dtype(s):
            nums = pd.to_numeric(s, errors="coerce").dropna()
            if len(nums):
                q1, q3 = nums.quantile([.25, .75])
                iqr = q3 - q1
                item["numeric"] = {
                    "count": int(nums.count()), "mean": float(nums.mean()), "median": float(nums.median()),
                    "std": clean_scalar(nums.std()), "min": float(nums.min()), "p25": float(q1),
                    "p75": float(q3), "p95": float(nums.quantile(.95)), "max": float(nums.max()),
                    "outlier_fence_count_1_5_iqr": int(((nums < q1 - 1.5 * iqr) | (nums > q3 + 1.5 * iqr)).sum()),
                }
        elif pd.api.types.is_datetime64_any_dtype(s):
            item["datetime"] = {"min": clean_scalar(s.min()), "max": clean_scalar(s.max())}
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            text = s.dropna().astype(str)
            if TEXT_HINT.search(str(col)) or (len(text) and text.str.len().median() > 40):
                lengths = text.str.len()
                item["text"] = {
                    "empty_or_whitespace_count": int(text.str.strip().eq("").sum()),
                    "characters_median": float(lengths.median()) if len(lengths) else None,
                    "characters_p95": float(lengths.quantile(.95)) if len(lengths) else None,
                    "characters_max": int(lengths.max()) if len(lengths) else None,
                }
            item["top_values"] = [
                {"value": clean_scalar(k), "count": int(v)}
                for k, v in s.value_counts(dropna=False).head(MAX_PROFILE_VALUES).items()
            ]
        profiles.append(item)
    return profiles


def correlations(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > MAX_CORR_COLUMNS:
        numeric = numeric.loc[:, numeric.nunique(dropna=True).sort_values(ascending=False).head(MAX_CORR_COLUMNS).index]
    return {
        method: numeric.corr(method=method, min_periods=2).replace({np.nan: None}).to_dict()
        for method in ("pearson", "spearman", "kendall")
    }


def run(input_path: Path, output: Path) -> dict:
    df = load_frame(input_path)
    if df.empty:
        raise ValueError(f"No rows found in {input_path}")
    output.mkdir(parents=True, exist_ok=True)
    profiles = column_profile(df)
    corr = correlations(df)
    hints = []
    if df.duplicated().any():
        hints.append(f"{int(df.duplicated().sum())} fully duplicated rows found; inspect before deciding whether to remove.")
    if len(df.select_dtypes(include="number").columns) < 2:
        hints.append("Fewer than two numeric columns: numeric correlation analysis is limited. Encode categories only for a defined analysis.")
    hints.append("Correlation is association, not causation; it does not identify a model target.")
    hints.append("Possible target columns are hints from names only. Choose a target only after defining the product-support task and label quality.")
    summary = {
        "input": str(input_path), "rows": int(len(df)), "columns": int(len(df.columns)),
        "duplicate_full_rows": int(df.duplicated().sum()), "profiles": profiles,
        "correlations_numeric": corr, "review_hints": hints,
    }
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame([{k: v for k, v in item.items() if k not in {"top_values", "examples"}} for item in profiles]).to_json(
        output / "column_profile.json", orient="records", indent=2, force_ascii=False)
    for method, matrix in corr.items():
        pd.DataFrame(matrix).to_csv(output / f"correlation_{method}.csv")
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        ax = numeric.hist(figsize=(12, max(4, 2.5 * ((len(numeric.columns) + 3) // 4))), bins=30)
        plt.gcf().tight_layout()
        plt.gcf().savefig(output / "numeric_distributions.png", dpi=130)
        plt.close("all")
        for method, matrix in corr.items():
            if matrix and len(matrix) > 1:
                values = pd.DataFrame(matrix).astype(float)
                fig, ax = plt.subplots(figsize=(max(6, len(values) * .6), max(5, len(values) * .5)))
                image = ax.imshow(values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
                ax.set_xticks(range(len(values.columns)), values.columns, rotation=70, ha="right", fontsize=8)
                ax.set_yticks(range(len(values.index)), values.index, fontsize=8)
                ax.set_title(f"{method.title()} correlation (numeric columns)")
                fig.colorbar(image, ax=ax)
                fig.tight_layout()
                fig.savefig(output / f"correlation_{method}.png", dpi=130)
                plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a product-support dataset without changing it.")
    parser.add_argument("--input", required=True, type=Path, help="CSV/JSONL/JSON/Parquet file or directory of JSONL files.")
    parser.add_argument("--output", type=Path, default=Path("datasets/eda/custom"), help="Separate report directory.")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
