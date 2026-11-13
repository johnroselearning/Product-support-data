"""Conservative cleaner for the public customer-support CSV candidate.

Does not impute values or modify the source. Removes direct identifiers, normalizes
blank strings, parses date-like fields for consistent representation, and writes
a separate missingness/quality report for human review.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

DIRECT_ID_PATTERNS = (
    r"^ticket\s*id$", r"^customer\s*name$", r"^customer\s*email$",
    r"^email$", r"^phone$", r"^phone\s*number$", r"^account\s*id$",
)
DATE_HINT = re.compile(r"(date|timestamp|time)", re.I)
EXPECTED_FIELDS = {
    "Resolution": "Do not impute; verify resolution quality and task eligibility.",
    "Time to Resolution": "Meaning is unverified; sample values look like timestamps. Do not derive duration without a confirmed start/end definition.",
    "Customer Satisfaction Rating": "Do not impute when used as a target; assess whether missingness depends on ticket status.",
}


def match_direct_identifier(column: str) -> bool:
    name = str(column).strip()
    return any(re.search(pattern, name, flags=re.I) for pattern in DIRECT_ID_PATTERNS)


def clean(input_path: Path, output_path: Path, report_path: Path) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    df = pd.read_csv(input_path, low_memory=False)
    if df.empty:
        raise ValueError("Input CSV has no rows.")

    # Convert whitespace-only strings to missing values; preserve actual labels/text.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda value: pd.NA if isinstance(value, str) and not value.strip() else value)

    removed = [str(c) for c in df.columns if match_direct_identifier(str(c))]
    profile = []
    for col in df.columns:
        series = df[col]
        item = {
            "column": str(col),
            "dtype_before_parse": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percent": round(float(series.isna().mean() * 100), 3),
            "distinct_nonmissing": int(series.nunique(dropna=True)),
        }
        if str(col) in EXPECTED_FIELDS:
            item["handling"] = EXPECTED_FIELDS[str(col)]
        if DATE_HINT.search(str(col)) and series.notna().any():
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            valid = int(parsed.notna().sum())
            item["date_parse_valid_count"] = valid
            item["date_parse_failure_count"] = int(series.notna().sum() - valid)
            if valid:
                item["date_min"] = parsed.min().isoformat()
                item["date_max"] = parsed.max().isoformat()
                # Store parsed datetimes as ISO-like text to avoid CSV timezone/type surprises.
                df[col] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S").where(parsed.notna(), pd.NA)
        profile.append(item)

    status_col = next((c for c in df.columns if str(c).strip().casefold() == "ticket status"), None)
    missing_by_status = {}
    if status_col is not None:
        for field in EXPECTED_FIELDS:
            if field in df.columns:
                table = pd.crosstab(df[status_col].fillna("<missing>"), df[field].isna(), normalize="index")
                missing_by_status[field] = {
                    str(status): {("missing" if bool(flag) else "present"): round(float(value) * 100, 2)
                                  for flag, value in row.items()}
                    for status, row in table.iterrows()
                }

    duplicate_count = int(df.duplicated().sum())
    cleaned = df.drop(columns=removed, errors="ignore")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(df)),
        "columns_before": int(df.shape[1]),
        "columns_after": int(cleaned.shape[1]),
        "removed_direct_identifier_columns": removed,
        "duplicate_rows_before_deduplication": duplicate_count,
        "duplicates_removed": 0,
        "imputation_applied": False,
        "missingness_by_column": profile,
        "missingness_by_ticket_status_percent": missing_by_status,
        "policy": [
            "Raw input is never overwritten.",
            "No blanket row deletion: records are retained for task-specific review.",
            "No imputation: missing outcome labels are not invented.",
            "Rows with missing labels should be excluded only from training/evaluation for that specific target.",
            "Review the CSV and reports for sensitive values embedded in free-text fields before sharing.",
            "Use date/time columns only after confirming what each timestamp represents.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a separate, conservative product-support CSV copy.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/cleaned/customer_support_tickets.csv"))
    parser.add_argument("--report", type=Path, default=Path("datasets/cleaned/customer_support_tickets_quality_report.json"))
    args = parser.parse_args()
    print(json.dumps(clean(args.input, args.output, args.report), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
