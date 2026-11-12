"""Descriptive exploratory analysis of normalized support records."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, gaussian_kde

SPLITS = ("train", "validation", "test")
NUMERIC_COLUMNS = ("text_chars", "text_words")


def load_records(root):
    rows = []
    for split in SPLITS:
        path = root / "master" / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                description = record["ticket"]["description"]
                rows.append({
                    "split": split,
                    "source": record["source"],
                    "task": record["task"],
                    "description": description,
                    "text_chars": len(description),
                    "text_words": len(description.split()),
                    "label_key": next(iter(record["labels"]), ""),
                    "label_value": str(next(iter(record["labels"].values()), "")),
                    "case_id": record["case_id"],
                    "fingerprint": record["fingerprint"],
                    "source_group": record["source_group"],
                    "privacy_flags": record.get("quality", {}).get("privacy_flags", []),
                })
    return pd.DataFrame(rows)


def _json_number(value):
    if pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _mode_values(series):
    modes = series.mode(dropna=True)
    counts = series.value_counts(dropna=True)
    return [
        {"value": int(value), "count": int(counts.loc[value])}
        for value in modes.tolist()
    ]


def _numeric_summary(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    count = int(values.count())
    return {
        "count": count,
        "mean": _json_number(values.mean()),
        "median": _json_number(values.median()),
        "mode": _mode_values(values),
        "variance_sample": _json_number(values.var(ddof=1)) if count > 1 else None,
        "standard_deviation_sample": _json_number(values.std(ddof=1)) if count > 1 else None,
        "min": _json_number(values.min()),
        "q1": _json_number(q1),
        "q3": _json_number(q3),
        "iqr": _json_number(q3 - q1),
        "p95": _json_number(values.quantile(0.95)),
        "max": _json_number(values.max()),
        "skewness": _json_number(values.skew()) if count >= 3 else None,
        "excess_kurtosis_fisher": _json_number(values.kurtosis()) if count >= 4 else None,
        "iqr_outlier_count": int(((values < q1 - 1.5 * (q3 - q1)) |
                                  (values > q3 + 1.5 * (q3 - q1))).sum()),
    }


def _correlations(df):
    result = {}
    for method in ("pearson", "spearman", "kendall"):
        matrix = df[list(NUMERIC_COLUMNS)].corr(method=method)
        result[method] = {
            str(row): {
                str(column): _json_number(matrix.loc[row, column])
                for column in matrix.columns
            }
            for row in matrix.index
        }
    return result


def _source_split_association(df):
    table = pd.crosstab(df["source"], df["split"])
    table_json = {
        str(source): {str(split): int(count) for split, count in row.items()}
        for source, row in table.iterrows()
    }
    result = {
        "variables": ["source", "split"],
        "contingency_table": table_json,
        "interpretation_note": (
            "Exploratory association only. Split assignment is group-based, so "
            "the independence assumption may not hold for every record."
        ),
    }
    if table.shape[0] < 2 or table.shape[1] < 2:
        result["status"] = "not_run_requires_at_least_two_categories_per_variable"
        return result

    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.to_numpy().sum()
    min_expected = float(expected.min())
    result["minimum_expected_count"] = min_expected
    if min_expected < 5:
        result["status"] = "not_run_expected_counts_below_5"
        return result

    statistic, p_value, degrees_of_freedom, _ = chi2_contingency(
        table.to_numpy(), correction=False
    )
    result.update({
        "status": "computed_exploratory",
        "chi_square": float(statistic),
        "degrees_of_freedom": int(degrees_of_freedom),
        "p_value": float(p_value),
    })
    return result


def _save_count_charts(df, output_dir):
    charts = [
        ("source_counts.png", df["source"].value_counts(), "Records by source"),
        ("split_counts.png", df["split"].value_counts(), "Records by split"),
        (
            "label_counts.png",
            df.groupby(["source", "label_value"]).size(),
            "Label distribution by source",
        ),
    ]
    for filename, series, title in charts:
        fig, ax = plt.subplots(figsize=(11, 6))
        series.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Records")
        ax.tick_params(axis="x", rotation=55)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=140)
        plt.close(fig)


def _save_distribution_charts(df, output_dir):
    for column, label, filename in (
        ("text_chars", "Characters", "text_length_distribution.png"),
        ("text_words", "Words", "text_word_distribution.png"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        df[column].plot(kind="hist", bins=40, ax=ax)
        ax.set_title(f"Ticket description length ({label.lower()})")
        ax.set_xlabel(label)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=140)
        plt.close(fig)

    sources = sorted(df["source"].dropna().astype(str).unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, column, label in zip(
        axes, NUMERIC_COLUMNS, ("Characters", "Words")
    ):
        groups = [
            df.loc[df["source"].astype(str) == source, column].dropna().to_numpy()
            for source in sources
        ]
        ax.boxplot(groups, labels=sources, showfliers=True)
        ax.set_title(f"Ticket length by source ({label.lower()})")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "text_length_boxplots_by_source.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, column, label in zip(
        axes, NUMERIC_COLUMNS, ("Characters", "Words")
    ):
        values = df[column].dropna().to_numpy(dtype=float)
        if len(values) >= 2 and np.ptp(values) > 0:
            estimator = gaussian_kde(values)
            grid = np.linspace(values.min(), values.max(), 300)
            density = estimator(grid)
            ax.plot(grid, density)
            ax.fill_between(grid, density, alpha=0.2)
        else:
            ax.text(0.5, 0.5, "Insufficient variation for KDE",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Ticket length density ({label.lower()})")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(output_dir / "text_length_kde.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, column, label in zip(
        axes, NUMERIC_COLUMNS, ("Characters", "Words")
    ):
        grouped = [
            df.loc[df["source"].astype(str) == source, column]
            .dropna()
            .to_numpy(dtype=float)
            for source in sources
        ]
        if grouped and all(len(values) > 0 for values in grouped):
            ax.violinplot(grouped, showmedians=True, showextrema=True)
            ax.set_xticks(range(1, len(sources) + 1), sources, rotation=35)
        ax.set_title(f"Ticket length by source ({label.lower()})")
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(output_dir / "text_length_violins_by_source.png", dpi=140)
    plt.close(fig)


def _save_relationship_charts(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["text_words"], df["text_chars"], alpha=0.3, s=16)
    ax.set_title("Ticket text length: words vs characters")
    ax.set_xlabel("Words")
    ax.set_ylabel("Characters")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "text_words_vs_chars_scatter.png", dpi=140)
    plt.close(fig)

    corr = df[list(NUMERIC_COLUMNS)].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(NUMERIC_COLUMNS)), NUMERIC_COLUMNS, rotation=30)
    ax.set_yticks(range(len(NUMERIC_COLUMNS)), NUMERIC_COLUMNS)
    for row in range(len(NUMERIC_COLUMNS)):
        for column in range(len(NUMERIC_COLUMNS)):
            value = corr.iloc[row, column]
            label = "NA" if pd.isna(value) else f"{value:.2f}"
            ax.text(column, row, label, ha="center", va="center")
    ax.set_title("Pearson correlation of numeric features")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "numeric_correlation_heatmap.png", dpi=140)
    plt.close(fig)

    axes = pd.plotting.scatter_matrix(
        df[list(NUMERIC_COLUMNS)],
        figsize=(8, 8),
        diagonal="hist",
        alpha=0.3,
        s=12,
    )
    axes[0, 0].figure.suptitle("Pair plot of numeric features", y=1.02)
    axes[0, 0].figure.tight_layout()
    axes[0, 0].figure.savefig(output_dir / "numeric_pairplot.png", dpi=140)
    plt.close(axes[0, 0].figure)


def run(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_records(input_dir)
    if df.empty:
        raise ValueError(f"No master records found under {input_dir}")

    summary = {
        "records": int(len(df)),
        "by_source": {str(k): int(v) for k, v in df["source"].value_counts().items()},
        "by_split": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        "by_task": {str(k): int(v) for k, v in df["task"].value_counts().items()},
        "label_distribution": {
            str(k): int(v)
            for k, v in df.groupby(["source", "label_value"], dropna=False).size().items()
        },
        "text_length_chars": {
            "min": int(df["text_chars"].min()),
            "median": float(df["text_chars"].median()),
            "p95": float(df["text_chars"].quantile(0.95)),
            "max": int(df["text_chars"].max()),
        },
        "text_length_words": {
            "min": int(df["text_words"].min()),
            "median": float(df["text_words"].median()),
            "p95": float(df["text_words"].quantile(0.95)),
            "max": int(df["text_words"].max()),
        },
        "numerical_statistics": {
            column: _numeric_summary(df[column]) for column in NUMERIC_COLUMNS
        },
        "correlations": _correlations(df),
        "categorical_association": {
            "source_vs_split": _source_split_association(df)
        },
        "empty_text_count": int((df["description"].str.strip() == "").sum()),
        "duplicate_fingerprints_within_source": int(
            df.duplicated(["source", "fingerprint"]).sum()
        ),
        "fingerprint_groups_crossing_splits": int(
            (df.groupby(["source", "fingerprint"])["split"].nunique() > 1).sum()
        ),
        "privacy_flagged_master_records": int(df["privacy_flags"].map(bool).sum()),
        "missing_label_count": int(df["label_value"].str.strip().eq("").sum()),
        "statistical_interpretation_notes": [
            "Text character and word counts are derived from the same description and are not independent features.",
            "Skewness and excess kurtosis describe distribution shape; they do not identify data errors by themselves.",
            "The source-by-split chi-square result is exploratory and may be affected by group-based split assignment.",
        ],
    }

    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    df.groupby(["source", "split"]).size().rename("records").reset_index().to_csv(
        output_dir / "source_split_counts.csv", index=False
    )
    df.groupby(["source", "label_value"]).size().rename("records").reset_index().to_csv(
        output_dir / "label_distribution.csv", index=False
    )
    df[[
        "case_id", "source", "split", "task", "text_chars", "text_words",
        "label_key", "label_value",
    ]].to_csv(output_dir / "record_profile.csv", index=False)
    pd.crosstab(df["source"], df["split"]).to_csv(
        output_dir / "source_split_contingency.csv"
    )

    _save_count_charts(df, output_dir)
    _save_distribution_charts(df, output_dir)
    _save_relationship_charts(df, output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent / "generated",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "eda",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
