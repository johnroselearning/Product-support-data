"""Exploratory analysis of normalized master JSONL files; never mutates source data."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLITS=("train","validation","test")
NUMERIC_FEATURES=("text_chars","text_words")

def load_records(root):
    rows=[]
    for split in SPLITS:
        path=root/"master"/f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                r=json.loads(line)
                description=r["ticket"]["description"]
                rows.append({
                    "split":split,"source":r["source"],"task":r["task"],
                    "description":description,
                    "text_chars":len(description),
                    "text_words":len(description.split()),
                    "label_key":next(iter(r["labels"]), ""),
                    "label_value":str(next(iter(r["labels"].values()), "")),
                    "case_id":r["case_id"],"fingerprint":r["fingerprint"],
                    "source_group":r["source_group"],
                    "source_group_type":r.get("source_group_type", "unknown"),
                    "privacy_flags":r.get("quality",{}).get("privacy_flags",[])})
    return pd.DataFrame(rows)

def _finite_number(value):
    try:
        number=float(value)
    except (TypeError,ValueError):
        return None
    return number if math.isfinite(number) else None

def _json_number(value):
    number=_finite_number(value)
    if number is None:
        return None
    return int(number) if number.is_integer() else number

def _numeric_summary(values):
    series=pd.to_numeric(values,errors="coerce").dropna()
    if series.empty:
        return {
            "count":0,"mean":None,"median":None,"mode_values":[],"mode_frequency":None,
            "mode_values_truncated":0,"variance_sample":None,"standard_deviation_sample":None,
            "iqr":None,"min":None,"p25":None,"p75":None,"p95":None,"max":None,
            "skewness":None,"kurtosis_excess":None
        }
    counts=series.value_counts()
    top_frequency=int(counts.max())
    modes=counts[counts==top_frequency]
    max_modes=20
    return {
        "count":int(series.count()),
        "mean":_json_number(series.mean()),
        "median":_json_number(series.median()),
        "mode_values":[_json_number(v) for v in modes.index[:max_modes]],
        "mode_frequency":top_frequency,
        "mode_values_truncated":max(0,int(len(modes)-max_modes)),
        "variance_sample":_json_number(series.var(ddof=1)) if len(series)>1 else None,
        "standard_deviation_sample":_json_number(series.std(ddof=1)) if len(series)>1 else None,
        "iqr":_json_number(series.quantile(.75)-series.quantile(.25)),
        "min":_json_number(series.min()),
        "p25":_json_number(series.quantile(.25)),
        "p75":_json_number(series.quantile(.75)),
        "p95":_json_number(series.quantile(.95)),
        "max":_json_number(series.max()),
        "skewness":_json_number(series.skew()),
        "kurtosis_excess":_json_number(series.kurt())
    }

def _correlation_summary(frame):
    numeric=frame.loc[:,list(NUMERIC_FEATURES)].apply(pd.to_numeric,errors="coerce")
    valid=numeric.dropna()
    result={"pair":"text_chars_vs_text_words","valid_records":int(len(valid))}
    for method in ("pearson","spearman","kendall"):
        if len(valid)<2:
            result[method]=None
        else:
            result[method]=_json_number(valid.corr(method=method).iloc[0,1])
    return result

def run(input_dir, output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    df=load_records(input_dir)
    if df.empty:
        raise ValueError(f"No master records found under {input_dir}")
    stats_all={feature:_numeric_summary(df[feature]) for feature in NUMERIC_FEATURES}
    stats_by_split={
        str(split):{
            feature:_numeric_summary(group[feature])
            for feature in NUMERIC_FEATURES
        }
        for split,group in df.groupby("split",sort=True)
    }
    cross_source_fingerprints = df.groupby("fingerprint")["source"].nunique()
    cross_source_group_ids = set(cross_source_fingerprints[cross_source_fingerprints > 1].index)
    summary={
      "records":int(len(df)),
      "by_source":{str(k):int(v) for k,v in df.source.value_counts().items()},
      "by_split":{str(k):int(v) for k,v in df.split.value_counts().items()},
      "by_task":{str(k):int(v) for k,v in df.task.value_counts().items()},
      "label_distribution":{str(k):int(v) for k,v in df.groupby(["source","label_value"],dropna=False).size().items()},
      "text_length_chars":{"min":int(df.text_chars.min()),"median":float(df.text_chars.median()),"p95":float(df.text_chars.quantile(.95)),"max":int(df.text_chars.max())},
      "text_length_words":{"min":int(df.text_words.min()),"median":float(df.text_words.median()),"p95":float(df.text_words.quantile(.95)),"max":int(df.text_words.max())},
      "numeric_statistics":{"all_records":stats_all,"by_split":stats_by_split},
      "correlations":_correlation_summary(df),
      "empty_text_count":int((df.description.str.strip()=="").sum()),
      "duplicate_fingerprints_within_source":int(df.duplicated(["source","fingerprint"]).sum()),
      "fingerprint_groups_appearing_in_multiple_sources":int((cross_source_fingerprints > 1).sum()),
      "records_in_cross_source_exact_text_groups":int(df.fingerprint.isin(cross_source_group_ids).sum()),
      "fingerprint_groups_crossing_splits":int((df.groupby(["fingerprint"])["split"].nunique()>1).sum()),
      "incident_groups_crossing_splits":int((df[df.source_group_type == "incident_id"].groupby(["source","source_group"])["split"].nunique()>1).sum()) if "source_group_type" in df.columns else 0,
      "privacy_flagged_master_records":int(df.privacy_flags.map(bool).sum()),
      "missing_label_count":int(df.label_value.str.strip().eq("").sum())
    }
    (output_dir/"eda_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    df.groupby(["source","split"]).size().rename("records").reset_index().to_csv(output_dir/"source_split_counts.csv",index=False)
    df.groupby(["source","label_value"]).size().rename("records").reset_index().to_csv(output_dir/"label_distribution.csv",index=False)
    df[["case_id","source","split","task","text_chars","text_words","label_key","label_value"]].to_csv(output_dir/"record_profile.csv",index=False)
    charts=[
      ("source_counts.png",df.source.value_counts(),"Records by source"),
      ("split_counts.png",df.split.value_counts(),"Records by split"),
      ("label_counts.png",df.groupby(["source","label_value"]).size(),"Label distribution by source")
    ]
    for filename,series,title in charts:
      fig,ax=plt.subplots(figsize=(11,6))
      series.plot(kind="bar",ax=ax)
      ax.set_title(title)
      ax.set_ylabel("Records")
      ax.tick_params(axis="x",rotation=55)
      fig.tight_layout()
      fig.savefig(output_dir/filename,dpi=140)
      plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    df.text_chars.plot(kind="hist",bins=40,ax=ax)
    ax.set_title("Ticket description length (characters)")
    ax.set_xlabel("Characters")
    fig.tight_layout()
    fig.savefig(output_dir/"text_length_distribution.png",dpi=140)
    plt.close(fig)
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path(__file__).resolve().parent/"generated")
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"eda")
    a=p.parse_args()
    print(json.dumps(run(a.input,a.output),indent=2,allow_nan=False))

if __name__=="__main__":
    main()
