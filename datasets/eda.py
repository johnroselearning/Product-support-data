"""Exploratory analysis of normalized master JSONL files; never mutates source data."""
import argparse
import json
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLITS=("train","validation","test")
def load_records(root):
    rows=[]
    for split in SPLITS:
        path=root/"master"/f"{split}.jsonl"
        if not path.exists(): continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                r=json.loads(line)
                rows.append({"split":split,"source":r["source"],"task":r["task"],
                    "description":r["ticket"]["description"],
                    "text_chars":len(r["ticket"]["description"]),
                    "text_words":len(r["ticket"]["description"].split()),
                    "label_key":next(iter(r["labels"]), ""),
                    "label_value":str(next(iter(r["labels"].values()), "")),
                    "case_id":r["case_id"],"fingerprint":r["fingerprint"],
                    "source_group":r["source_group"],
                    "source_group_type":r.get("source_group_type", "unknown"),
                    "privacy_flags":r.get("quality",{}).get("privacy_flags",[])})
    return pd.DataFrame(rows)
def run(input_dir, output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    df=load_records(input_dir)
    if df.empty: raise ValueError(f"No master records found under {input_dir}")
    summary={
      "records":int(len(df)),
      "by_source":{str(k):int(v) for k,v in df.source.value_counts().items()},
      "by_split":{str(k):int(v) for k,v in df.split.value_counts().items()},
      "by_task":{str(k):int(v) for k,v in df.task.value_counts().items()},
      "label_distribution":{str(k):int(v) for k,v in df.groupby(["source","label_value"],dropna=False).size().items()},
      "text_length_chars":{"min":int(df.text_chars.min()),"median":float(df.text_chars.median()),"p95":float(df.text_chars.quantile(.95)),"max":int(df.text_chars.max())},
      "text_length_words":{"min":int(df.text_words.min()),"median":float(df.text_words.median()),"p95":float(df.text_words.quantile(.95)),"max":int(df.text_words.max())},
      "empty_text_count":int((df.description.str.strip()=="").sum()),
      "duplicate_fingerprints_within_source":int(df.duplicated(["source","fingerprint"]).sum()),
      "duplicate_fingerprints_across_sources":int(df.duplicated(["fingerprint"]).sum()),
      "fingerprint_groups_crossing_splits":int((df.groupby(["fingerprint"])["split"].nunique()>1).sum()),
      "incident_groups_crossing_splits":int((df[df.source_group_type == "incident_id"].groupby(["source","source_group"])["split"].nunique()>1).sum()) if "source_group_type" in df.columns else 0,
      "privacy_flagged_master_records":int(df.privacy_flags.map(bool).sum()),
      "missing_label_count":int(df.label_value.str.strip().eq("").sum())
    }
    (output_dir/"eda_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
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
      ax.set_title(title); ax.set_ylabel("Records"); ax.tick_params(axis="x",rotation=55)
      fig.tight_layout(); fig.savefig(output_dir/filename,dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    df.text_chars.plot(kind="hist",bins=40,ax=ax)
    ax.set_title("Ticket description length (characters)"); ax.set_xlabel("Characters")
    fig.tight_layout(); fig.savefig(output_dir/"text_length_distribution.png",dpi=140); plt.close(fig)
    return summary
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path(__file__).resolve().parent/"generated")
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"eda")
    a=p.parse_args()
    print(json.dumps(run(a.input,a.output),indent=2))
if __name__=="__main__": main()
