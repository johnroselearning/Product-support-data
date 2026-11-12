"""Conservative, auditable cleaning of normalized records. Never edits raw/master inputs."""
import argparse
import sys
import json
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exporters.jsonl import write_jsonl
from quality.privacy import detect_flags
from normalization.sft import to_sft

SPLITS=("train","validation","test")
def clean_text(value):
    # Only safe, non-semantic normalization: normalize line endings, remove NULs, trim edges.
    return value.replace("\r\n","\n").replace("\r","\n").replace("\x00","").strip()
def clean_record(record):
    out=dict(record)
    ticket=dict(record.get("ticket",{}))
    text=ticket.get("description")
    if not isinstance(text,str):
        return None,["invalid_description_type"]
    text=clean_text(text)
    if not text:
        return None,["empty_after_trim"]
    ticket["description"]=text
    out["ticket"]=ticket
    labels=record.get("labels")
    if not isinstance(labels,dict) or not labels:
        return None,["missing_labels"]
    cleaned_labels={}
    for key,value in labels.items():
        if value is None or not str(value).strip():
            return None,["empty_label"]
        # Trim label edges but preserve original spelling, case and class identity.
        cleaned_labels[key]=str(value).strip()
    out["labels"]=cleaned_labels
    flags=detect_flags(text+"\n"+" ".join(cleaned_labels.values()))
    if flags:
        return None,["privacy:"+flag for flag in flags]
    return out,[]
def run(input_dir,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    seen={}
    accepted={s:[] for s in SPLITS}
    quarantine=[]
    report={"input_records":0,"accepted_records":0,"excluded_records":0,
            "exclusion_reasons":{},"by_split":{},"actions":[
              "Normalize line endings and remove NUL bytes",
              "Trim description and label edges",
              "Preserve label values and spelling",
              "Do not remove records solely for unusual length",
              "Do not synthesize or infer labels, root causes, evidence or resolutions"
            ]}
    reasons=Counter()
    # Read test first so if an exact duplicate exists across splits, retain its test copy.
    for split in ("test","validation","train"):
        path=input_dir/"master"/f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as f:
            for line_no,line in enumerate(f,1):
                report["input_records"]+=1
                record=json.loads(line)
                cleaned,errors=clean_record(record)
                if errors:
                    for reason in errors: reasons[reason]+=1
                    quarantine.append({"case_id":record.get("case_id"),"source":record.get("source"),
                                       "original_split":split,"line":line_no,"reasons":errors})
                    continue
                # Recompute fingerprint using the same whitespace/case-insensitive fingerprint rule.
                import hashlib
                fp=hashlib.sha256(" ".join(cleaned["ticket"]["description"].casefold().split()).encode()).hexdigest()
                key=(cleaned["source"],fp)
                if key in seen:
                    reasons["duplicate_after_cleaning"]+=1
                    quarantine.append({"case_id":cleaned["case_id"],"source":cleaned["source"],
                                       "original_split":split,"line":line_no,
                                       "reasons":["duplicate_after_cleaning"],
                                       "retained_case_id":seen[key]})
                    continue
                seen[key]=cleaned["case_id"]
                cleaned["fingerprint"]=fp
                accepted[split].append(cleaned)
    for split,records in accepted.items():
        write_jsonl(output_dir/"master"/f"{split}.jsonl",records)
        sft_records=[to_sft(record) for record in records]
        write_jsonl(output_dir/"sft"/f"{split}.jsonl",sft_records)
        report["by_split"][split]=len(records)
    report["accepted_records"]=sum(map(len,accepted.values()))
    report["excluded_records"]=len(quarantine)
    report["exclusion_reasons"]=dict(reasons)
    write_jsonl(output_dir/"quarantine.jsonl",quarantine)
    (output_dir/"cleaning_report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path(__file__).resolve().parents[1]/"generated")
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"cleaned")
    a=p.parse_args()
    print(json.dumps(run(a.input,a.output),indent=2))
if __name__=="__main__": main()
