"""Consolidated data-quality and support-agent capability-gap report."""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

def read_jsonl(path):
    if not path.exists(): return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def run(root, out):
    generated=root/"generated"; cleaned=root/"cleaned"
    duplicates=read_jsonl(generated/"duplicate_audit.jsonl")
    quarantine=read_jsonl(generated/"quarantine.jsonl")
    clean_quarantine=read_jsonl(cleaned/"quarantine.jsonl")
    split_rows={s:read_jsonl(cleaned/"master"/(s+".jsonl")) for s in ("train","validation","test")}
    records=[r for rows in split_rows.values() for r in rows]
    by_source=Counter(r["source"] for r in records)
    by_task=Counter(r["task"] for r in records)
    labels=defaultdict(Counter)
    for r in records:
        for key,value in r.get("labels",{}).items():
            labels[r["source"]+":"+key][str(value)]+=1
    conflicts=[r for r in duplicates if r.get("label_conflict")]
    dup_by_source=Counter(r["source"] for r in duplicates)
    split_counts=Counter()
    for item in duplicates:
        if item.get("candidates"):
            for candidate in item["candidates"]:
                split_counts[str(candidate.get("upstream_split"))]+=1
        else:
            split_counts[str(item.get("excluded_upstream_split"))+" -> "+str(item.get("retained_upstream_split"))]+=1
    privacy_count=sum(1 for r in quarantine if r.get("flags"))
    conflict_record_count=sum(1 for r in quarantine if "duplicate_text_with_conflicting_labels" in r.get("reasons",[]))
    quarantine_reasons=Counter(reason for r in quarantine for reason in r.get("reasons",[]))
    lengths=sorted(len(r["ticket"]["description"]) for r in records)
    def percentile(p):
        return lengths[min(len(lengths)-1,round((len(lengths)-1)*p))] if lengths else None
    capability={
      "ticket_classification":"partially_supported: IT Support category labels",
      "priority_classification":"partially_supported: Help Desk source priority labels",
      "troubleshooting":"not_supported_by_current_verified_targets",
      "log_analysis":"not_supported_by_current_verified_targets",
      "root_cause_analysis":"not_supported_by_current_verified_targets",
      "evidence_collection":"not_supported_by_current_verified_targets",
      "tool_selection_and_execution":"not_supported_by_current_verified_targets",
      "resolution_recommendation":"not_supported_by_current_verified_targets",
      "technical_communication":"not_supported_by_current_verified_targets"
    }
    result={
      "report_version":"1.1","cleaned_record_count":len(records),
      "cleaned_by_split":{k:len(v) for k,v in split_rows.items()},
      "cleaned_by_source":dict(by_source),"cleaned_by_task":dict(by_task),
      "label_counts":{k:dict(v) for k,v in labels.items()},
      "duplicate_audit":{"audit_entries":len(duplicates),"same_label_excluded_records":sum(1 for r in duplicates if not r.get("label_conflict")),
        "conflicting_label_groups":len(conflicts),"by_source":dict(dup_by_source),
        "candidate_records_by_upstream_split":dict(split_counts),
        "conflicting_label_examples":conflicts[:25]},
      "quarantine":{"total_ingestion_records":len(quarantine),
        "privacy_flagged_records":privacy_count,
        "conflicting_duplicate_records":conflict_record_count,
        "ingestion_reasons":dict(quarantine_reasons),
        "cleaning_records":len(clean_quarantine),
        "cleaning_reasons":dict(Counter(reason for r in clean_quarantine for reason in r.get("reasons",[])))},
      "text_length_chars":{"min":min(lengths) if lengths else None,"median":percentile(.5),
        "p95":percentile(.95),"max":max(lengths) if lengths else None},
      "capability_coverage":capability,
      "recommendations":[
        "Manually review duplicate label conflicts before deciding whether any candidate label is valid.",
        "Inspect duplicate source/split groups and distinguish repeated descriptions from repeated incident identity.",
        "Review privacy-flagged quarantine manually; do not publish sensitive text in reports or artifacts.",
        "Collect more independently verified examples for rare labels; do not balance by naive duplication.",
        "Collect multi-turn troubleshooting traces with verified resolutions.",
        "Collect sanitized logs linked to independently verified diagnoses and evidence.",
        "Collect tool-call traces with schemas, arguments, outputs, errors and action-safety labels.",
        "Collect confirmed resolution records with verification evidence.",
        "Do not treat issue labels or generic customer-service answers as root-cause supervision."
      ],
      "limitations":[
        "This report summarizes metadata and does not judge semantic correctness.",
        "Zero regex privacy flags is not proof of de-identification.",
        "The current two sources supervise classification and priority, not a complete technical-support agent."
      ]
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"dataset_quality_report.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Dataset Quality and Capability Gap Report","",
      "Cleaned records: **"+str(len(records))+"**","","## Records by source",""]
    lines += ["- "+k+": "+str(v) for k,v in by_source.items()] or ["- No cleaned records found."]
    lines += ["","## Duplicate audit","","- Duplicate audit entries: "+str(len(duplicates)),
      "- Same-label excluded records: "+str(result["duplicate_audit"]["same_label_excluded_records"]),
      "- Conflicting-label groups: "+str(len(conflicts)),
      "- Conflicting-label candidate records quarantined: "+str(conflict_record_count),
      "- Privacy-flagged records quarantined: "+str(privacy_count),
      "- Cleaning quarantine records: "+str(len(clean_quarantine)),"","## Capability coverage",""]
    lines += ["- **"+k.replace("_"," ").title()+"**: "+v for k,v in capability.items()]
    lines += ["","## Recommended next data acquisition","",
      "1. Multi-turn IT troubleshooting traces with verified resolutions.",
      "2. Incident timelines linking symptoms, diagnostic steps, evidence and confirmed root causes.",
      "3. Sanitized logs with independently verified diagnoses.",
      "4. Tool-call traces with schemas, arguments, outputs, errors and action-safety labels.",
      "5. Resolution records with explicit verification evidence.",
      "6. Human-reviewed technical communication examples for customer updates and handoffs."]
    (out/"DATASET_QUALITY_REPORT.md").write_text("\n".join(lines)+"\n")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"quality_report")
    a=p.parse_args()
    print(json.dumps(run(a.root,a.output),indent=2))
if __name__=="__main__": main()
