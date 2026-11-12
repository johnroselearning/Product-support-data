import json
SYSTEM = "You are a technical support engineer. Never invent evidence or verified root causes."
def to_sft(record):
    text, labels = record["ticket"]["description"], record["labels"]
    if record["task"] == "classification":
        user = "Classify this support ticket. Return JSON with category only:\n" + text
        answer = json.dumps({"category": labels["category"]}, ensure_ascii=False)
    elif record["task"] == "priority_label":
        user = "Return the original ticket priority label as JSON with priority_source only:\n" + text
        answer = json.dumps({"priority_source": labels["priority_source"]}, ensure_ascii=False)
    else:
        raise ValueError(f"No SFT target formatter for task {record['task']}")
    return {"case_id": record["case_id"], "source": record["source"],
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": answer}]}
