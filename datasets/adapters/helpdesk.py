from .common import build_record

def adapt(source, row, features, split, index):
    required = {"issue_id", "text", "issue_priority"}
    if not required.issubset(row):
        raise ValueError(f"Unexpected Help Desk columns: {sorted(row)}")
    # issue_id is the incident identity; it is hashed by build_record before export.
    return build_record(source, split, index, row["text"], "priority_label",
                        {"priority_source": row["issue_priority"]}, row["issue_id"], "incident_id")
