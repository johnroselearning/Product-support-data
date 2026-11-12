from .common import build_record
def adapt(source, row, features, split, index):
    required = {"issue_id", "text", "issue_priority"}
    if not required.issubset(row):
        raise ValueError(f"Unexpected helpdesk columns: {sorted(row)}")
    # Group by issue_id across all source splits; upstream split is metadata, not group identity.
    return build_record(source, split, index, row["text"], "priority_label",
                        {"priority_source": row["issue_priority"]}, row["issue_id"])
