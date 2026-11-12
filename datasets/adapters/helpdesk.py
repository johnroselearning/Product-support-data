from .common import build_record

# Explicit structured-feature allowlist. Identity and target columns are excluded.
FEATURE_FIELDS = (
    "created_at", "updated_at", "opened_at", "closed_at", "resolved_at",
    "created_date", "updated_date", "timestamp", "channel", "contact_channel",
    "product", "product_type", "response_time_hours", "resolution_time_hours",
    "ticket_age_days",
)

def adapt(source, row, features, split, index):
    required = {"issue_id", "text", "issue_priority"}
    if not required.issubset(row):
        raise ValueError(f"Unexpected Help Desk columns: {sorted(row)}")
    # issue_id is the incident identity; it is hashed by build_record before export.
    attributes = {key: row[key] for key in FEATURE_FIELDS if key in row}
    return build_record(source, split, index, row["text"], "priority_label",
                        {"priority_source": row["issue_priority"]}, row["issue_id"],
                        "incident_id", attributes=attributes)
