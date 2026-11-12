"""Source adapters: map upstream rows into the canonical master record."""
import hashlib

def digest(value):
    return hashlib.sha256(" ".join(str(value).casefold().split()).encode("utf-8")).hexdigest()

def build_record(source, upstream_split, row_index, text, task, labels,
                 source_group, source_group_type="incident_id"):
    if not isinstance(text, str) or not text.strip():
        return None
    if not labels or any(v is None or not str(v).strip() for v in labels.values()):
        return None
    fingerprint = digest(text)
    return {
        "schema_version": "1.3",
        "case_id": f"{source['id']}-{upstream_split}-{row_index}-{fingerprint[:12]}",
        "source": source["id"],
        "source_revision": source.get("revision", "main"),
        "license": source["license"],
        "attribution": source.get("attribution", ""),
        "upstream_split": str(upstream_split),
        "upstream_row": row_index,
        # Hash incident identifiers so raw IDs are not exported in canonical records.
        "source_group": digest(source_group),
        "source_group_type": source_group_type,
        # Exact-text group prevents identical inputs crossing data splits.
        "split_group": fingerprint,
        "task": task,
        "ticket": {"description": text.strip()},
        "labels": {k: str(v) for k, v in labels.items()},
        "evidence": [],
        "actions": [],
        "root_cause": "unknown",
        "resolution": "unknown",
        "quality": {"status": "unreviewed", "privacy_flags": [], "privacy_redactions": {}},
        "fingerprint": fingerprint,
    }
