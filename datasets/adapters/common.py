"""Source adapters: map upstream rows into the canonical master record."""
import hashlib

def digest(value):
    return hashlib.sha256(" ".join(str(value).casefold().split()).encode("utf-8")).hexdigest()

def build_record(source, upstream_split, row_index, text, task, labels, source_group):
    if not isinstance(text, str) or not text.strip():
        return None
    if not labels or any(v is None or not str(v).strip() for v in labels.values()):
        return None
    fingerprint = digest(text)
    return {
        "schema_version": "1.2",
        "case_id": f"{source['id']}-{upstream_split}-{row_index}-{fingerprint[:12]}",
        "source": source["id"],
        "source_revision": source.get("revision", "main"),
        "license": source["license"],
        "attribution": source.get("attribution", ""),
        "upstream_split": str(upstream_split),
        "upstream_row": row_index,
        "source_group": str(source_group),
        "task": task,
        "ticket": {"description": text.strip()},
        "labels": {k: str(v) for k, v in labels.items()},
        "evidence": [],
        "actions": [],
        "root_cause": "unknown",
        "resolution": "unknown",
        "quality": {"status": "unreviewed", "privacy_flags": []},
        "fingerprint": fingerprint,
    }
