import hashlib

def assign_group(source_id, group_id):
    digest = hashlib.sha256(f"{source_id}:{group_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"

def _split_group(record):
    return record.get("split_group") or record["source_group"]

def assign_records(records):
    """Assign exact-text groups together; publisher test groups remain held out."""
    forced_test = {
        (r["source"], _split_group(r))
        for r in records if r["upstream_split"] == "test"
    }
    return {
        (r["source"], _split_group(r)):
        ("test" if (r["source"], _split_group(r)) in forced_test
         else assign_group(r["source"], _split_group(r)))
        for r in records
    }
