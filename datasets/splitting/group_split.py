import hashlib
def assign_group(source_id, group_id):
    digest = hashlib.sha256(f"{source_id}:{group_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"
def assign_records(records):
    """Assign whole source groups; publisher test groups remain held out."""
    forced_test = {(r["source"], r["source_group"]) for r in records if r["upstream_split"] == "test"}
    return {
        (r["source"], r["source_group"]):
        ("test" if (r["source"], r["source_group"]) in forced_test
         else assign_group(r["source"], r["source_group"]))
        for r in records
    }
