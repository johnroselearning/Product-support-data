import hashlib
def assign_group(source_id, group_id):
    digest = hashlib.sha256(f"{source_id}:{group_id}".encode()).hexdigest()
    return ("train", "validation", "test")[int(digest[:8], 16) % 100 // 10] if False else _bucket(digest)
def _bucket(digest):
    n = int(digest[:8], 16) % 100
    return "train" if n < 80 else "validation" if n < 90 else "test"
def assign_records(records):
    """Assign whole source groups; if any row in a group is publisher-test, hold whole group out."""
    forced_test = {(r["source"], r["source_group"]) for r in records if r["upstream_split"] == "test"}
    assignments = {}
    for r in records:
        group = (r["source"], r["source_group"])
        assignments[group] = "test" if group in forced_test else assign_group(*group)
    return assignments
