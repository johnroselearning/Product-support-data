import hashlib

def assign_group(group_id):
    digest = hashlib.sha256(f"exact-text:{group_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _split_group(record):
    return record.get("split_group") or record["fingerprint"]


def assign_records(records):
    """Keep identical model inputs together across all sources and upstream splits."""
    forced_test = {
        _split_group(record)
        for record in records
        if record["upstream_split"] == "test"
    }
    return {
        (record["source"], _split_group(record)):
        (
            "test" if _split_group(record) in forced_test
            else assign_group(_split_group(record))
        )
        for record in records
    }
