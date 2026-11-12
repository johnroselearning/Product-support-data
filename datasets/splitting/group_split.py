"""Leakage-safe split assignment across exact text and incident identities."""
import hashlib
import json


def assign_group(group_id):
    digest = hashlib.sha256(f"split-component:{group_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _split_group(record):
    return record.get("split_group") or record["fingerprint"]


def assign_records(records):
    """Keep exact inputs and all messages from one incident in the same split.

    Exact-text groups are joined globally across sources. Verified incident IDs
    are joined within a source. Connected components are assigned as a unit.
    """
    count = len(records)
    parent = list(range(count))
    rank = [0] * count

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1

    fingerprint_owner = {}
    incident_owner = {}
    for index, record in enumerate(records):
        fingerprint = _split_group(record)
        if fingerprint in fingerprint_owner:
            union(index, fingerprint_owner[fingerprint])
        else:
            fingerprint_owner[fingerprint] = index

        if record.get("source_group_type") == "incident_id":
            identity = (record["source"], record["source_group"])
            if identity in incident_owner:
                union(index, incident_owner[identity])
            else:
                incident_owner[identity] = index

    components = {}
    for index, record in enumerate(records):
        components.setdefault(find(index), []).append(record)

    assignments = {}
    for component_records in components.values():
        signatures = set()
        forced_test = False
        for record in component_records:
            signatures.add("text:" + _split_group(record))
            if record.get("source_group_type") == "incident_id":
                signatures.add(
                    "incident:" + record["source"] + ":" + record["source_group"]
                )
            forced_test = forced_test or record["upstream_split"] == "test"

        component_id = hashlib.sha256(
            json.dumps(sorted(signatures), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        split = "test" if forced_test else assign_group(component_id)

        for record in component_records:
            assignments[(record["source"], _split_group(record))] = split

    return assignments
