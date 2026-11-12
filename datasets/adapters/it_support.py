from .common import build_record

def adapt(source, row, features, split, index):
    if set(row) != {"text", "label"}:
        raise ValueError(f"Unexpected IT support columns: {sorted(row)}")
    label = row["label"]
    feature = features.get("label") if hasattr(features, "get") else None
    if isinstance(label, int) and feature is not None and hasattr(feature, "int2str"):
        label = feature.int2str(label)
    # This source has no verified incident ID. The text fingerprint is only a split group,
    # not proof that repeated rows describe the same real-world incident.
    return build_record(source, split, index, row["text"], "classification",
                        {"category": label}, row["text"], "text_fingerprint")
