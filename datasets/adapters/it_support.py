from .common import build_record
def adapt(source, row, features, split, index):
    if set(row) != {"text", "label"}:
        raise ValueError(f"Unexpected IT support columns: {sorted(row)}")
    label = row["label"]
    feature = features.get("label") if hasattr(features, "get") else None
    if isinstance(label, int) and feature is not None and hasattr(feature, "int2str"):
        label = feature.int2str(label)
    return build_record(source, split, index, row["text"], "classification",
                        {"category": label}, build_record_group(row["text"]))
def build_record_group(text):
    from .common import digest
    return digest(text)
