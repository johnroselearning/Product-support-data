"""Download Hugging Face datasets and persist an immutable local source snapshot."""
from pathlib import Path
from datasets import load_dataset, load_from_disk

def download_source(source, raw_dir, cache_dir, force=False):
    target = raw_dir / source["id"]
    if target.exists() and not force:
        return load_from_disk(str(target)), {"mode": "local_snapshot", "path": str(target)}
    kwargs = {
        "path": source["repo"],
        "cache_dir": str(cache_dir),
        "revision": source.get("revision", "main"),
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    dataset = load_dataset(**kwargs)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        import shutil
        shutil.rmtree(target)
    dataset.save_to_disk(str(target))
    return dataset, {"mode": "downloaded", "path": str(target), "revision": kwargs["revision"]}
