"""Launch D-Tale locally for product support ticket data.

Examples:
    python datasets/research/dtale_viewer.py --input datasets/cleaned/master/train.jsonl
    python datasets/research/dtale_viewer.py --input path/to/support_tickets.csv
    python datasets/research/dtale_viewer.py --input datasets/cleaned/master
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON records into columns while retaining scalar values."""
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, (dict, list)):
                if isinstance(item, dict):
                    result.update(flatten(item, name))
                else:
                    result[name] = json.dumps(item, ensure_ascii=False)
            else:
                result[name] = item
    else:
        result[prefix or "value"] = value
    return result


def load_jsonl(path: Path):
    import pandas as pd

    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(flatten(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    if not records:
        raise ValueError(f"No records found in {path}")
    return pd.DataFrame(records)


def load_input(path: Path):
    import pandas as pd

    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise ValueError(f"No .jsonl files found in directory: {path}")
        frames = [load_jsonl(file).assign(_source_file=file.name) for file in files]
        return pd.concat(frames, ignore_index=True, sort=False)

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".jsonl", ".ndjson"}:
        return load_jsonl(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Supported inputs: CSV, JSONL/NDJSON, or Parquet")


def find_default_input() -> Path | None:
    for root in (DEFAULT_ROOT / "cleaned" / "master", DEFAULT_ROOT / "generated" / "master"):
        if root.is_dir() and any(root.glob("*.jsonl")):
            return root
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore product support ticket data with D-Tale.")
    parser.add_argument(
        "--input",
        type=Path,
        help="CSV, JSONL/NDJSON, Parquet file, or directory of JSONL files. "
             "Defaults to cleaned/master, then generated/master.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only).")
    parser.add_argument("--port", type=int, default=40000, help="Local web port (default: 40000).")
    args = parser.parse_args()

    input_path = args.input or find_default_input()
    if input_path is None:
        parser.error(
            "No normalized dataset found. Run the data pipeline first, or pass "
            "--input /path/to/tickets.csv (or a JSONL/Parquet file)."
        )

    try:
        df = load_input(input_path)
    except (OSError, ValueError, ImportError) as exc:
        print(f"Could not load dataset: {exc}", file=sys.stderr)
        return 2

    if df.empty:
        print(f"The selected dataset is empty: {input_path}", file=sys.stderr)
        return 2

    try:
        import dtale
    except ImportError:
        print(
            "D-Tale is not installed. Install the optional dependency with:\n"
            "  pip install -r datasets/requirements-dtale.txt",
            file=sys.stderr,
        )
        return 2

    print(f"Loaded {len(df):,} rows and {len(df.columns):,} columns from {input_path}")
    print("D-Tale runs on this machine only. Do not change --host for private ticket data.")
    instance = dtale.show(
        df,
        name="Product support data",
        host=args.host,
        port=args.port,
        open_browser=True,
        subprocess=False,
        reaper_on=False,
    )
    print(f"D-Tale is available at {instance._url}")
    print("Press Ctrl+C in this terminal to stop it.")
    try:
        while True:
            input()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        try:
            dtale.close(instance._data_id)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
