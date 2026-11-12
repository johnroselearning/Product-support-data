# Phase 1: modular dataset foundation

## Pipeline
1. `downloaders/huggingface.py` downloads/loads each enabled Hugging Face dataset and persists an unmodified DatasetDict snapshot under `datasets/raw/<source>/`.
2. `adapters/` maps each supported source row into the canonical master schema.
3. `quality/privacy.py` flags obvious emails, phone numbers and secret-like strings for quarantine. This is heuristic screening, not anonymization.
4. `splitting/group_split.py` assigns whole source groups; publisher test groups remain held out. Exact duplicate text prefers the publisher test copy.
5. `normalization/sft.py` converts supported task labels into conversational SFT records.
6. `exporters/jsonl.py` writes master and SFT JSONL outputs.
7. `validate.py` checks the exported master records.

## Run locally
```bash
pip install -r datasets/requirements.txt
python -m unittest discover -s datasets/tests -v
python datasets/prepare.py
python datasets/validate.py
```

## Output
- `datasets/raw/<source>/`: downloaded, source-level Hugging Face snapshot (ignored by Git).
- `datasets/cache/`: Hugging Face cache (ignored by Git).
- `datasets/generated/master/{train,validation,test}.jsonl`: canonical records.
- `datasets/generated/sft/{train,validation,test}.jsonl`: SFT messages.
- `datasets/generated/quarantine.jsonl`: privacy-flagged records.
- `datasets/generated/manifest.json`: source and count report.

Use `--force-download` to refresh source snapshots. Use `--include-bitext` only after reviewing its license obligations. Only sources with status `enabled` are downloaded by default. Raw snapshots and generated outputs are not committed.

Source revisions default to `main`; for reproducible releases, pin each source to a commit SHA in `sources.json`. Verify commercial-use and redistribution rights before use. The current source registry is a candidate inventory, not a legal opinion.

## Exploratory Data Analysis

After ingestion and validation, run `python datasets/eda.py`. It reads only normalized master JSONL files and writes `datasets/eda/` with a JSON summary, CSV profiles, and PNG charts. It never edits raw snapshots or normalized records. Review label distributions, text lengths, duplicates, and privacy flags before defining cleaning rules. EDA is descriptive; it does not automatically remove or relabel records.
