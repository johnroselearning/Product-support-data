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

## Conservative cleaning

After reviewing EDA, run `python datasets/cleaning/clean.py`. It reads normalized master records and writes a separate `datasets/cleaned/` output plus `cleaning_report.json` and `quarantine.jsonl`. It only normalizes line endings, removes NUL bytes, trims edge whitespace, checks required labels, rechecks obvious PII, and removes exact duplicates created by safe normalization. It preserves label identity and does not delete records for unusual text length. It never modifies raw snapshots or normalized master files. Validate the cleaned output with `python datasets/validate.py --input datasets/cleaned`.

## Consolidated quality report and capability gaps

After EDA and cleaning, run `python datasets/quality_report.py`. It writes JSON and Markdown summaries under `datasets/quality_report/`, including cleaned counts, duplicate audit conflicts, quarantine counts, label distributions, text lengths, and a support-agent capability gap. Review `DATASET_GAP_ANALYSIS.md` before enabling any new source. New candidates remain `review_required` and are not downloaded by default.

Cleaned SFT files are written to `datasets/cleaned/sft/` alongside cleaned master files. Keep raw, normalized and cleaned layers separate. The cleaning stage is intentionally conservative and does not infer technical diagnoses or resolutions.

## Candidate customer-support datasets

Initial public-page/preview review is recorded in `research/CUSTOMER_SUPPORT_DATASET_REVIEW_2026-11-12.md`. Candidate registrations in `sources.json` do not enable ingestion. Do not commit raw Kaggle files. After downloading approved candidates locally, profile them without printing row text or personal values:

```bash
python datasets/research/profile_customer_support_csvs.py \
  datasets/raw/kaggle/suraj520 \
  datasets/raw/kaggle/waseem_alastal \
  datasets/raw/kaggle/tobi_multilingual
```

The script records file SHA-256 hashes, schema, missingness, duplicates, label distributions, text-template indicators, and text/label conflict counts. It writes aggregate output to `datasets/research/customer_support_csv_profile.json`, which is gitignored. It does not certify privacy or semantic correctness.


## Statistical EDA and optional feature engineering

The EDA summary includes central tendency, spread, distribution shape, and Pearson/Spearman/Kendall correlation for text-length measures. See datasets/FEATURE_ENGINEERING.md for the full scope.

For a classical classification baseline, run the separate feature exporter against cleaned records:

    python datasets/feature_engineering.py --label-key category
    python datasets/feature_engineering.py --label-key priority_source

This produces train/validation/test matrices without changing master or SFT records. Transformations fit on train only. Feature generation is optional; TF-IDF and numeric engineering are not needed for the SFT path. No timestamp or embedding model is present in the current canonical schema; see the feature-engineering guide for those limits and supported options. Generated matrices and the saved preprocessor are gitignored.
