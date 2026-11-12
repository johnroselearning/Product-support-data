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


## Privacy protection and identity-aware deduplication

The canonical pipeline now redacts detected emails, phone numbers, secrets, cloud/API tokens, labelled account identifiers, IP addresses and detected person/location/facility entities before writing canonical records. Person/location detection uses spaCy's en_core_web_sm model.

Install and load the model before local ingestion:

```bash
pip install -r datasets/requirements.txt
python -m spacy download en_core_web_sm
```

The default pipeline fails closed if the NER model is unavailable. The explicit --allow-regex-only option is for controlled local testing only and marks output as not production-ready. Regex and NER screening are not a guarantee of anonymization; manually review samples and quarantine uncertain records. Raw source snapshots remain unmodified under datasets/raw and must be access-controlled and excluded from Git.

Incident identifiers are hashed before canonical export. Exact-text groups are assigned to a single split. Repeated text is collapsed only within the same source identity; different incident IDs are preserved. If identical model inputs have conflicting labels, all candidates are quarantined because the current input schema cannot disambiguate them.

Near-duplicate candidates are generated for human review; they are never automatically deleted:

```bash
python datasets/quality/near_duplicates.py
```

The report is written under datasets/quality_report. A similarity score is not proof of incident identity.

## Training/evaluation leakage prevention

SFT exports now contain only train and validation records. The held-out test split remains in master JSONL for evaluation sampling and is not exported as SFT. The pipeline removes a stale SFT test file from older runs when regenerated.

Use:
- generated or cleaned SFT train for fine-tuning.
- validation for model/prompt iteration.
- the held-out test split only for evaluation review and final benchmark creation.

Never train on test records or on the evaluation review queue.

## Human-reviewed evaluation framework

See evaluation/README.md. Generate a 150-case stratified review queue from the cleaned test split:

```bash
python datasets/evaluation/sample_eval.py --size 150 --seed 20261112
```

This creates pending cases, not gold answers. Existing source labels are prefilled but unverified. Root causes, resolutions, expected tools, must-include facts and answerability remain blank until human review. Only approved cases may be copied into a versioned eval_v1.jsonl. Use a separate dev set for tuning.

The deterministic evaluator accepts approved gold cases and model predictions:

```bash
python datasets/evaluation/evaluate.py --gold datasets/evaluation/eval_v1.jsonl --predictions predictions.jsonl --output datasets/evaluation/reports/eval_v1_summary.json
```

The evaluator reports label metrics, answerability, required/forbidden fact checks and expected-tool metrics where reviewed gold fields exist. LLM judging is advisory and must be calibrated against at least 20–30 human-reviewed cases using evaluation/JUDGE_RUBRIC.md.
