# Phase 1: modular dataset foundation

## Pipeline
1. `downloaders/huggingface.py` downloads/loads each enabled Hugging Face dataset and persists an unmodified DatasetDict snapshot under `datasets/raw/<source>/`.
2. `adapters/` maps each supported source row into the canonical master schema.
3. `quality/privacy.py` redacts supported PII patterns and detected English person/location/facility entities before canonical export. This is automated screening, not guaranteed anonymization.
4. `splitting/group_split.py` assigns identical text to one split across sources; publisher test examples remain held out. Repeats are collapsed within the same source identity, distinct incident IDs are preserved, and conflicting labels for identical task inputs are quarantined.
5. `normalization/sft.py` converts supported task labels into conversational SFT records.
6. `exporters/jsonl.py` writes master and SFT JSONL outputs.
7. `validate.py` checks the exported master records.
8. After EDA and conservative cleaning, `features/engineer.py` builds numerical model inputs. All learned transforms fit on train only.

## Run locally
```bash
pip install -r datasets/requirements.txt
python -m spacy download en_core_web_sm
python -m unittest discover -s datasets/tests -v
python datasets/prepare.py
python datasets/validate.py
```

## Output
- `datasets/raw/<source>/`: downloaded, source-level Hugging Face snapshot (ignored by Git).
- `datasets/cache/`: Hugging Face cache (ignored by Git).
- `datasets/generated/master/{train,validation,test}.jsonl`: canonical records.
- `datasets/generated/sft/{train,validation}.jsonl`: SFT messages; test records are never exported as SFT.
- `datasets/generated/quarantine.jsonl`: records with sensitive labels or conflicting exact-input labels; raw sensitive values are not copied into quarantine.
- `datasets/generated/manifest.json`: source and count report.
- `datasets/cleaned/features/{train,validation,test}.npz`: sparse model matrices when feature engineering is enabled during cleaning.
- `datasets/cleaned/features/feature_schema.json`: feature names, target mapping and transform configuration.

Use `--force-download` to refresh source snapshots. Use `--include-bitext` only after reviewing its license obligations. Only sources with status `enabled` are downloaded by default. Raw snapshots and generated outputs are not committed.

Source revisions default to `main`; for reproducible releases, pin each source to a commit SHA in `sources.json`. Verify commercial-use and redistribution rights before use. The current source registry is a candidate inventory, not a legal opinion.

## Exploratory Data Analysis

After ingestion and validation, run `python datasets/eda.py`. It reads only normalized master JSONL files and writes `datasets/eda/` with a JSON summary, CSV profiles, and PNG charts. It never edits raw snapshots or normalized records. Review label distributions, text lengths, duplicates, and privacy flags before defining cleaning rules. EDA is descriptive; it does not automatically remove or relabel records.

## Conservative cleaning

After reviewing EDA, run `python datasets/cleaning/clean.py`. It reads normalized master records and writes a separate `datasets/cleaned/` output plus `cleaning_report.json` and `quarantine.jsonl`. It normalizes line endings, removes NUL bytes, trims edge whitespace, redacts detected PII, checks required labels, and collapses exact repeats only within the same source incident identity. It preserves label identity and does not delete records for unusual text length. It never modifies raw snapshots or normalized master files. Validate the cleaned output with `python datasets/validate.py --input datasets/cleaned`.

## Numerical feature engineering

Generate model matrices after cleaning:
```bash
python datasets/cleaning/clean.py --feature-engineering
```

Or run the stage against an existing normalized or cleaned dataset:
```bash
python datasets/features/engineer.py --input datasets/cleaned
```

This writes sparse SciPy CSR matrices (`features/{train,validation,test}.npz`), row-aligned metadata JSONL files containing `case_id` and target IDs, a feature schema, and a reusable `feature_pipeline.joblib`. Targets are never input columns. The default matrix combines text statistics and ratios, technical regex counts, word/character TF-IDF, source/task one-hot features, median imputation with missing indicators, standard scaling and degree-2 polynomial interactions.

Options include one-hot, ordinal and smoothed target encoding; standard, min-max and robust scaling; mean, median, mode or KNN numeric imputation; and constant or mode categorical imputation. One-hot is the safe default for nominal categories. Ordinal encoding is opt-in for structured categorical attributes that have a meaningful order; source/task metadata stays one-hot. Datetime attributes produce calendar fields and cyclical hour/weekday/month encodings. Sentence embeddings are optional: install `sentence-transformers` and pass `--embedding-model <model-or-local-path>`; the default does not download a model. Current enabled sources contain few structured/date fields, so those features are emitted only when an adapter has an explicit allowlist for them.

TF-IDF vocabularies, imputers, encoders and scalers are fitted on training records only. Target encoding uses leave-one-out values for training rows, and held-out rows use mappings learned from train. Labels remain in the metadata sidecars. Feature matrices include ticket-derived information and need the same privacy protections and access controls as cleaned source records.

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

The default pipeline fails closed if the English NER model is unavailable. This NER model is English-only; multilingual sources need language-specific PII detection before ingestion. The explicit --allow-regex-only option is for controlled local testing only and marks output as not production-ready. Regex and NER screening are not a guarantee of anonymization; manually review samples and quarantine uncertain records. Raw source snapshots remain unmodified under datasets/raw and must be access-controlled and excluded from Git.

For sources with structured PII columns (for example customer name, email, phone, address or account fields), adapters must use an explicit field allowlist and must not copy those columns into the canonical record. The current Help Desk adapter only selects specific non-identifying date, channel, product and numeric fields; selected string values also pass through PII redaction before export.

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


## Outcome eligibility is task-specific

Do not globally discard every ticket without a confirmed resolution. The current enabled sources supervise ticket category and source priority; those tasks can use tickets without a verified fix. They must not be repurposed as root-cause or resolution supervision. For future RCA/resolution tasks, require a verified technical diagnosis, a documented fix, and verification evidence; exclude spam, customer-only closures, and ambiguous outcomes from those task-specific targets. Keep unresolved examples for abstention/evidence-sufficiency evaluation only when a human reviewer explicitly labels them.

## Human-reviewed evaluation framework

See evaluation/README.md. Generate a 150-case stratified review queue from the cleaned test split; generate a separate dev queue from validation using --split validation:

```bash
python datasets/evaluation/sample_eval.py --size 150 --seed 20261112
```

This creates pending cases, not gold answers. Existing source labels are prefilled but unverified. Root causes, resolutions, expected tools, must-include facts and answerability remain blank until human review. Only approved cases may be copied into a versioned eval_v1.jsonl. Use a separate dev set for tuning.

The deterministic evaluator accepts approved gold cases and model predictions:

```bash
python datasets/evaluation/evaluate.py --gold datasets/evaluation/eval_v1.jsonl --predictions predictions.jsonl --output datasets/evaluation/reports/eval_v1_summary.json
```

The evaluator reports label metrics, answerability, required/forbidden fact checks and expected-tool metrics where reviewed gold fields exist. LLM judging is advisory and must be calibrated against at least 20–30 human-reviewed cases using evaluation/JUDGE_RUBRIC.md.
