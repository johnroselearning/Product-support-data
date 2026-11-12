# Canonical normalization contract (schema 1.4)

Each enabled source has a dedicated adapter in `adapters/`. Adapters preserve source labels and provenance; they do not infer root causes or resolutions.

## Master record
- `case_id`: deterministic source-scoped row identifier based on the redacted text
- `source`, `source_revision`, `license`, `attribution`: provenance
- `upstream_split`, `upstream_row`: source lineage
- `source_group`, `source_group_type`: hashed source identity and its declared meaning
- `split_group`: exact-text group used to keep identical inputs in one split
- `task`, `ticket.description`, `labels`: task input and target
- `attributes`: explicit allowlist of non-identifying scalar model inputs; must omit IDs, labels, outcomes and personal fields
- `evidence`, `actions`: empty unless explicitly supplied by a trustworthy source
- `root_cause`, `resolution`: unknown unless verified
- `quality.privacy_flags`, `quality.privacy_redactions`: detected PII categories and redaction counts
- `fingerprint`: hash of the redacted, normalized ticket text

## Current adapters
| Source | Mapping | Group |
|---|---|---|
| IT Support Tickets | text -> description; label -> category | text fingerprint (split-only identity) |
| Help Desk Tickets | text -> description; issue_priority -> priority_source; selected optional structured fields -> attributes | hashed issue ID |

Structured attributes are selected by explicit adapter allowlists and passed through the PII redactor before canonical export. The feature pipeline rechecks privacy patterns before building matrices.

## Pipeline modules
- `downloaders/huggingface.py`: downloads and persists the upstream DatasetDict under `raw/<source>/`.
- `adapters/`: source-specific row mapping and structured-field allowlists.
- `quality/privacy.py`: regex and spaCy screening/redaction before canonical export.
- `splitting/group_split.py`: assigns connected exact-text and verified-incident groups as a unit; publisher test components remain held out.
- `normalization/sft.py`: creates task-specific conversational records for train/validation only.
- `features/engineer.py`: creates model matrices; vocabularies, imputers, encoders and scalers fit on train only.
- `exporters/jsonl.py`: writes JSONL.
- `prepare.py`: orchestrates ingestion, redaction, deduplication, splitting and export.
- `cleaning/clean.py`: redacts and cleans a separate output layer; may optionally invoke feature engineering after cleaning.
- `validate.py`: validates schema, split boundaries and detected PII.

## Quality gates
1. Verify source revision, license, attribution and schema.
2. Download and preserve a source snapshot with restricted local permissions.
3. Redact detected PII before canonical export; do not export raw structured PII columns.
4. Normalize without inventing labels, evidence or outcomes.
5. Quarantine conflicting labels for identical task inputs; collapse only repeats from the same source incident identity.
6. Keep exact-text and verified incident groups inside one split.
7. Keep test records out of SFT and training; use them only for evaluation.
8. Run schema/privacy validation, EDA and conservative cleaning before treating outputs as training-ready.
9. Fit feature transforms on train only and keep targets in metadata sidecars, outside the feature matrix.
