# Canonical normalization contract (schema 1.2)

Each enabled source has a dedicated adapter in `adapters/`. Adapters preserve source labels and provenance; they do not infer root causes or resolutions.

## Master record
- `case_id`: source-scoped deterministic identifier
- `source`, `source_revision`, `license`, `attribution`: provenance
- `upstream_split`, `upstream_row`, `source_group`: source lineage and leakage-safe grouping
- `task`, `ticket.description`, `labels`: task-specific target
- `evidence`, `actions`: empty unless explicitly supplied by a trustworthy source
- `root_cause`, `resolution`: unknown unless verified
- `quality.status`: unreviewed until later EDA and human review
- `fingerprint`: normalized exact-text hash

## Current adapters
| Source | Mapping | Group |
|---|---|---|
| IT Support Tickets | text -> description; label -> category | normalized text fingerprint |
| Help Desk Tickets | text -> description; issue_priority -> priority_source | issue_id |

## Pipeline modules
- `downloaders/huggingface.py`: downloads and persists the upstream DatasetDict under `raw/<source>/`.
- `adapters/`: source-specific row mapping.
- `quality/privacy.py`: heuristic screening; flagged rows go to quarantine.
- `splitting/group_split.py`: assigns whole source groups; any group represented in publisher test remains in test.
- `normalization/sft.py`: creates task-specific conversational records.
- `exporters/jsonl.py`: writes JSONL.
- `prepare.py`: orchestrates the pipeline.
- `validate.py`: validates exported master records.

## Not yet implemented
Polaris, Glaive, SWE-bench Extra, Bitext, Tobi-Bueck, ITBench and Loghub require individual license review and dedicated adapters. Registration is not ingestion.

## Quality gates
1. Verify source revision, license, attribution and schema.
2. Download and preserve a source snapshot.
3. Normalize without inventing labels, evidence or outcomes.
4. Quarantine heuristic privacy flags for manual review.
5. Deduplicate within source, preferring the publisher test record when exact duplicates straddle splits.
6. Split by source group, preserving publisher test groups.
7. Run schema and leakage validation.
8. Conduct EDA and data cleaning before treating any output as training-ready.
