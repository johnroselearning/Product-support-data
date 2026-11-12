# Canonical normalization contract (v1)

## Dataset status
The source registry is a catalog of **candidates**, not downloaded files. Only `it_support` is enabled for default ingestion. `bitext` is opt-in. The other sources require dedicated adapters, licensing review and validation before training. Never silently interpret unsupported source columns.

## Master record
- `case_id`: deterministic source-scoped identifier
- `source`, `upstream_split`, `upstream_row`: provenance
- `task`: classification or response (future: triage, tool_call, investigation)
- `ticket.description`: raw reported text, with no synthetic root cause added
- `label_or_answer`: original label or response
- `evidence`: empty unless source provides verifiable observations
- `actions`: empty unless source provides observed tool actions
- `root_cause`, `resolution`: unknown unless explicitly verified
- `quality`: unreviewed until validated
- `fingerprint`: normalized text hash for exact duplicate detection

## Planned adapters (not yet operational)
| Source | Input mapping | Grouping / caveat |
|---|---|---|
| Polaris | subject+body -> ticket; topic/type/priority/routing -> separate labels | group by event_id to prevent incident leakage |
| Glaive | parse SYSTEM, USER, ASSISTANT function calls and FUNCTION RESPONSE | validate tool schemas and arguments; never flatten tool calls |
| SWE-bench Extra | problem_statement -> issue; patch/test_patch -> verified patch candidates | group by repo and instance_id; honor per-repo license |
| ITBench | session.jsonl -> steps; judge_output -> measured outcomes | group by scenario, not individual run; noncommercial |
| Loghub | log lines -> evidence observations | log source alone does not establish root cause |
| Tobi-Bueck | ticket body, queue, priority and answer -> distinct fields | noncommercial |

## Required gates before fine-tuning
1. Verify dataset revision and legal terms; store attribution and provenance.
2. Run source-specific schema validation; fail on unexpected fields.
3. Quarantine secrets and personal information; conduct human privacy review.
4. Normalize to the master schema without inventing missing labels or outcomes.
5. Deduplicate, then split by incident/event/repository group; preserve benchmark test sets.
6. Generate conversational SFT data only for tasks with reliable targets.
7. Publish a manifest with source counts, exclusions, splits, and checksums.

Current pipeline is an initial implementation, **not** a validated training release. It has not been run against live Hugging Face data in this repository.
