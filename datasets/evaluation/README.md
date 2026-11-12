# PSI evaluation framework

## What is implemented

- JSON Schema for task-specific support-agent evaluation cases.
- Deterministic stratified sampling from either the cleaned validation split (dev queue) or held-out test split (final evaluation queue).
- Human-review queues with source labels prefilled but explicitly unverified.
- Immutable promotion of approved queues into versioned dev/evaluation files with SHA-256 manifests.
- Deterministic evaluator for approved classification labels, answerability, required/forbidden facts and expected tools.
- Judge rubric and human review guide.

## Generate review queues

Run after ingestion and cleaning.

Final evaluation candidates (held-out test split):

```bash
python datasets/evaluation/sample_eval.py --split test --size 150 --seed 20261112
```

Development candidates (validation split):

```bash
python datasets/evaluation/sample_eval.py --split validation --size 100 --seed 20261112 --output datasets/evaluation/dev_review_queue_v1.jsonl --manifest datasets/evaluation/dev_review_queue_manifest.json
```

The default output for the test split is review_queue_v1.jsonl. The sampler stratifies by task and source label, removes cross-source exact-text repeats and excludes exact-text groups with conflicting labels. It does not invent root causes, resolutions, expected tools or answerability.

The generated files are review queues, not approved gold datasets. Source labels are prefilled but remain unverified until a human reviewer approves each case.

## Review and promote

Follow REVIEW_GUIDE.md. After every applicable field has been reviewed, set review.status to approved, provide reviewer and reviewed_at, and complete evidence-backed gold fields. Rejected cases must remain excluded.

Promote a development queue:

```bash
python datasets/evaluation/promote_eval.py --input datasets/evaluation/dev_review_queue_v1.jsonl --output datasets/evaluation/dev_v1.jsonl --manifest datasets/evaluation/dev_v1_manifest.json --split validation
```

Promote the final test queue:

```bash
python datasets/evaluation/promote_eval.py --input datasets/evaluation/review_queue_v1.jsonl --output datasets/evaluation/eval_v1.jsonl --manifest datasets/evaluation/eval_v1_manifest.json --split test
```

Promotion refuses pending/rejected cases, missing reviewer metadata, empty gold labels, or a mismatched source split. Output files are created exclusively: an existing version is never overwritten. The manifest records the SHA-256 checksum. Create a new version filename for any new release.

Use dev_v1.jsonl for prompt/model iteration. Never tune prompts, retrieval or models against eval_v1.jsonl.

## Run deterministic evaluation

Prediction JSONL format:

```json
{"case_id":"source-test-12-abcd1234","labels":{"category":"Software"},"answerable":true,"answer":"...","tool_calls":["search_logs"]}
```

Run:

```bash
python datasets/evaluation/evaluate.py --gold datasets/evaluation/eval_v1.jsonl --predictions predictions.jsonl --output datasets/evaluation/reports/eval_v1_summary.json
```

The evaluator refuses pending/rejected gold cases and fails on missing or extra predictions. It reports label exact match, per-class precision/recall/F1, macro F1, answerability, required/forbidden fact checks and expected-tool precision/recall/F1 where those gold fields exist. It refuses to mix validation and test cases in one run.

Phrase checks use normalized substring matching, not semantic equivalence. Tool scoring compares tool names only; query and argument correctness need separate review. Use JUDGE_RUBRIC.md for qualitative assessment and calibrate an LLM judge against at least 20–30 human-scored cases before trusting it.

## Data safety

Evaluation JSONL files and reports are ignored by Git by default. Do not commit raw or personal data. Store only redacted, reviewed cases in an access-controlled location. Review queues and gold evaluation JSONL files remain local and gitignored. CI uploads aggregate manifests and quality summaries only; it does not publish ticket text.

## Schema enforcement

Every review queue and released benchmark case must conform to `evaluation_case.schema.json` (JSON Schema Draft 2020-12). Both promotion and deterministic evaluation validate cases against this schema. An approved case must include a non-empty reviewer identity and an ISO-8601 review timestamp. This prevents incomplete or structurally inconsistent cases from being promoted or scored.

The schema is intentionally task-extensible: task-specific labels live in `gold.labels`; root cause, resolution, answerability, required/forbidden facts, and expected tools remain nullable or empty until a human reviewer has evidence to approve them. Do not fill these fields from assumptions or language-model guesses.
