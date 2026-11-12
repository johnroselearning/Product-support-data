# PSI evaluation framework

## What is implemented

- A JSON Schema for task-specific support-agent evaluation cases.
- A deterministic stratified sampler that selects from the cleaned, held-out test split.
- A human-review queue with source labels prefilled but explicitly unverified.
- A deterministic evaluator for approved classification labels, answerability, required/forbidden facts and expected tools.
- A judge rubric and human review guide.

## Generate a 150-case review queue

Run after ingestion and cleaning:

\`\`\`bash
python datasets/evaluation/sample_eval.py --size 150 --seed 20261112
\`\`\`

Outputs:
- review_queue_v1.jsonl: pending cases for human annotation.
- review_queue_manifest.json: counts, strata, seed and provenance.

This queue is not an evaluation benchmark until each case is reviewed. The sampler uses only cleaned/master/test.jsonl, stratifies by task and source label, removes cross-source exact-text repeats, and excludes exact-text groups with conflicting labels. It does not fabricate root causes, resolutions, expected tools or answerability.

## Approve and release a benchmark

Follow REVIEW_GUIDE.md. Copy only fully reviewed and approved cases into eval_v1.jsonl. Keep the released file read-only and record its SHA-256 checksum. Use dev_v1.jsonl for prompt/model iteration; do not tune against eval_v1.jsonl.

## Run deterministic evaluation

Prediction JSONL format:

\`\`\`json
{"case_id":"source-test-12-abcd1234","labels":{"category":"Software"},"answerable":true,"answer":"...","tool_calls":["search_logs"]}
\`\`\`

Run:

\`\`\`bash
python datasets/evaluation/evaluate.py \\
  --gold datasets/evaluation/eval_v1.jsonl \\
  --predictions predictions.jsonl \\
  --output datasets/evaluation/reports/eval_v1_summary.json
\`\`\`

The evaluator refuses pending/rejected gold cases and fails on missing or extra predictions. It reports label exact match, per-class precision/recall/F1, macro F1, answerability, required/forbidden fact checks and expected-tool precision/recall/F1 where those gold fields exist.

Phrase checks are deterministic substring checks, not semantic equivalence. Use the judge rubric for reviewed qualitative assessment; calibrate against at least 20–30 human-scored cases before trusting an LLM judge.

## Data safety

Evaluation JSONL files and reports are ignored by Git by default. Do not commit raw or personal data. Store only redacted, reviewed cases in an access-controlled location. The review queue may be shared as a private GitHub Actions artifact for authorized repository users.
