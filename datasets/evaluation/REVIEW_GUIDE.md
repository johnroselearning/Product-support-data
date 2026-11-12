# Human review guide for PSI evaluation cases

## Purpose

The generated review queue is sampled either from the validation split for dev_v1 or from the held-out test split for eval_v1. It is NOT an approved gold evaluation set. Source labels are prefilled for convenience and must be checked against the ticket.

## Review workflow

For each case:

1. Confirm that the ticket is understandable and contains no remaining personal or secret information.
2. Verify the source label. Correct it only when the ticket evidence supports a different label; otherwise reject or mark unanswerable.
3. Add evidence-backed root_cause and resolution only if the source provides enough verified information. Never infer a diagnosis from a category or priority label.
4. Fill must_include with independently checkable facts a correct answer must contain.
5. Fill must_not with dangerous or explicitly unsupported recommendations.
6. Fill expected_tools only when the case includes enough environment and access context to specify tools and queries.
7. Set answerable to true or false only after reviewing evidence. Use null if the task does not assess answerability.
8. Add tags where appropriate: hard, ambiguous, multi_cause, missing_logs, contradictory, out_of_scope, long_thread, needs_escalation.
9. Set review.status to approved only after all applicable gold fields are complete. Add reviewer identity and an ISO-8601 timestamp.
10. Set review.status to rejected if the ticket is unusable, privacy-unsafe, or has irreconcilable source labels.

## Evaluation split rules

- Sample dev cases only from the source validation split.
- Sample final evaluation cases only from the held-out source test split.
- Never copy review-queue cases into training or fine-tuning data.
- Keep review_queue_v1.jsonl pending until human review is complete.
- Promote approved validation cases to dev_v1.jsonl.
- Promote approved test cases to eval_v1.jsonl using promote_eval.py; promotion creates immutable files and SHA-256 manifests.
- Do not tune prompts, retrieval or models against the final test set. Use a separate dev set for iteration.
- Store a checksum and reviewer metadata for each released evaluation version.

## Important limitation

The current public datasets mainly provide category and priority labels. They do not establish verified root causes, diagnostic evidence, tool calls or resolutions. Those fields must remain empty unless a reviewer has trustworthy supporting material.
