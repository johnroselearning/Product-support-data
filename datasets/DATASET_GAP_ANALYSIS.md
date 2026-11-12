# Technical Support Agent: Dataset Gap Analysis

Last reviewed: 2026-11-12

## 1. What the current verified corpus can teach

The current ingested sources are:
- IT Support Tickets: single-ticket category classification.
- Help Desk Tickets: source priority classification.

They do **not** provide verified targets for diagnosis, causal reasoning, log interpretation, tool selection, remediation, or resolution verification. Empty `evidence`, `actions`, `root_cause`, and `resolution` fields are placeholders, not supervision.

## 2. Capability coverage and acquisition priorities

| Capability | Current coverage | Required data shape | Priority |
|---|---|---|---|
| Ticket classification | Partial | Ticket text + trusted category | P2 |
| Priority/impact triage | Partial | Ticket + impact/urgency + priority rubric | P1 |
| Clarifying questions | Missing | Initial ticket + missing context + next question + answer | P1 |
| Troubleshooting | Missing | Multi-turn symptoms, diagnostic steps, observations, next action | P0 |
| Log analysis | Missing | Sanitized logs + time window + diagnosis + supporting lines | P0 |
| Root-cause analysis | Missing | Incident timeline + evidence + confirmed cause + alternatives rejected | P0 |
| Tool selection | Missing | Tool schemas + state + arguments + tool output + rationale labels | P0 |
| Safe remediation | Missing | Preconditions + approval policy + action + outcome + rollback | P0 |
| Resolution verification | Missing | Before/after telemetry + explicit success criterion | P0 |
| Customer communication | Missing | Verified technical state + audience + update/handoff | P2 |
| Incident process/SLA | Missing | Ticket lifecycle events + assignment/escalation/SLA outcomes | P1 |

P0 means essential for a genuine investigation agent, not necessarily data to fine-tune immediately.

## 3. Public candidates researched (not automatically approved)

### A. Infra-Ops Agent — incident-resolution trajectories
- Dataset: https://huggingface.co/datasets/quantranger/infra-ops-incidents
- License shown on dataset card: MIT.
- Dataset card reports 150,000 SFT trajectories, 30,000 preference pairs and 20,000 rejection trajectories; 15 incident types and a 25-tool registry.
- Important limitation: the card explicitly describes the data as synthetic and templated, not real production incidents. Treat as synthetic scenario/tool-use augmentation and evaluate separately from real-world evidence.
- The Hugging Face viewer currently reports a dataset-generation error, although individual files are available. Inspect raw JSONL and schema locally before adapter implementation.
- Decision: candidate for controlled pilot after schema, provenance, license and quality review; do not mix silently with real incidents.

### B. Incident Management event log
- Repository: https://github.com/processminingdata/incidentmanagement
- Repository describes 31,000+ tickets and 242,000+ events, with assignment, escalation, work-in-progress, solved and closed lifecycle stages.
- Repository includes an MIT license file. Verify that the license covers the bundled data itself and inspect the handbook/source provenance before commercial use.
- Best use: process mining, escalation paths, cycle time and SLA workflow analytics. It is not automatically root-cause or troubleshooting supervision.
- Decision: candidate for process/SLA adapter after archive inspection and data-license confirmation.

### C. Help Desk Tickets — richer Mendeley version
- Dataset page: https://data.mendeley.com/datasets/btm76zndnt/2
- Description includes issues.csv and issues_change_history.csv; fields include ticket category, priority, assignment, timestamps, resolution time and status/assignee changes. It also describes anonymization.
- The existing 357-row Hugging Face Help Desk source is not equivalent to the full event-history tables.
- Decision: inspect version-specific license and available tables; potentially valuable for lifecycle/SLA/assignment analysis. Avoid duplicating the already ingested messages.

### D. ITBench trajectories
- Dataset: https://huggingface.co/datasets/ibm-research/ITBench-Trajectories
- The dataset description reports agent investigation trajectories across SRE scenarios, with alerts, events, metrics, traces, logs and Kubernetes state, plus root-cause evaluation metrics.
- Existing registry marks this source noncommercial. Re-check the exact current license and terms before any use; do not enable for commercial training unless explicitly permitted.
- Decision: valuable for research/evaluation only if terms allow the intended use; maintain a separate evaluation split and do not leak benchmark scenarios into training.

### E. Planning2Interaction (P2I)
- Dataset: https://huggingface.co/datasets/anonymous-p2i/Planning2Interaction
- Described as multi-turn function-calling trajectories focused on recovery from tool errors, invalid state, stale identifiers and missing prerequisites.
- It is general tool-use data rather than IT support domain data. Verify license and schema independently.
- Decision: optional tool-recovery augmentation, not a substitute for IT incident trajectories.

## 4. Recommended acquisition strategy

1. Prioritize first-party or partner-contributed, de-identified support cases with verified fixes and explicit training rights.
2. Add real event histories and resolution verification, not just ticket titles and priority labels.
3. Add sanitized logs/metrics/traces linked to a confirmed incident and root cause.
4. Add tool traces with versioned schemas, arguments, outputs, error cases, approvals and rollback paths.
5. Use synthetic datasets only as clearly tagged augmentation; keep real and synthetic evaluation results separate.
6. Build a small expert-reviewed gold set (at least 100 cases across major incident families) before fine-tuning.
7. Keep benchmark/evaluation datasets out of training and deduplicate by incident/scenario/repository group.

## 5. Canonical investigation trajectory fields to collect

A future investigation example should include:
- incident_id, source, license, source_revision, split_group
- initial_report, environment, affected_service, impact, urgency
- observations: timestamp, source_system, query/tool, raw_result_reference, sanitized_excerpt
- hypotheses: candidate_cause, evidence_for, evidence_against, confidence
- actions: tool_name, schema_version, arguments, approval_required, tool_result, error
- verified_root_cause, resolution_action, rollback_plan
- verification: pre_state, post_state, SLO/acceptance criterion, outcome
- customer_update, handoff, reviewer_id/pseudonymous reviewer, review_status

Never fill root cause or resolution from an LLM guess. Require source evidence or expert review.

## 6. Release gates

A dataset source is not enabled until:
- exact license and commercial-use rights are recorded;
- source data and schema are inspected;
- PII/security review is completed;
- group-level split strategy is defined;
- adapter tests pass;
- representative records are manually reviewed;
- benchmark leakage is ruled out.
