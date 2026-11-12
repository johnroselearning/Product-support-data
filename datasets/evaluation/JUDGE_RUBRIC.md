# PSI technical-support response judge rubric

Use this rubric only after a case has been human-approved. The judge must receive the ticket, permitted context, approved gold facts and model output. It must not invent missing ground truth.

Score each dimension from 0 to 2:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Technical correctness | Incorrect or materially misleading | Partly correct; important gap | Correct and consistent with gold |
| Evidence grounding | Unsupported claims or fabricated evidence | Some claims grounded; citations incomplete | Claims trace to supplied evidence |
| Root-cause discipline | Invents or overstates cause | States uncertainty inconsistently | Separates observed, inferred and unknown |
| Resolution quality | Unsafe or ineffective recommendation | Plausible but incomplete | Safe, actionable and matches verified fix |
| Tool choice | Wrong, unnecessary or unsafe tools | Partly appropriate | Expected tools and queries used appropriately |
| Communication | Confusing or misleading | Understandable but incomplete | Clear, concise and appropriately qualified |

## Critical failure flags

Record separately from the numeric score:
- hallucinated_evidence
- unsupported_root_cause
- unsafe_action
- privacy_disclosure
- failed_to_abstain
- unnecessary_tool_execution
- missed_required_escalation

A critical failure must not be hidden by a high fluency or communication score.

## Judge calibration

Before using the judge for routine evaluation:
1. Have a human independently score at least 20–30 approved cases.
2. Compare each dimension and critical-failure flag.
3. Report exact agreement and weighted agreement where applicable.
4. Inspect disagreements and revise the rubric/prompt.
5. Re-calibrate after changing judge model or rubric.

LLM-as-judge is advisory. Human adjudication remains authoritative for benchmark release.
