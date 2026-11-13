# Synthetic product-support datasets

These AI-generated datasets were created for Product-support-data learning and experimentation. They are not exports of real support tickets.

## Separate tracks

- `classification/synthetic_product_support_tickets.csv`: 300 rows and 12 columns. Includes synthetic ticket text, product/channel context, category and priority labels, plus illustrative root-cause and resolution fields.
- `investigation/synthetic_product_support_model_dataset.csv`: 500 rows and 25 columns. Includes synthetic reports, impact/environment context, investigation evidence, recommended checks, illustrative causes/resolution steps, and escalation fields.

All rows in both files have `is_synthetic=true`. Stable case/ticket IDs support row traceability; exclude them from predictive features. Keep the two files separate by default.

## Training and evaluation cautions

Fields such as root cause, resolution status, resolution steps, outcome labels and escalation results may be known only after an intake-time prediction. Treat them as targets or exclude them from input features unless their availability at prediction time is established. The investigation outcomes are illustrative labels, not expert-verified ground truth. Review label definitions, class coverage, leakage and split design before training or evaluation.

## Privacy scan

A local scan on 2026-11-13 found no email, phone, token or IPv4 pattern matches in either file. Automated screening can miss sensitive information; this is not a privacy certification. Do not add real customer-support exports to this synthetic directory.
