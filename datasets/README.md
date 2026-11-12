# Phase 1: dataset foundation

This repository stores reproducible ingestion code, not raw support tickets. Review the source license, attribution and privacy obligations before commercial use or redistribution.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r datasets/requirements.txt
python -m unittest discover -s datasets/tests -v
python datasets/prepare.py
# Include conversational data only after license review:
python datasets/prepare.py --include-bitext
```

Outputs under `datasets/generated/` are ignored by Git. `master/` retains provenance and unknown investigation fields; `sft/` contains conversational fine-tuning examples; `manifest.json` reports counts. Suspicious PII is excluded and counted, not considered fully anonymized. A human privacy review is still required. Publisher test partitions are held out; exact duplicates are removed, but semantic and incident-level leakage checks remain necessary.

Source labels must not be confused with verified root causes, priorities or resolutions. The Bitext corpus is conversational customer service, not evidence of correct technical investigations.

## Verify generated data

```bash
python datasets/prepare.py
python datasets/validate.py
```

Validation checks canonical required fields, empty ticket descriptions, missing labels, duplicate IDs and fingerprints, and source-group leakage between splits. It does not certify semantic quality or absence of sensitive data. The helpdesk source uses its documented default configuration, not a separate `reporter_messages` config argument.
