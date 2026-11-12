# Training data boundary

Use only these files:

- Fine-tuning: cleaned/sft/train.jsonl (or generated/sft/train.jsonl before cleaning).
- Development/validation: cleaned/sft/validation.jsonl.

The pipeline deliberately does not export held-out test records into SFT JSONL. Evaluation review queues and final evaluation files are never training inputs.

Every future training entrypoint should call training.guard.assert_safe_training_input(path, role) before reading data. Example from the repository root:

```python
import sys
sys.path.insert(0, "datasets")
from training.guard import assert_safe_training_input

train_path = assert_safe_training_input(
    "datasets/cleaned/sft/train.jsonl", role="train"
)
```

Run the guard directly:

```bash
python datasets/training/guard.py datasets/cleaned/sft/train.jsonl --role train
```

This path guard is a defense-in-depth check, not a substitute for reviewing the actual training code and data lineage.
