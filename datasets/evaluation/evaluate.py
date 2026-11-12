"""Deterministic evaluation for approved PSI gold cases.

Prediction JSONL fields: case_id, labels, answerable, answer, tool_calls.
This evaluator intentionally refuses pending or rejected review cases.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def normalize(value):
    return " ".join(str(value).casefold().split())


def f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(gold_rows, prediction_rows):
    gold_by_id = {}
    gold_splits = set()
    for row in gold_rows:
        case_id = row.get("case_id")
        if not case_id or case_id in gold_by_id:
            raise ValueError(f"Missing or duplicate gold case_id: {case_id}")
        review = row.get("review", {})
        if review.get("status") != "approved":
            raise ValueError(
                f"Gold case {case_id} is not approved; status={review.get('status')}"
            )
        if not review.get("reviewer") or not review.get("reviewed_at"):
            raise ValueError(f"Approved gold case {case_id} lacks reviewer metadata")
        if not row.get("gold", {}).get("labels"):
            raise ValueError(f"Approved gold case {case_id} has no gold labels")
        source_split = row.get("provenance", {}).get("source_split")
        if source_split not in {"validation", "test"}:
            raise ValueError(f"Gold case {case_id} has invalid source split")
        gold_splits.add(source_split)
        gold_by_id[case_id] = row

    if len(gold_splits) > 1:
        raise ValueError("Do not mix dev/validation and final test cases in one evaluation run")

    pred_by_id = {}
    for row in prediction_rows:
        case_id = row.get("case_id")
        if not case_id or case_id in pred_by_id:
            raise ValueError(f"Missing or duplicate prediction case_id: {case_id}")
        pred_by_id[case_id] = row

    missing = sorted(set(gold_by_id) - set(pred_by_id))
    extra = sorted(set(pred_by_id) - set(gold_by_id))
    if missing or extra:
        raise ValueError(
            f"Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    task_counts = defaultdict(Counter)
    label_counts = defaultdict(lambda: defaultdict(Counter))
    answerability = Counter()
    tool_totals = Counter()
    must_include_total = Counter()
    must_not_total = Counter()
    cases_with_required = 0
    cases_with_forbidden = 0

    for case_id, gold_row in gold_by_id.items():
        pred = pred_by_id[case_id]
        task = gold_row["task"]
        expected_labels = gold_row["gold"]["labels"]
        predicted_labels = pred.get("labels", {})
        exact = expected_labels == predicted_labels
        task_counts[task]["cases"] += 1
        task_counts[task]["label_exact_match"] += int(exact)

        for key, expected in expected_labels.items():
            actual = predicted_labels.get(key, "__MISSING__")
            field = (task, key)
            counts = label_counts[field][str(expected)]
            if str(actual) == str(expected):
                counts["tp"] += 1
            else:
                counts["fn"] += 1
                label_counts[field][str(actual)]["fp"] += 1

        expected_answerable = gold_row["gold"].get("answerable")
        predicted_answerable = pred.get("answerable")
        if isinstance(expected_answerable, bool):
            answerability["cases"] += 1
            answerability["correct"] += int(
                isinstance(predicted_answerable, bool)
                and expected_answerable == predicted_answerable
            )
            if expected_answerable is False and predicted_answerable is True:
                answerability["false_confident_answers"] += 1

        expected_tools = set(gold_row["gold"].get("expected_tools") or [])
        actual_tools_raw = pred.get("tool_calls") or []
        actual_tools = {
            item.get("name") if isinstance(item, dict) else str(item)
            for item in actual_tools_raw
        }
        if expected_tools:
            tool_totals["cases"] += 1
            tool_totals["tp"] += len(expected_tools & actual_tools)
            tool_totals["fp"] += len(actual_tools - expected_tools)
            tool_totals["fn"] += len(expected_tools - actual_tools)

        answer = normalize(pred.get("answer", ""))
        required = [normalize(x) for x in gold_row["gold"].get("must_include", [])]
        forbidden = [normalize(x) for x in gold_row["gold"].get("must_not", [])]
        if required:
            cases_with_required += 1
            must_include_total["required_facts"] += len(required)
            must_include_total["included_facts"] += sum(x in answer for x in required)
        if forbidden:
            cases_with_forbidden += 1
            must_not_total["forbidden_facts"] += len(forbidden)
            must_not_total["violations"] += sum(x in answer for x in forbidden)

    task_metrics = {}
    for task, counts in task_counts.items():
        task_metrics[task] = {
            "cases": counts["cases"],
            "label_exact_match_accuracy": (
                counts["label_exact_match"] / counts["cases"]
                if counts["cases"] else None
            )
        }

    label_metrics = {}
    for (task, field), classes in label_counts.items():
        class_scores = []
        for label, counts in classes.items():
            score = f1_from_counts(
                counts["tp"], counts["fp"], counts["fn"]
            )
            class_scores.append(score["f1"])
            label_metrics[f"{task}:{field}:{label}"] = {
                "support": counts["tp"] + counts["fn"],
                **score
            }
        label_metrics[f"{task}:{field}:macro"] = {
            "macro_f1": sum(class_scores) / len(class_scores) if class_scores else None
        }

    total_cases = len(gold_by_id)
    exact_cases = sum(x["label_exact_match"] for x in task_counts.values())
    report = {
        "report_version": "1.0",
        "gold_cases": total_cases,
        "prediction_cases": len(pred_by_id),
        "overall_label_exact_match_accuracy": (
            exact_cases / total_cases if total_cases else None
        ),
        "by_task": task_metrics,
        "label_metrics": label_metrics,
        "answerability": {
            "cases": answerability["cases"],
            "accuracy": (
                answerability["correct"] / answerability["cases"]
                if answerability["cases"] else None
            ),
            "false_confident_answers": answerability["false_confident_answers"]
        },
        "source_split": next(iter(gold_splits)) if gold_splits else None,
        "tool_use": {
            "cases_with_expected_tools": tool_totals["cases"],
            **(
                f1_from_counts(tool_totals["tp"], tool_totals["fp"], tool_totals["fn"])
                if tool_totals["cases"] else
                {"precision": None, "recall": None, "f1": None}
            )
        },
        "required_fact_checks": {
            "cases": cases_with_required,
            "included": must_include_total["included_facts"],
            "total": must_include_total["required_facts"],
            "compliance": (
                must_include_total["included_facts"] / must_include_total["required_facts"]
                if must_include_total["required_facts"] else None
            )
        },
        "forbidden_fact_checks": {
            "cases": cases_with_forbidden,
            "violations": must_not_total["violations"],
            "total": must_not_total["forbidden_facts"],
            "violation_rate": (
                must_not_total["violations"] / must_not_total["forbidden_facts"]
                if must_not_total["forbidden_facts"] else None
            )
        },
        "limitations": [
            "Only human-approved gold cases are accepted.",
            "Must-include and must-not checks use normalized substring matching, not semantic equivalence.",
            "Tool-call scoring compares tool names only; query and argument correctness require a separate rubric.",
            "LLM-as-judge and human adjudication are not performed by this deterministic evaluator."
        ]
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate(read_jsonl(args.gold), read_jsonl(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
