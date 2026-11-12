"""Shared JSON Schema validation for PSI human-reviewed evaluation cases."""
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_PATH = Path(__file__).resolve().parent / "evaluation_case.schema.json"


def load_validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_case(case, validator=None):
    validator = validator or load_validator()
    errors = sorted(
        validator.iter_errors(case),
        key=lambda error: (list(error.absolute_path), error.message)
    )
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in errors
    ]


def require_valid_case(case, case_id="<unknown>", validator=None):
    errors = validate_case(case, validator)
    if errors:
        raise ValueError(
            f"Evaluation case {case_id} does not match schema: " + "; ".join(errors)
        )
