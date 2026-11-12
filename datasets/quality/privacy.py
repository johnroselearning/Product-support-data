"""PII screening and redaction for support-ticket text.

Regex coverage is supplemented by spaCy NER for person and location entities.
The production pipeline requires en_core_web_sm. No detector guarantees complete
anonymization; human review and language-specific screening remain required.
"""
from __future__ import annotations

import ipaddress
import re
from functools import lru_cache

PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)"),
    "secret_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|token|password|passwd|pwd|secret|client[_-]?secret)"
        r"\s*[:=]\s*['\"]?[^,\s;'\"}]+"
    ),
    "bearer_token": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "provider_token": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|sk_(?:live|test)_[A-Za-z0-9]{16,}|"
        r"rk_live_[A-Za-z0-9]{16,}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
    ),
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
        re.DOTALL
    ),
    "account_identifier": re.compile(
        r"(?i)\b(?:account|customer|tenant|subscription|user|employee|order|member)"
        r"\s*(?:id|number|no\.?|#)\s*[:=#-]?\s*[A-Z0-9][A-Z0-9_-]{3,}\b"
    ),
    "ipv4": re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"),
    "ipv6": re.compile(r"(?i)(?<![\w.])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w.])"),
    "street_address": re.compile(
        r"(?i)\b\d{1,6}\s+[A-Z0-9.'-]+(?:\s+[A-Z0-9.'-]+){0,5}\s+"
        r"(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|"
        r"nagar|layout|cross|main road)\b"
    ),
}

REDACTION = "[REDACTED_PII]"


@lru_cache(maxsize=1)
def _load_ner():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def privacy_capabilities():
    return {
        "regex_screening": True,
        "person_and_location_ner": _load_ner() is not None,
        "ner_model": "en_core_web_sm",
        "ner_language": "en"
    }


def ensure_person_ner_available():
    if _load_ner() is None:
        raise RuntimeError(
            "PII-safe ingestion requires spaCy NER model en_core_web_sm. "
            "Install dependencies and run: python -m spacy download en_core_web_sm. "
            "For a deliberate local-only regex trial, use the explicit --allow-regex-only flag."
        )


def _is_valid_ip(value):
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _collect_spans(text, use_ner=True):
    spans = []
    flags = set()
    counts = {}

    for name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(0)
            if name in {"ipv4", "ipv6"} and not _is_valid_ip(value):
                continue
            spans.append((match.start(), match.end(), name))
            flags.add(name)
            counts[name] = counts.get(name, 0) + 1

    nlp = _load_ner() if use_ner else None
    if nlp is not None and text.strip():
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "GPE", "LOC", "FAC"}:
                name = "person_name" if ent.label_ == "PERSON" else "location_or_facility"
                spans.append((ent.start_char, ent.end_char, name))
                flags.add(name)
                counts[name] = counts.get(name, 0) + 1

    return spans, sorted(flags), counts


def detect_flags(text, use_ner=True):
    """Return PII categories found; does not expose matched values."""
    if not isinstance(text, str):
        return []
    _, flags, _ = _collect_spans(text, use_ner=use_ner)
    return flags


def redact_text(text, require_ner=False, use_ner=True):
    """Redact detected spans and return (safe_text, categories, per-category counts)."""
    if not isinstance(text, str):
        raise TypeError("redact_text expects a string")
    if require_ner:
        ensure_person_ner_available()

    spans, flags, counts = _collect_spans(text, use_ner=use_ner)
    if not spans:
        return text, flags, counts

    # Merge overlapping spans to avoid leaking fragments or applying offsets twice.
    spans.sort(key=lambda item: (item[0], item[1]))
    merged = []
    for start, end, name in spans:
        if merged and start <= merged[-1][1]:
            old_start, old_end, names = merged[-1]
            merged[-1] = (old_start, max(old_end, end), names | {name})
        else:
            merged.append((start, end, {name}))

    safe = text
    for start, end, _names in reversed(merged):
        safe = safe[:start] + REDACTION + safe[end:]
    return safe, flags, counts
