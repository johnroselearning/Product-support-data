import re
PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){10,15}(?!\d)"),
    "secret": re.compile(r"(?i)\b(?:api[_-]?key|password|secret|bearer)\s*[:=]\s*\S+"),
}
def detect_flags(text):
    return [name for name, pattern in PATTERNS.items() if pattern.search(text)]
