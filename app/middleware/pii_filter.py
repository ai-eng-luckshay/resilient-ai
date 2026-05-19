import re

_SENSITIVE_KEYS = frozenset({
    "authorization", "x-api-key", "api-key", "token", "password",
    "secret", "credential", "access_key", "private_key",
})

_CARD_PATTERN = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact_headers(headers: dict) -> dict:
    return {
        k: ("***" if k.lower() in _SENSITIVE_KEYS else v)
        for k, v in headers.items()
    }


def redact_text(text: str) -> str:
    text = _CARD_PATTERN.sub("****-****-****-****", text)
    text = _EMAIL_PATTERN.sub("***@***.***", text)
    text = _SSN_PATTERN.sub("***-**-****", text)
    return text


def redact_dict(data: dict) -> dict:
    result = {}
    for k, v in data.items():
        if k.lower() in _SENSITIVE_KEYS:
            result[k] = "***"
        elif isinstance(v, dict):
            result[k] = redact_dict(v)
        elif isinstance(v, str):
            result[k] = redact_text(v)
        else:
            result[k] = v
    return result
