from __future__ import annotations

import re


ACRONYM_LABELS = {"aws", "dns", "erp", "gcp", "mfa", "vpn", "wifi"}

CATEGORY_MAP = {
    "acc": "Access",
    "access": "Access",
    "clo": "Cloud",
    "cloud": "Cloud",
    "dat": "Database",
    "database": "Database",
    "har": "Hardware",
    "hardware": "Hardware",
    "mob": "Mobile Device",
    "mobile device": "Mobile Device",
    "net": "Network",
    "network": "Network",
    "pri": "Printer",
    "printer": "Printer",
    "sec": "Security",
    "security": "Security",
    "sof": "Software",
    "software": "Software",
}

PRIORITY_MAP = {
    "critical": "Critical",
    "urgent": "Critical",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}

SUBCATEGORY_MAP = {
    "account compromise": "Account Compromise",
    "adobe": "Adobe",
    "android": "Android",
    "aws": "AWS",
    "azure": "Azure",
    "docking station": "Docking Station",
    "dns": "DNS",
    "erp": "ERP",
    "gcp": "GCP",
    "iphone": "iPhone",
    "laptop": "Laptop",
    "malware": "Malware",
    "mfa": "MFA",
    "monitor": "Monitor",
    "network printer": "Network Printer",
    "outlook": "Outlook",
    "password reset": "Password Reset",
    "permissions": "Permissions",
    "phishing": "Phishing",
    "postgresql": "PostgreSQL",
    "printer queue": "Printer Queue",
    "sql server": "SQL Server",
    "teams": "Teams",
    "vpn": "VPN",
    "wifi": "WiFi",
}

TEXT_PREFIX_RE = re.compile(
    r"\b(ticket from email|user reported|can someone check this|hi it|fyi)\b:?",
    re.IGNORECASE,
)
DEPT_RE = re.compile(r"\bdept\s*=\s*[a-z]+(?:\s+[a-z]+)?\b", re.IGNORECASE)
NOISE_RE = re.compile(
    r"\b(?:asset|tag|error|err|code)[-_:\s]*[a-z0-9-]+\b|"
    r"\b(?:screenshot attached|thanks|regards)\b",
    re.IGNORECASE,
)
PUNCT_RE = re.compile(r"[^a-z0-9#+.\s-]+")
REPEATED_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)


def clean_text(text: object) -> str:
    cleaned = str(text or "").lower().strip()
    cleaned = TEXT_PREFIX_RE.sub(" ", cleaned)
    cleaned = DEPT_RE.sub(" ", cleaned)
    cleaned = NOISE_RE.sub(" ", cleaned)
    cleaned = PUNCT_RE.sub(" ", cleaned)
    cleaned = REPEATED_WORD_RE.sub(r"\1", cleaned)
    return " ".join(cleaned.split())


def normalize_label(value: object, target: str) -> str:
    raw = " ".join(str(value or "").strip().split())
    key = raw.lower()
    if target == "category":
        return CATEGORY_MAP.get(key, raw.title())
    if target == "priority":
        return PRIORITY_MAP.get(key, raw.title())
    if target == "subcategory":
        if key in SUBCATEGORY_MAP:
            return SUBCATEGORY_MAP[key]
        if key in ACRONYM_LABELS:
            return key.upper()
        return raw.title()
    return raw
