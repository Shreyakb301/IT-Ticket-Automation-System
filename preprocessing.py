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
    "account lockout": "Account Lockout",
    "adobe": "Adobe",
    "android": "Android",
    "aws": "AWS",
    "azure": "Azure",
    "backup restore": "Backup Restore",
    "battery": "Battery",
    "cloud storage": "Cloud Storage",
    "connection issue": "Connection Issue",
    "crm": "CRM",
    "desktop": "Desktop",
    "docking station": "Docking Station",
    "dock": "Dock",
    "dns": "DNS",
    "erp": "ERP",
    "ethernet": "Ethernet",
    "excel": "Excel",
    "firewall": "Firewall",
    "gcp": "GCP",
    "internet": "Internet",
    "iphone": "iPhone",
    "keyboard": "Keyboard",
    "laptop": "Laptop",
    "malware": "Malware",
    "malware alert": "Malware Alert",
    "mfa": "MFA",
    "monitor": "Monitor",
    "network printer": "Network Printer",
    "new user setup": "New User Setup",
    "oracle": "Oracle",
    "outlook": "Outlook",
    "password reset": "Password Reset",
    "permission request": "Permission Request",
    "permissions": "Permissions",
    "phishing": "Phishing",
    "postgresql": "PostgreSQL",
    "printer queue": "Printer Queue",
    "security training": "Security Training",
    "serverless function": "Serverless Function",
    "shared drive access": "Shared Drive Access",
    "sql server": "SQL Server",
    "sso login": "SSO Login",
    "suspicious login": "Suspicious Login",
    "teams": "Teams",
    "toner": "Toner",
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
