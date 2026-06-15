from __future__ import annotations

import os
from pathlib import Path
import urllib.request
import zipfile

import joblib
import numpy as np

from preprocessing import clean_text


TARGETS = ["category", "subcategory", "priority"]
CATEGORY_AUTO_ROUTE_THRESHOLD = 0.70
SUBCATEGORY_SUGGESTION_THRESHOLD = 0.50
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "auto").strip().lower()
SUPPORT_GROUP_BY_CATEGORY = {
    "Access": "Account Access and Identity",
    "Administrative Rights": "Account Access and Identity",
    "Hardware": "Endpoint Support",
    "Hr Support": "HR Systems Support",
    "HR Support": "HR Systems Support",
    "Internal Project": "Software and Business Apps",
    "Network": "Network Operations",
    "Printer": "Endpoint Support",
    "Purchase": "IT Procurement",
    "Security": "Security Operations",
    "Software": "Software and Business Apps",
    "Storage": "Storage and Collaboration",
}
SUPPORT_GROUP_BY_ISSUE_TYPE = {
    "Account Lockout": "Account Access and Identity",
    "Active Directory": "Account Access and Identity",
    "Adobe": "Software and Business Apps",
    "Archive Request": "Storage and Collaboration",
    "Backup Request": "Storage and Collaboration",
    "Benefits": "HR Systems Support",
    "Browser": "Software and Business Apps",
    "CRM": "Software and Business Apps",
    "DNS": "Network Operations",
    "Data Loss Concern": "Storage and Collaboration",
    "Developer Tool Permission": "Account Access and Identity",
    "Docking Station": "Endpoint Support",
    "ERP": "Software and Business Apps",
    "Elevated Permission": "Account Access and Identity",
    "Email Access": "Account Access and Identity",
    "Employee Verification": "HR Systems Support",
    "Equipment Procurement": "IT Procurement",
    "Ethernet": "Network Operations",
    "File Recovery": "Storage and Collaboration",
    "IDE": "Software and Business Apps",
    "Install Approval": "Account Access and Identity",
    "Internal App Support": "Software and Business Apps",
    "Jira Project": "Software and Business Apps",
    "Laptop": "Endpoint Support",
    "Laptop Request": "IT Procurement",
    "Leave Request": "HR Systems Support",
    "Local Admin": "Account Access and Identity",
    "MFA": "Account Access and Identity",
    "MFA Enrollment": "Account Access and Identity",
    "MFA Failure": "Account Access and Identity",
    "Malware Alert": "Security Operations",
    "Monitor": "Endpoint Support",
    "Monitor Request": "IT Procurement",
    "Office Apps": "Software and Business Apps",
    "OneDrive Full": "Storage and Collaboration",
    "Password Reset": "Account Access and Identity",
    "Payroll": "HR Systems Support",
    "Permissions": "Account Access and Identity",
    "Account Compromise": "Security Operations",
    "Phishing": "Security Operations",
    "Policy Question": "HR Systems Support",
    "Network Printer": "Endpoint Support",
    "Project Permission": "Account Access and Identity",
    "Purchase Approval": "IT Procurement",
    "Quota Increase": "Storage and Collaboration",
    "Remote Office Network": "Network Operations",
    "Reporting Dashboard": "Software and Business Apps",
    "SAP Access": "Account Access and Identity",
    "Salesforce Access": "Account Access and Identity",
    "SharePoint Site": "Storage and Collaboration",
    "Shared Drive Access": "Storage and Collaboration",
    "Shared Drive Cleanup": "Storage and Collaboration",
    "Slow Internet": "Network Operations",
    "Software License": "IT Procurement",
    "Suspicious Login": "Security Operations",
    "Team Workspace": "Software and Business Apps",
    "Teams": "Software and Business Apps",
    "Temporary Admin": "Account Access and Identity",
    "Timesheet": "HR Systems Support",
    "VPN": "Network Operations",
    "VPN Access": "Network Operations",
    "VPN Connectivity": "Network Operations",
    "Vendor Quote": "IT Procurement",
    "WiFi": "Network Operations",
    "Zoom": "Software and Business Apps",
}


def _support_group_for(label: str, mapping: dict[str, str]) -> str | None:
    candidates = [
        label,
        str(label).strip(),
        str(label).strip().title(),
    ]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def _has_local_artifacts(root: Path) -> bool:
    has_transformers = all(
        (root / "transformer_models" / target / "model.safetensors").exists()
        for target in TARGETS
    )
    has_xgboost = all(
        (root / "models" / artifact).exists()
        for artifact in [
            "category_model.joblib",
            "subcategory_model.joblib",
            "priority_model.joblib",
            "label_encoders.joblib",
            "embedding_model_name.txt",
        ]
    )
    if MODEL_BACKEND == "xgboost":
        return has_xgboost
    if MODEL_BACKEND == "transformer":
        return has_transformers
    return has_transformers or has_xgboost


def _ensure_remote_artifacts(root: Path) -> None:
    if _has_local_artifacts(root):
        return

    artifact_url = os.getenv("MODEL_ARTIFACT_URL")
    if not artifact_url:
        return

    cache_dir = root / ".artifact_cache"
    cache_dir.mkdir(exist_ok=True)
    archive_path = cache_dir / "model_artifacts.zip"
    if not archive_path.exists():
        urllib.request.urlretrieve(artifact_url, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)


def _load_transformer_artifacts(root: Path) -> dict | None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    transformer_dir = root / "transformer_models"
    if not transformer_dir.exists():
        return None

    models = {}
    tokenizers = {}
    encoders = {}
    for target in TARGETS:
        target_dir = transformer_dir / target
        if not (target_dir / "model.safetensors").exists():
            return None
        tokenizers[target] = AutoTokenizer.from_pretrained(target_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(target_dir, local_files_only=True)
        model.eval()
        models[target] = model
        encoders[target] = joblib.load(target_dir / "label_encoder.joblib")
    return {
        "type": "transformer",
        "models": models,
        "tokenizers": tokenizers,
        "encoders": encoders,
    }


def _load_xgboost_artifacts(root: Path) -> dict:
    from sentence_transformers import SentenceTransformer

    model_dir = root / "models"
    if not (model_dir / "label_encoders.joblib").exists():
        raise RuntimeError("Models not found. Add artifacts or run `python train.py` first.")
    model_name = (model_dir / "embedding_model_name.txt").read_text().strip()
    return {
        "type": "xgboost",
        "embedder": SentenceTransformer(model_name),
        "encoders": joblib.load(model_dir / "label_encoders.joblib"),
        "models": {target: joblib.load(model_dir / f"{target}_model.joblib") for target in TARGETS},
    }


def load_artifacts(root: Path) -> dict:
    _ensure_remote_artifacts(root)
    if MODEL_BACKEND == "xgboost":
        return _load_xgboost_artifacts(root)
    if MODEL_BACKEND == "transformer":
        artifacts = _load_transformer_artifacts(root)
        if artifacts is None:
            raise RuntimeError("Transformer artifacts not found. Check MODEL_ARTIFACT_URL or use MODEL_BACKEND=xgboost.")
        return artifacts
    return _load_transformer_artifacts(root) or _load_xgboost_artifacts(root)


def predict_with_artifacts(artifacts: dict, ticket_text: str) -> dict:
    cleaned = clean_text(ticket_text)
    response = {"ticket_text": ticket_text, "model_type": artifacts["type"]}

    if artifacts["type"] == "transformer":
        import torch

        for target in TARGETS:
            tokenizer = artifacts["tokenizers"][target]
            model = artifacts["models"][target]
            inputs = tokenizer(cleaned, truncation=True, padding=True, max_length=128, return_tensors="pt")
            inputs = {key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask"}}
            with torch.no_grad():
                logits = model(**inputs).logits
                proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            idx = int(np.argmax(proba))
            response[target] = artifacts["encoders"][target].inverse_transform([idx])[0]
            response[f"{target}_confidence"] = round(float(proba[idx]), 4)
    else:
        X = artifacts["embedder"].encode([cleaned], normalize_embeddings=True)
        for target in TARGETS:
            proba = artifacts["models"][target].predict_proba(X)[0]
            idx = int(np.argmax(proba))
            response[target] = artifacts["encoders"][target].inverse_transform([idx])[0]
            response[f"{target}_confidence"] = round(float(proba[idx]), 4)

    response["support_group"] = (
        _support_group_for(response["subcategory"], SUPPORT_GROUP_BY_ISSUE_TYPE)
        or _support_group_for(response["category"], SUPPORT_GROUP_BY_CATEGORY)
        or response["category"]
    )
    response["support_group_confidence"] = response["category_confidence"]
    response["issue_type"] = response["subcategory"]
    response["issue_type_confidence"] = response["subcategory_confidence"]
    response["auto_route"] = response["category_confidence"] >= CATEGORY_AUTO_ROUTE_THRESHOLD
    response["needs_human_review"] = not response["auto_route"]
    if response["auto_route"]:
        response["routing_decision"] = f"Auto-route to {response['support_group']}"
        response["review_reason"] = None
    else:
        response["routing_decision"] = "Human review required"
        response["review_reason"] = (
            f"Category confidence is below {CATEGORY_AUTO_ROUTE_THRESHOLD:.0%}; "
            "route to triage queue before assignment."
        )

    if response["subcategory_confidence"] < SUBCATEGORY_SUGGESTION_THRESHOLD:
        response["review_reason"] = (
            (response["review_reason"] + " " if response["review_reason"] else "")
            + f"Subcategory confidence is below {SUBCATEGORY_SUGGESTION_THRESHOLD:.0%}; "
            "treat subcategory as a suggestion."
        )
    return response
