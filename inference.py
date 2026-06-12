from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import clean_text


TARGETS = ["category", "subcategory", "priority"]
CATEGORY_AUTO_ROUTE_THRESHOLD = 0.70
SUBCATEGORY_SUGGESTION_THRESHOLD = 0.50


def _load_transformer_artifacts(root: Path) -> dict | None:
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
    return _load_transformer_artifacts(root) or _load_xgboost_artifacts(root)


def predict_with_artifacts(artifacts: dict, ticket_text: str) -> dict:
    cleaned = clean_text(ticket_text)
    response = {"ticket_text": ticket_text, "model_type": artifacts["type"]}

    if artifacts["type"] == "transformer":
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

    response["auto_route"] = response["category_confidence"] >= CATEGORY_AUTO_ROUTE_THRESHOLD
    response["needs_human_review"] = not response["auto_route"]
    if response["auto_route"]:
        response["routing_decision"] = f"Auto-route to {response['category']}"
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
