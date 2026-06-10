from pathlib import Path
import sys
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocessing import clean_text

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
TARGETS = ["category", "subcategory", "priority"]


def load_artifacts():
    model_name = (MODEL_DIR / "embedding_model_name.txt").read_text().strip()
    embedder = SentenceTransformer(model_name)
    encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")
    models = {target: joblib.load(MODEL_DIR / f"{target}_model.joblib") for target in TARGETS}
    return embedder, encoders, models


def predict(ticket_text: str) -> dict:
    embedder, encoders, models = load_artifacts()
    X = embedder.encode([clean_text(ticket_text)], normalize_embeddings=True)
    result = {"ticket_text": ticket_text}
    for target in TARGETS:
        proba = models[target].predict_proba(X)[0]
        idx = int(np.argmax(proba))
        result[target] = encoders[target].inverse_transform([idx])[0]
        result[f"{target}_confidence"] = round(float(proba[idx]), 4)
    return result


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "My VPN disconnects every few minutes when I work from home"
    print(predict(text))
