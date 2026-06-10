from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from preprocessing import clean_text, normalize_label

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "tickets.csv"
TARGETS = ["category", "subcategory", "priority"]

app = FastAPI(title="IT Ticket Automated Classifier", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TicketRequest(BaseModel):
    ticket_text: str = Field(..., min_length=5, examples=["VPN disconnects every few minutes from home."])

class PredictionResponse(BaseModel):
    ticket_text: str
    category: str
    subcategory: str
    priority: str
    category_confidence: float
    subcategory_confidence: float
    priority_confidence: float

_artifacts = {}


def load_artifacts():
    if _artifacts:
        return _artifacts
    if not (MODEL_DIR / "label_encoders.joblib").exists():
        raise RuntimeError("Models not found. Run `python train.py` first.")
    model_name = (MODEL_DIR / "embedding_model_name.txt").read_text().strip()
    _artifacts["embedder"] = SentenceTransformer(model_name)
    _artifacts["encoders"] = joblib.load(MODEL_DIR / "label_encoders.joblib")
    _artifacts["models"] = {target: joblib.load(MODEL_DIR / f"{target}_model.joblib") for target in TARGETS}
    return _artifacts

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(payload: TicketRequest):
    try:
        artifacts = load_artifacts()
        X = artifacts["embedder"].encode([clean_text(payload.ticket_text)], normalize_embeddings=True)
        response = {"ticket_text": payload.ticket_text}
        for target in TARGETS:
            proba = artifacts["models"][target].predict_proba(X)[0]
            idx = int(np.argmax(proba))
            response[target] = artifacts["encoders"][target].inverse_transform([idx])[0]
            response[f"{target}_confidence"] = round(float(proba[idx]), 4)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/analytics")
def analytics():
    df = pd.read_csv(DATA_PATH)
    category = df["category"].dropna().map(lambda value: normalize_label(value, "category"))
    priority = df["priority"].dropna().map(lambda value: normalize_label(value, "priority"))
    resolution_col = "resolution_time_hours" if "resolution_time_hours" in df.columns else "resolution_hours"
    resolution = pd.to_numeric(df.get(resolution_col), errors="coerce")
    department_counts = (
        df["department"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").value_counts().head(10).to_dict()
        if "department" in df.columns
        else {}
    )
    return {
        "total_tickets": int(len(df)),
        "category_counts": category.value_counts().to_dict(),
        "priority_counts": priority.value_counts().to_dict(),
        "department_counts": department_counts,
        "avg_resolution_hours": round(float(resolution.dropna().mean()), 2) if resolution.notna().any() else None,
    }
