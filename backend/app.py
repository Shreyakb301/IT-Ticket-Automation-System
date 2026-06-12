from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from inference import load_artifacts as load_model_artifacts, predict_with_artifacts
from preprocessing import normalize_label

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tickets.csv"

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
    model_config = ConfigDict(protected_namespaces=())

    ticket_text: str
    category: str
    subcategory: str
    priority: str
    category_confidence: float
    subcategory_confidence: float
    priority_confidence: float
    auto_route: bool
    needs_human_review: bool
    routing_decision: str
    review_reason: str | None = None
    model_type: str

_artifacts = {}


def load_artifacts():
    if _artifacts:
        return _artifacts
    _artifacts.update(load_model_artifacts(ROOT))
    return _artifacts

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(payload: TicketRequest):
    try:
        artifacts = load_artifacts()
        return predict_with_artifacts(artifacts, payload.ticket_text)
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
