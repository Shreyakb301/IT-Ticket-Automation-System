"""Train IT ticket classifiers using SentenceTransformer embeddings + XGBoost.

Outputs:
- models/embedding_model_name.txt
- models/label_encoders.joblib
- models/category_model.joblib
- models/subcategory_model.joblib
- models/priority_model.joblib
- reports/metrics.csv
"""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "tickets.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TARGETS = ["category", "subcategory", "priority"]


def clean_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def build_classifier(num_classes: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=250,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    required = ["ticket_text", *TARGETS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()
    df["clean_text"] = df["ticket_text"].map(clean_text)

    print(f"Loaded {len(df):,} tickets")
    print(f"Encoding tickets with {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    X = embedder.encode(
        df["clean_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    label_encoders: dict[str, LabelEncoder] = {}
    metrics: list[dict[str, float | str]] = []

    for target in TARGETS:
        print(f"\nTraining {target} classifier...")
        le = LabelEncoder()
        y = le.fit_transform(df[target].astype(str))
        label_encoders[target] = le

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = build_classifier(num_classes=len(le.classes_))
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")
        weighted_f1 = f1_score(y_test, pred, average="weighted")

        metrics.append({
            "target": target,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "classes": len(le.classes_),
        })

        print(classification_report(y_test, pred, target_names=le.classes_))
        joblib.dump(clf, MODEL_DIR / f"{target}_model.joblib")

    joblib.dump(label_encoders, MODEL_DIR / "label_encoders.joblib")
    (MODEL_DIR / "embedding_model_name.txt").write_text(EMBEDDING_MODEL)
    pd.DataFrame(metrics).to_csv(REPORT_DIR / "metrics.csv", index=False)

    summary = {
        "embedding_model": EMBEDDING_MODEL,
        "targets": TARGETS,
        "tickets": int(len(df)),
        "metrics": metrics,
    }
    (REPORT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved models to models/ and metrics to reports/")


if __name__ == "__main__":
    main()
