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
import os
import json
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from preprocessing import clean_text, normalize_label

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "tickets.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TARGETS = ["category", "subcategory", "priority"]
TRAIN_SAMPLE_SIZE = os.getenv("TRAIN_SAMPLE_SIZE")
TARGET_LABEL_SOURCE = os.getenv("TARGET_LABEL_SOURCE", "noisy").lower()
XGB_PROFILE = os.getenv("XGB_PROFILE", "fast").lower()
CLASS_BALANCE = os.getenv("CLASS_BALANCE", "0") == "1"


def build_classifier(num_classes: int) -> XGBClassifier:
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    if XGB_PROFILE == "tuned":
        params.update(
            n_estimators=650,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
        )
    else:
        params.update(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
        )
    return XGBClassifier(**params)


def select_target_series(df: pd.DataFrame, target: str) -> pd.Series:
    if TARGET_LABEL_SOURCE == "clean" and target == "category" and "true_category_hidden" in df.columns:
        return df["true_category_hidden"]
    if TARGET_LABEL_SOURCE == "clean" and target == "subcategory" and "true_subcategory_hidden" in df.columns:
        return df["true_subcategory_hidden"]
    return df[target]


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
    df = df[df["clean_text"].str.len() > 0].copy()

    for target in TARGETS:
        df[target] = select_target_series(df, target).map(lambda value: normalize_label(value, target))

    if TRAIN_SAMPLE_SIZE:
        sample_size = min(int(TRAIN_SAMPLE_SIZE), len(df))
        df = df.sample(sample_size, random_state=42).copy()

    print(f"Loaded {len(df):,} tickets")
    if {"true_category_hidden", "true_subcategory_hidden"} & set(df.columns):
        print(f"Target label source: {TARGET_LABEL_SOURCE}")
        if TARGET_LABEL_SOURCE == "clean":
            print("Using hidden clean labels as targets for category/subcategory; ticket_text remains the only model input.")
        else:
            print("Detected hidden clean-label columns; training uses only ticket_text and visible target labels.")
    print(f"XGBoost profile: {XGB_PROFILE}")
    print(f"Class-balanced sample weights: {CLASS_BALANCE}")
    if "label_quality" in df.columns:
        print("Label quality distribution:")
        print(df["label_quality"].value_counts().to_string())
    print("Target classes:")
    for target in TARGETS:
        print(f"- {target}: {df[target].nunique()} classes")
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
        sample_weight = compute_sample_weight("balanced", y_train) if CLASS_BALANCE else None
        clf.fit(X_train, y_train, sample_weight=sample_weight)
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
        "train_sample_size": int(len(df)) if TRAIN_SAMPLE_SIZE else None,
        "target_label_source": TARGET_LABEL_SOURCE,
        "xgb_profile": XGB_PROFILE,
        "class_balance": CLASS_BALANCE,
        "preprocessing": "clean_text + target label normalization",
        "metrics": metrics,
    }
    (REPORT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved models to models/ and metrics to reports/")


if __name__ == "__main__":
    main()
