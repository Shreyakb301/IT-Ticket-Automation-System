from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from preprocessing import clean_text, normalize_label


DATA_PATH = ROOT / "data" / "tickets.csv"
REPORT_DIR = ROOT / "reports"
TARGETS = ["category", "subcategory", "priority"]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TARGET_LABEL_SOURCE = os.getenv("TARGET_LABEL_SOURCE", "noisy").lower()
BENCHMARK_SAMPLE_SIZE = os.getenv("BENCHMARK_SAMPLE_SIZE")
BENCHMARK_MODE = os.getenv("BENCHMARK_MODE", "all").lower()
RANDOM_STATE = 42


def select_target_series(df: pd.DataFrame, target: str) -> pd.Series:
    if TARGET_LABEL_SOURCE == "clean" and target == "category" and "true_category_hidden" in df.columns:
        return df["true_category_hidden"]
    if TARGET_LABEL_SOURCE == "clean" and target == "subcategory" and "true_subcategory_hidden" in df.columns:
        return df["true_subcategory_hidden"]
    return df[target]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).dropna(subset=["ticket_text", *TARGETS]).copy()
    df["clean_text"] = df["ticket_text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()
    for target in TARGETS:
        df[target] = select_target_series(df, target).map(lambda value: normalize_label(value, target))
    if BENCHMARK_SAMPLE_SIZE:
        df = df.sample(min(int(BENCHMARK_SAMPLE_SIZE), len(df)), random_state=RANDOM_STATE).copy()
    return df


def metrics_row(model_name: str, target: str, y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> dict:
    return {
        "model": model_name,
        "target": target,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "classes": classes,
        "target_label_source": TARGET_LABEL_SOURCE,
        "sample_size": int(BENCHMARK_SAMPLE_SIZE) if BENCHMARK_SAMPLE_SIZE else None,
    }


def stratify_or_none(y: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(y, return_counts=True)
    return y if counts.min() >= 2 else None


def make_tfidf_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def benchmark_tfidf_models(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for target in TARGETS:
        le = LabelEncoder()
        y = le.fit_transform(df[target].astype(str))
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["clean_text"],
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=stratify_or_none(y),
        )

        features = make_tfidf_features()
        X_train = features.fit_transform(X_train_text)
        X_test = features.transform(X_test_text)

        logreg = LogisticRegression(
            C=4.0,
            max_iter=3000,
            class_weight="balanced" if target == "priority" else None,
            n_jobs=1,
        )
        logreg.fit(X_train, y_train)
        rows.append(metrics_row("tfidf_word_char_logreg", target, y_test, logreg.predict(X_test), len(le.classes_)))

        svc = LinearSVC(C=1.5, class_weight="balanced" if target == "priority" else None)
        svc.fit(X_train, y_train)
        rows.append(metrics_row("tfidf_word_char_linearsvc", target, y_test, svc.predict(X_test), len(le.classes_)))
    return rows


def benchmark_embedding_models(df: pd.DataFrame) -> list[dict]:
    print(f"Encoding {len(df):,} tickets with {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    X = embedder.encode(
        df["clean_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    rows: list[dict] = []
    for target in TARGETS:
        le = LabelEncoder()
        y = le.fit_transform(df[target].astype(str))
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=stratify_or_none(y),
        )

        logreg = LogisticRegression(
            C=8.0,
            max_iter=3000,
            class_weight="balanced" if target == "priority" else None,
            n_jobs=1,
        )
        logreg.fit(X_train, y_train)
        rows.append(metrics_row("minilm_logreg", target, y_test, logreg.predict(X_test), len(le.classes_)))

        xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=len(le.classes_),
            n_estimators=450,
            max_depth=4,
            learning_rate=0.045,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        rows.append(metrics_row("minilm_xgboost_tuned", target, y_test, xgb.predict(X_test), len(le.classes_)))
    return rows


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df):,} rows")
    print(f"Target label source: {TARGET_LABEL_SOURCE}")
    print(f"Benchmark mode: {BENCHMARK_MODE}")

    rows = []
    if BENCHMARK_MODE in {"all", "tfidf"}:
        rows.extend(benchmark_tfidf_models(df))
    if BENCHMARK_MODE in {"all", "embedding", "embeddings"}:
        rows.extend(benchmark_embedding_models(df))
    if not rows:
        raise ValueError("BENCHMARK_MODE must be one of: all, tfidf, embedding")

    results = pd.DataFrame(rows).sort_values(["target", "weighted_f1", "accuracy"], ascending=[True, False, False])
    output_path = REPORT_DIR / "model_benchmark.csv"
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
