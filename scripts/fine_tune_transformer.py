from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import clean_text, normalize_label


DATA_PATH = ROOT / "data" / "tickets.csv"
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "transformer_models"
TARGET = os.getenv("TARGET", "category")
TARGETS = {"category", "subcategory", "priority"}
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
TARGET_LABEL_SOURCE = os.getenv("TARGET_LABEL_SOURCE", "noisy").lower()
TRAIN_SAMPLE_SIZE = os.getenv("TRAIN_SAMPLE_SIZE")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "3"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
RANDOM_STATE = 42


class TicketDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer: AutoTokenizer) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def select_target_series(df: pd.DataFrame, target: str) -> pd.Series:
    if TARGET_LABEL_SOURCE == "clean" and target == "category" and "true_category_hidden" in df.columns:
        return df["true_category_hidden"]
    if TARGET_LABEL_SOURCE == "clean" and target == "subcategory" and "true_subcategory_hidden" in df.columns:
        return df["true_subcategory_hidden"]
    return df[target]


def stratify_or_none(y: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(y, return_counts=True)
    return y if counts.min() >= 2 else None


def load_data() -> tuple[list[str], np.ndarray, LabelEncoder]:
    if TARGET not in TARGETS:
        raise ValueError(f"TARGET must be one of {sorted(TARGETS)}")

    df = pd.read_csv(DATA_PATH).dropna(subset=["ticket_text", TARGET]).copy()
    df["clean_text"] = df["ticket_text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()
    df[TARGET] = select_target_series(df, TARGET).map(lambda value: normalize_label(value, TARGET))

    if TRAIN_SAMPLE_SIZE:
        df = df.sample(min(int(TRAIN_SAMPLE_SIZE), len(df)), random_state=RANDOM_STATE).copy()

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df[TARGET].astype(str))
    return df["clean_text"].tolist(), labels, encoder


def evaluate(model: AutoModelForSequenceClassification, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            preds.append(torch.argmax(output.logits, dim=-1).cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(labels), np.concatenate(preds)


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    texts, labels, encoder = load_data()
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_or_none(labels),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TicketDataset(train_texts, y_train, tokenizer)
    test_ds = TicketDataset(test_texts, y_test, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(encoder.classes_),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Fine-tuning {MODEL_NAME} for target={TARGET}")
    print(f"Rows: {len(texts):,}; classes: {len(encoder.classes_)}; target_label_source={TARGET_LABEL_SOURCE}")
    print(f"Device: {device}; epochs={EPOCHS}; batch_size={BATCH_SIZE}; max_length={MAX_LENGTH}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.item())
            progress.set_postfix(loss=round(total_loss / max(1, progress.n), 4))

    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = {
        "model": MODEL_NAME,
        "target": TARGET,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "classes": len(encoder.classes_),
        "target_label_source": TARGET_LABEL_SOURCE,
        "train_sample_size": int(TRAIN_SAMPLE_SIZE) if TRAIN_SAMPLE_SIZE else None,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
    }
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))
    print(metrics)

    target_dir = OUTPUT_DIR / TARGET
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    joblib.dump(encoder, target_dir / "label_encoder.joblib")
    (target_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2))

    metrics_path = REPORT_DIR / "finetune_metrics.csv"
    existing = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([metrics])], ignore_index=True).to_csv(metrics_path, index=False)
    print(f"Saved model to {target_dir}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
