import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from preprocessing import normalize_label

DATA_PATH = ROOT / "data" / "tickets.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).head(12))
    print("\nCategory distribution:")
    print(df["category"].dropna().map(lambda value: normalize_label(value, "category")).value_counts())
    print("\nPriority distribution:")
    print(df["priority"].dropna().map(lambda value: normalize_label(value, "priority")).value_counts())
    if "label_quality" in df.columns:
        print("\nLabel quality:")
        print(df["label_quality"].value_counts())
    if {"category", "true_category_hidden"}.issubset(df.columns):
        mismatch = (df["category"] != df["true_category_hidden"]).mean()
        print(f"\nNoisy category label rate: {mismatch:.2%}")
    if {"subcategory", "true_subcategory_hidden"}.issubset(df.columns):
        mismatch = (df["subcategory"] != df["true_subcategory_hidden"]).mean()
        print(f"Noisy subcategory label rate: {mismatch:.2%}")
    print("\nSample ticket:")
    columns = [col for col in ["ticket_text", "category", "subcategory", "priority", "label_quality"] if col in df.columns]
    print(df[columns].sample(1, random_state=42).to_string(index=False))

if __name__ == "__main__":
    main()
