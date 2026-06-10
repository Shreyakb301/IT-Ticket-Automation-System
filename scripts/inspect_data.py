import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tickets.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    print("\nPriority distribution:")
    print(df["priority"].value_counts())
    print("\nSample ticket:")
    print(df[["ticket_text", "category", "subcategory", "priority", "assigned_team"]].sample(1).to_string(index=False))

if __name__ == "__main__":
    main()
