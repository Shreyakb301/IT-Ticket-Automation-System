from pathlib import Path
import sys
from inference import load_artifacts, predict_with_artifacts

ROOT = Path(__file__).resolve().parent


def predict(ticket_text: str) -> dict:
    return predict_with_artifacts(load_artifacts(ROOT), ticket_text)


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "My VPN disconnects every few minutes when I work from home"
    print(predict(text))
