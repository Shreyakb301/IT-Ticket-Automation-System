setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

inspect:
	python scripts/inspect_data.py

train:
	python train.py

api:
	uvicorn backend.app:app --reload --port 8000
