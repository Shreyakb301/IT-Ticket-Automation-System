# IT Ticket Automated Classification System

A resume-ready NLP project that classifies enterprise IT helpdesk tickets into:

- **Category** — Network, Software, Hardware, Security, Access, Database, Printer, Mobile Device, etc.
- **Subcategory** — VPN, Outlook, Laptop, MFA, PostgreSQL, Printer Queue, iPhone, etc.
- **Priority** — Low, Medium, High, Critical

The system uses **Sentence Transformers (`all-MiniLM-L6-v2`)** to convert cleaned ticket text into semantic embeddings and **XGBoost** classifiers for prediction. It includes a FastAPI backend and a React dashboard.

## Project Architecture

```text
Ticket Text
   ↓
Text Cleaning
   ↓
Label Normalization
   ↓
Sentence Transformer Embedding, 384 dimensions
   ↓
XGBoost Models
   ↓
Category + Subcategory + Priority + Confidence
   ↓
FastAPI + React Dashboard
```

## Folder Structure

```text
it-ticket-automated-classifier/
├── data/tickets.csv
├── models/
├── reports/
├── backend/app.py
├── frontend/
├── scripts/inspect_data.py
├── train.py
├── predict_local.py
├── requirements.txt
└── README.md
```

## 1. Setup Python Environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Research Journey

For the full problem analysis, literature review, model comparison, architecture, business impact, and final recommendation, see:

```text
docs/research_journey.md
```

## 2. Inspect Dataset

```bash
python scripts/inspect_data.py
```

Dataset columns include:

- `ticket_id`
- `created_at`
- `department`
- `user_role`
- `channel`
- `category`
- `subcategory`
- `true_category_hidden`
- `true_subcategory_hidden`
- `priority`
- `ticket_text`
- `status`
- `impact`
- `response_time_hours`
- `resolution_time_hours`
- `label_quality`

The dataset is intentionally noisy and randomly generated. It includes overlapping issue descriptions, misleading urgency words, typos, vague text, and intentionally noisy category/subcategory labels. The `true_category_hidden` and `true_subcategory_hidden` fields are included only for evaluation and leakage checks; the training code ignores them and trains from `ticket_text` to the visible `category`, `subcategory`, and `priority` labels.

## 3. Train Models

```bash
python train.py
```

For a quick smoke test before full training, run:

```bash
TRAIN_SAMPLE_SIZE=5000 python train.py
```

This creates:

```text
models/category_model.joblib
models/subcategory_model.joblib
models/priority_model.joblib
models/label_encoders.joblib
models/embedding_model_name.txt
reports/metrics.csv
reports/training_summary.json
```

## 4. Test Local Prediction

```bash
python predict_local.py "My laptop cannot connect to the office WiFi"
```

Example output:

```python
{
  'ticket_text': 'My laptop cannot connect to the office WiFi',
  'category': 'Network',
  'category_confidence': 0.94,
  'subcategory': 'WiFi',
  'subcategory_confidence': 0.91,
  'priority': 'High',
  'priority_confidence': 0.87
}
```

## 5. Run FastAPI Backend

```bash
uvicorn backend.app:app --reload --port 8000
```

Open API docs:

```text
http://localhost:8000/docs
```

### Prediction Endpoint

```http
POST /predict
```

Request:

```json
{
  "ticket_text": "Outlook crashes when opening large attachments"
}
```

Response:

```json
{
  "ticket_text": "Outlook crashes when opening large attachments",
  "category": "Software",
  "subcategory": "Outlook",
  "priority": "Medium",
  "category_confidence": 0.96,
  "subcategory_confidence": 0.94,
  "priority_confidence": 0.89
}
```

## 6. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Resume Bullets

**IT Ticket Automated Classification System**

- Built an NLP-based ticket triage system that predicts category, subcategory, and priority for IT helpdesk incidents using Sentence Transformers and XGBoost.
- Trained multi-class classification models on 20,000 noisy synthetic enterprise support tickets with overlapping issue descriptions, intentionally noisy labels, urgency metadata, and service desk workflow fields.
- Developed a FastAPI inference service and React dashboard for real-time ticket prediction, confidence scoring, and support analytics.
- Created an end-to-end ML pipeline covering data preprocessing, semantic embeddings, model training, evaluation, API deployment, and frontend visualization.

## Suggested GitHub Description

> NLP-powered IT helpdesk ticket classifier using Sentence Transformers, XGBoost, FastAPI, and React for automated ticket triage and priority prediction.
