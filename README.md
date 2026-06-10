# IT Ticket Automated Classification System

A resume-ready NLP project that classifies enterprise IT helpdesk tickets into:

- **Category** — Network, Software, Hardware, Security, Access, etc.
- **Subcategory** — VPN, Outlook, Laptop, MFA, SAP, etc.
- **Priority** — Low, Medium, High, Critical

The system uses **Sentence Transformers (`all-MiniLM-L6-v2`)** to convert ticket text into semantic embeddings and **XGBoost** classifiers for prediction. It includes a FastAPI backend and a React dashboard.

## Project Architecture

```text
Ticket Text
   ↓
Text Cleaning
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

## 2. Inspect Dataset

```bash
python scripts/inspect_data.py
```

Dataset columns include:

- `ticket_id`
- `created_at`
- `department`
- `user_role`
- `category`
- `subcategory`
- `priority`
- `ticket_text`
- `assigned_team`
- `resolution_hours`

## 3. Train Models

```bash
python train.py
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
- Trained multi-class classification models on 10,000 synthetic enterprise support tickets with realistic departments, urgency levels, resolution metadata, and routing teams.
- Developed a FastAPI inference service and React dashboard for real-time ticket prediction, confidence scoring, and support analytics.
- Created an end-to-end ML pipeline covering data preprocessing, semantic embeddings, model training, evaluation, API deployment, and frontend visualization.

## Suggested GitHub Description

> NLP-powered IT helpdesk ticket classifier using Sentence Transformers, XGBoost, FastAPI, and React for automated ticket triage and priority prediction.
