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

For a portfolio-ready case study, demo script, architecture summary, and resume bullet, see:

```text
docs/portfolio_case_study.md
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

For the highest-accuracy synthetic benchmark, train category/subcategory against the dataset's hidden clean labels and use the tuned XGBoost profile:

```bash
TARGET_LABEL_SOURCE=clean XGB_PROFILE=tuned python train.py
```

To compare several model families before choosing the final model:

```bash
BENCHMARK_SAMPLE_SIZE=5000 python scripts/benchmark_models.py
TARGET_LABEL_SOURCE=clean python scripts/benchmark_models.py
```

To run only the fast TF-IDF baselines:

```bash
BENCHMARK_MODE=tfidf python scripts/benchmark_models.py
```

To fine-tune a transformer in Colab, switch to a GPU runtime and train one target at a time:

```bash
TARGET=category TARGET_LABEL_SOURCE=clean EPOCHS=3 BATCH_SIZE=16 python scripts/fine_tune_transformer.py
TARGET=subcategory TARGET_LABEL_SOURCE=clean EPOCHS=3 BATCH_SIZE=16 python scripts/fine_tune_transformer.py
TARGET=priority EPOCHS=3 BATCH_SIZE=16 python scripts/fine_tune_transformer.py
```

Fine-tuning writes model files under:

```text
transformer_models/
reports/finetune_metrics.csv
```

The benchmark writes:

```text
reports/model_benchmark.csv
```

## Final Experiment Results

Best observed results on the 20,000-row noisy/random synthetic dataset:

| Target | Best model | Label source | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | ---: | ---: | ---: |
| Category | Fine-tuned DistilBERT | Clean synthetic target | 0.8135 | 0.8319 | 0.8245 |
| Subcategory | Fine-tuned DistilBERT | Clean synthetic target | 0.7682 | 0.8013 | 0.7977 |
| Priority | Fine-tuned DistilBERT | Visible noisy target | 0.4470 | 0.2310 | 0.3401 |

Model comparison highlights:

| Model | Category accuracy | Subcategory accuracy | Priority accuracy |
| --- | ---: | ---: | ---: |
| TF-IDF + LinearSVC, noisy labels | 0.6897 | 0.5400 | 0.3157 |
| TF-IDF + LinearSVC, clean category/subcategory labels | 0.7987 | 0.7490 | 0.3157 |
| MiniLM + tuned XGBoost, clean category/subcategory labels | 0.6660 | 0.6308 | 0.4173 |
| Fine-tuned DistilBERT | 0.8135 | 0.7682 | 0.4470 |

The final recommendation is to use fine-tuned DistilBERT for category and subcategory prediction. Priority prediction remains experimental because the ticket text alone does not contain enough reliable urgency signal; in production, priority should combine model output with metadata such as impact, requester role, affected users, SLA, and service criticality.

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

## Dashboard

The project includes a React dashboard with:

- Real-time ticket text input.
- Category, subcategory, and priority predictions.
- Confidence scores for every prediction.
- Auto-route vs human-review routing decision.
- Dataset analytics and distribution charts.

The API marks low-confidence predictions for review:

```text
category_confidence >= 0.70 -> auto-route
category_confidence < 0.70  -> human review
subcategory_confidence < 0.50 -> suggestion only
```

## Resume Bullets

**IT Ticket Automated Classification System**

- Built an NLP-based ticket triage system that predicts category, subcategory, and priority for IT helpdesk incidents using TF-IDF baselines, Sentence Transformers, XGBoost, and fine-tuned DistilBERT.
- Trained multi-class classification models on 20,000 noisy synthetic enterprise support tickets with overlapping issue descriptions, intentionally noisy labels, urgency metadata, and service desk workflow fields.
- Improved category accuracy to 81.35% and subcategory accuracy to 76.82% with DistilBERT fine-tuning after benchmarking TF-IDF, MiniLM embeddings, and tuned XGBoost.
- Developed a FastAPI inference service and React dashboard for real-time ticket prediction, confidence scoring, and support analytics.
- Created an end-to-end ML pipeline covering data preprocessing, semantic embeddings, model training, evaluation, API deployment, and frontend visualization.

## Suggested GitHub Description

> NLP-powered IT helpdesk ticket classifier using TF-IDF baselines, Sentence Transformers, XGBoost, DistilBERT fine-tuning, FastAPI, and React for automated ticket triage.
