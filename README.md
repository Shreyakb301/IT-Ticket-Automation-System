# IT Ticket Routing Automation System

A resume-ready NLP project that helps Helpdesk teams route incoming enterprise service-desk tickets to the right support group. The system reads free-text tickets and predicts:

- **Support group** — Account Access and Identity, Network Operations, Endpoint Support, Software and Business Apps, Storage and Collaboration, IT Procurement, HR Systems Support, Security Operations.
- **Issue type** — VPN, Outlook, Laptop, MFA, PostgreSQL, Printer Queue, iPhone, etc.
- **Priority** — Low, Medium, High, Critical

The trained models still learn high-level `category` and detailed `subcategory` labels internally, then the inference layer maps the predicted category to an operational support group. The system includes a FastAPI backend and a React dashboard for confidence-aware routing.

## Project Architecture

```text
Ticket Text
   ↓
Text Cleaning
   ↓
Label Normalization
   ↓
NLP Model Layer
   ↓
Category + Subcategory + Priority + Confidence
   ↓
Support Group Routing Decision
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

For hosting instructions, environment variables, and model artifact deployment steps, see:

```text
docs/deployment.md
```

## 2. Inspect Dataset

```bash
python scripts/inspect_data.py
```

Dataset columns include:

- `ticket_id`
- `created_at`
- `department`
- `employee_role`
- `channel`
- `category`
- `subcategory`
- `true_category`
- `true_subcategory`
- `priority`
- `ticket_text`
- `status`
- `sentiment`
- `resolution_hours`
- `label_quality`

The dataset is intentionally noisy and randomly generated. It includes overlapping issue descriptions, misleading urgency words, typos, vague text, and intentionally noisy category/subcategory labels. The `true_category` and `true_subcategory` fields are included only for evaluation and leakage checks; the training code ignores them by default and trains from `ticket_text` to the visible `category`, `subcategory`, and `priority` labels.

For product usage, `category` is treated as the model's main routing signal and is mapped to the destination Helpdesk support group. Identity and security issue types can override the broad category when needed:

| Model category | Routed support group |
| --- | --- |
| Access | Account Access and Identity |
| Administrative Rights | Account Access and Identity |
| Hardware | Endpoint Support |
| HR Support | HR Systems Support |
| Internal Project, Software | Software and Business Apps |
| Network | Network Operations |
| Purchase | IT Procurement |
| Security | Security Operations |
| Storage | Storage and Collaboration |

| Issue-type override | Routed support group |
| --- | --- |
| Account Lockout, MFA, Password Reset, Permissions | Account Access and Identity |
| Account Compromise, Phishing | Security Operations |
| VPN Access, VPN Connectivity, WiFi, DNS, Ethernet, Slow Internet | Network Operations |
| OneDrive Full, SharePoint Site, Shared Drive Access, Quota Increase | Storage and Collaboration |
| Laptop, Monitor, Docking Station, Keyboard/Mouse, Headset, Webcam | Endpoint Support |
| Laptop Request, Monitor Request, Equipment Procurement, Vendor Quote | IT Procurement |
| Office Apps, Teams, Adobe, ERP, CRM, Browser, Zoom | Software and Business Apps |
| Benefits, Payroll, Onboarding, Offboarding, Timesheet | HR Systems Support |

> After replacing the dataset, retrain the models before using predictions as final results. Existing local model artifacts may still reflect the previous dataset.

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

The final recommendation is to use fine-tuned DistilBERT for support-group routing and issue-type prediction. Priority prediction remains experimental because the ticket text alone does not contain enough reliable urgency signal; in production, priority should combine model output with metadata such as impact, requester role, affected users, SLA, and service criticality.

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
  'support_group': 'Network Operations',
  'support_group_confidence': 0.94,
  'issue_type': 'WiFi',
  'issue_type_confidence': 0.91,
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
  "support_group": "Software Support",
  "support_group_confidence": 0.96,
  "issue_type": "Outlook",
  "issue_type_confidence": 0.94,
  "category": "Software",
  "subcategory": "Outlook",
  "priority": "Medium",
  "category_confidence": 0.96,
  "subcategory_confidence": 0.94,
  "priority_confidence": 0.89
}
```

## 6. Run Frontend Locally

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## 7. Single-Service Docker Deployment

For hosting, this project is configured as one Docker service. FastAPI serves both the React dashboard and the API.

```bash
docker build -t it-ticket-routing .
docker run -p 8000:8000 --env-file .env it-ticket-routing
```

Open:

```text
http://localhost:8000
```

For Render deployment details, see `docs/deployment.md`.

## Dashboard

The project includes a React dashboard with:

- Real-time ticket text input.
- Recommended support group, issue type, and priority predictions.
- Confidence scores for every prediction.
- Auto-route vs human-review routing decision.
- Dataset analytics and distribution charts.

The API marks low-confidence predictions for review:

```text
support_group_confidence >= 0.70 -> auto-route to support group
support_group_confidence < 0.70  -> human review
issue_type_confidence < 0.50     -> issue type is a suggestion only
```

## Resume Bullets

**IT Ticket Routing Automation System**

- Built an NLP-based Helpdesk routing system that recommends the right IT support group, issue type, and priority for incoming support tickets using TF-IDF baselines, Sentence Transformers, XGBoost, and fine-tuned DistilBERT.
- Trained multi-class classification models on 20,000 noisy synthetic enterprise support tickets with overlapping issue descriptions, intentionally noisy labels, urgency metadata, and service desk workflow fields.
- Improved support-group routing accuracy to 81.35% and issue-type accuracy to 76.82% with DistilBERT fine-tuning after benchmarking TF-IDF, MiniLM embeddings, and tuned XGBoost.
- Developed a FastAPI inference service and React dashboard for real-time ticket routing, confidence scoring, and support analytics.
- Created an end-to-end ML pipeline covering data preprocessing, semantic embeddings, model training, evaluation, API deployment, and frontend visualization.

## Suggested GitHub Description

> NLP-powered IT Helpdesk routing system using TF-IDF baselines, Sentence Transformers, XGBoost, DistilBERT fine-tuning, FastAPI, and React to route tickets to the right IT support group.
