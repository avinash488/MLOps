# Sentiment Analysis MLOps Pipeline

![CI Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/ci.yml/badge.svg)

An end-to-end MLOps project that fine-tunes a DistilBERT transformer on the SST-2 sentiment dataset, tracks experiments with MLflow, serves predictions via a REST API, and validates the full pipeline through a CI/CD workflow on every push.

---

## Results

| Metric | Value |
|--------|-------|
| Model | DistilBERT-base-uncased |
| Dataset | SST-2 (67,349 train / 872 val) |
| Val Accuracy | 90.48% |
| Epochs | 3 |
| Training Time | ~35 min (CPU) |

---

## Architecture

```
SST-2 Dataset
     │
     ▼
data_prep.py ──► Tokenized Dataset (HuggingFace)
     │
     ▼
train.py ──► Fine-tuned DistilBERT ──► MLflow Experiment Tracking
     │
     ▼
src/models/sentiment/
     │
     ▼
api/app.py ──► FastAPI REST API ──► /predict endpoint
     │
     ▼
Dockerfile ──► Docker Image
     │
     ▼
GitHub Actions ──► CI/CD (test + build on every push)
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Model | DistilBERT (HuggingFace Transformers) |
| Training | PyTorch + HuggingFace Trainer API |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | Pytest |

---

## Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**3. Prepare the data**
```bash
python src/data_prep.py
```

**4. Train the model**
```bash
python src/train.py
```

**5. View experiment results**
```bash
mlflow ui
# Open http://127.0.0.1:5000
```

**6. Run the API locally**
```bash
uvicorn api.app:app --reload --port 8000
# Open http://127.0.0.1:8000/docs
```

**7. Run with Docker**
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

---

## API Usage

**Predict sentiment:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

**Response:**
```json
{
  "label": "POSITIVE",
  "score": 0.9876,
  "text": "This movie was absolutely fantastic!"
}
```

**Health check:**
```bash
curl http://127.0.0.1:8000/health
```

---

## Project Structure

```
├── .github/workflows/   # CI/CD pipeline
├── api/
│   ├── app.py           # FastAPI server
│   └── requirements-api.txt
├── src/
│   ├── data_prep.py     # Dataset download and tokenization
│   └── train.py         # Model fine-tuning with MLflow tracking
├── tests/
│   └── test_api.py      # API tests
├── Dockerfile
└── requirements.txt
```

---

## Key Learnings

- Fine-tuning a pretrained transformer for sequence classification
- Tracking hyperparameters and metrics with MLflow experiment registry
- Building a production-style inference API with FastAPI
- Containerizing an ML service with Docker
- Automating testing and builds with GitHub Actions CI/CD