from pathlib import Path
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "src" / "models" / "sentiment"

model     = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded successfully.")
    yield
    model     = None
    tokenizer = None

app = FastAPI(
    title       = "Sentiment Analysis API",
    description = "DistilBERT fine-tuned on SST-2",
    version     = "1.0.0",
    lifespan    = lifespan
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float
    text:  str

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    tokens = tokenizer(
        request.text,
        return_tensors  = "pt",
        truncation      = True,
        max_length      = 128,
        padding         = True,
    )

    with torch.no_grad():
        logits = model(**tokens).logits

    probs      = torch.softmax(logits, dim=-1)
    pred_idx   = torch.argmax(probs).item()
    score      = probs[0][pred_idx].item()
    label      = "POSITIVE" if pred_idx == 1 else "NEGATIVE"

    return PredictResponse(label=label, score=round(score, 4), text=request.text)