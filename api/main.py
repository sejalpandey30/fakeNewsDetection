"""
Real-time API.

Endpoints:
  POST /predict            -> analyze pasted text or a URL, sub-second response
  POST /feedback            -> log human feedback on a verdict (continuous-learning loop)
  GET  /health               -> liveness check + which models are loaded
  GET  /stats                -> aggregate stats since server start
  POST /monitor/start         -> start background monitoring of an RSS feed
  POST /monitor/stop          -> stop monitoring
  WS   /ws/alerts             -> live stream of flagged articles as they're found

Run from the project root:
  uvicorn api.main:app --reload --port 8000
"""
import asyncio
import csv
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, List

import feedparser
import joblib
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models.classical_ml import ClassicalEnsemble  # noqa: F401 (needed for joblib unpickling)
from src.models.ensemble import MetaEnsemble  # noqa: F401
from src.models.deep_learning import DeepTextClassifier
from src.explainability import Explainer
from src.pipeline import FakeNewsDetector
from utils.scraper import extract_article

MODELS_DIR = "models"
FEEDBACK_LOG = "data/feedback_log.csv"

state = {"detector": None, "monitor_task": None, "connections": [], "stats": {"total": 0, "fake": 0, "real": 0}}


def load_detector() -> FakeNewsDetector:
    classical_model = joblib.load(f"{MODELS_DIR}/classical_ensemble.joblib")
    meta_model = joblib.load(f"{MODELS_DIR}/meta_ensemble.joblib")
    bundle = torch.load(f"{MODELS_DIR}/bilstm.pt", map_location="cpu", weights_only=False)
    deep_model = DeepTextClassifier.load(bundle)
    explainer = Explainer(classical_model)
    return FakeNewsDetector(classical_model, deep_model, meta_model, explainer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["detector"] = load_detector()
    print("Models loaded. API ready.")
    yield


app = FastAPI(title="Advanced Fake News Detection API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = ""
    explain: bool = False
    use_fact_check: bool = True


class FeedbackRequest(BaseModel):
    text: str
    predicted_verdict: str
    correct_verdict: str  # "FAKE" or "REAL", as judged by the human reviewer
    notes: Optional[str] = ""


class MonitorRequest(BaseModel):
    feed_url: str
    poll_seconds: int = 60
    fake_threshold: float = 0.7


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": state["detector"] is not None, "stats": state["stats"]}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    if not req.text and not req.url:
        return {"error": "Provide either 'text' or 'url'."}

    text, source = req.text, req.source
    article_meta = None
    if req.url:
        article_meta = extract_article(req.url)
        text = article_meta["text"]
        source = source or article_meta["domain"]

    result = state["detector"].predict(text, source=source, use_fact_check=req.use_fact_check, explain=req.explain)
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    if article_meta:
        result["article"] = {"title": article_meta["title"], "url": article_meta["url"]}

    state["stats"]["total"] += 1
    state["stats"]["fake" if result["verdict"] == "FAKE" else "real"] += 1
    return result


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """
    Continuous-learning hook: human reviewers (or downstream user reports)
    submit corrections here. In production this feeds a periodic retraining
    job; for this prototype it's appended to a CSV that src/train.py can be
    pointed at to incorporate corrected examples into the next training run.
    """
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    file_exists = os.path.isfile(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["text", "predicted_verdict", "correct_verdict", "notes", "timestamp"])
        w.writerow([req.text, req.predicted_verdict, req.correct_verdict, req.notes, time.time()])
    return {"status": "logged", "message": "Thanks — this will be included in the next retraining pass."}


@app.get("/stats")
def stats():
    return state["stats"]


@app.get("/models/weights")
def get_weights():
    if state["detector"] is None:
        return {"error": "Detector not loaded"}
    return {
        "signal_weights": state["detector"].meta_model.explain_weights(),
        "description": "Learned meta-classifier stacking weights for each of the 5 detection methods",
    }


@app.get("/models/evaluation")
def get_eval():
    eval_path = f"{MODELS_DIR}/eval_results.joblib"
    if os.path.exists(eval_path):
        try:
            return joblib.load(eval_path)
        except Exception as e:
            return {"error": f"Failed to load evaluation metrics: {e}"}
    return {"error": "Evaluation results not yet generated"}


# ---------------- Real-time RSS feed monitoring ----------------

async def _broadcast(message: dict):
    dead = []
    for ws in state["connections"]:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state["connections"].remove(ws)


async def _monitor_loop(feed_url: str, poll_seconds: int, fake_threshold: float):
    seen_links = set()
    await _broadcast({"type": "status", "message": f"Started monitoring {feed_url}"})
    while True:
        try:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries:
                link = entry.get("link")
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                text = entry.get("summary", "") or entry.get("title", "")
                if not text:
                    continue
                result = state["detector"].predict(text, source=link)
                if result["fake_probability"] >= fake_threshold:
                    await _broadcast({
                        "type": "alert",
                        "title": entry.get("title"),
                        "link": link,
                        "fake_probability": result["fake_probability"],
                        "verdict": result["verdict"],
                    })
                else:
                    await _broadcast({
                        "type": "checked",
                        "title": entry.get("title"),
                        "fake_probability": result["fake_probability"],
                    })
        except Exception as e:
            await _broadcast({"type": "error", "message": str(e)})
        await asyncio.sleep(poll_seconds)


@app.post("/monitor/start")
async def monitor_start(req: MonitorRequest):
    if state["monitor_task"] and not state["monitor_task"].done():
        return {"status": "already_running"}
    state["monitor_task"] = asyncio.create_task(
        _monitor_loop(req.feed_url, req.poll_seconds, req.fake_threshold)
    )
    return {"status": "started", "feed_url": req.feed_url}


@app.post("/monitor/stop")
async def monitor_stop():
    if state["monitor_task"]:
        state["monitor_task"].cancel()
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    state["connections"].append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive / ignore client pings
    except WebSocketDisconnect:
        if websocket in state["connections"]:
            state["connections"].remove(websocket)
