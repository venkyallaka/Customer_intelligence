from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import requests
import urllib3
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = PROJECT_ROOT / "artifacts" / "intelligence_models.joblib"
FRONTEND_INDEX = PROJECT_ROOT / "frontend" / "index.html"
DEFAULT_HF_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "openai/gpt-oss-120b:fastest",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

app = FastAPI(title="Automobile Intelligence Chatbot")


class ChatRequest(BaseModel):
    question: str
    customer_id: str | None = None
    use_case: str | None = None
    history: List[Dict[str, str]] = Field(default_factory=list)


class PredictRequest(BaseModel):
    record: Dict[str, Any] | None = None
    customer_id: str | None = None


@lru_cache(maxsize=1)
def load_bundle():
    if not ARTIFACT.exists():
        raise RuntimeError("Model artifact missing. Run train_models.py first.")
    return joblib.load(ARTIFACT)


@app.on_event("startup")
def preload_models() -> None:
    """Finish loading the trained bundle before the chat accepts its first request."""
    load_bundle()


USE_CASES = {
    "purchase_sales": {
        "label": "Customer Purchase & Sales Intelligence",
        "target": "future purchase or sales opportunity",
        "signals": "vehicle age and mileage, purchase recency, equity, incentives, and transaction activity",
    },
    "retention_service": {
        "label": "Customer Retention & Service Intelligence",
        "target": "retention and service opportunity",
        "signals": "defection alerts, ownership, and recency of service or transaction activity",
    },
    "value_financial": {
        "label": "Customer Value & Financial Intelligence",
        "target": "high customer value and financial opportunity",
        "signals": "sale and repair-order value, finance amount, payment behaviour, and trade-in value",
    },
    "engagement_marketing": {
        "label": "Customer Engagement & Marketing Intelligence",
        "target": "marketing engagement opportunity",
        "signals": "email, SMS and phone consent, verified contact details, and customer communication preferences",
    },
}


def engagement_score(record: Dict[str, Any]) -> float:
    """Compatible score for existing three-model artifacts until the bundle is retrained."""
    opted_in = ["email_optin", "sms_optin", "is_verified_email", "is_verified_address", "dms_cell_linkage_score"]
    values = [pd.to_numeric(record.get(field), errors="coerce") for field in opted_in]
    usable = [float(value) for value in values if pd.notna(value)]
    return round(sum(min(max(value, 0.0), 1.0) for value in usable) / len(usable), 4) if usable else 0.0


def predict_record(bundle: Dict, record: Dict[str, Any]) -> Dict[str, Any]:
    row = pd.DataFrame([record])
    scores = {}
    for name, model in bundle["models"].items():
        try:
            prob = float(model.predict_proba(row)[0, 1])
        except Exception:
            prob = float(model.predict(row)[0])
        scores[name] = round(prob, 4)
    # Older saved bundles have three estimators. Keep the new engagement use case
    # usable immediately; retraining replaces this with its learned estimator.
    if "engagement_marketing" not in scores:
        scores["engagement_marketing"] = engagement_score(record)
    return scores


def score_records(bundle: Dict, records: List[Dict[str, Any]], use_case: str) -> List[tuple[Dict[str, Any], float]]:
    """Score a customer group with the selected use case only."""
    if not records:
        return []
    if use_case == "engagement_marketing" and use_case not in bundle["models"]:
        return [(record, engagement_score(record)) for record in records]
    model = bundle["models"].get(use_case)
    if model is None:
        return []
    rows = pd.DataFrame(records)
    try:
        values = model.predict_proba(rows)[:, 1]
    except Exception:
        values = model.predict(rows)
    return [(record, float(value)) for record, value in zip(records, values)]


def find_record(bundle: Dict, customer_id: str | None) -> Dict[str, Any]:
    records = bundle.get("sample_records", [])
    if customer_id:
        for rec in records:
            if str(rec.get("id")) == str(customer_id):
                return rec
        return {}
    return records[0] if records else {}


def customer_id_from_question(question: str) -> str | None:
    """Allow a natural chat request such as 'What about customer 9521?'"""
    match = re.search(r"\bcustomer(?:\s+id)?\s*#?\s*([a-zA-Z0-9-]+)", question, flags=re.IGNORECASE)
    return match.group(1) if match else None


def asks_for_suggestions(question: str) -> bool:
    return bool(re.search(r"\b(suggest|suggestion|recommend|improve|increase|how can|what should|next step)\b", question, flags=re.IGNORECASE))


def choose_use_case(question: str) -> str:
    """Choose the relevant intelligence model from the owner's natural-language question."""
    text = question.lower()
    if re.search(r"\b(retention|retain|churn|service|repair|warranty|loyalty|defection)\b", text):
        return "retention_service"
    if re.search(r"\b(finance|financial|revenue|value|valuable|payment|apr|default|profit|clv)\b", text):
        return "value_financial"
    if re.search(r"\b(marketing|campaign|engagement|email|sms|communication|outreach|channel)\b", text):
        return "engagement_marketing"
    return "purchase_sales"


def requested_top_count(question: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", question, flags=re.IGNORECASE)
    return min(int(match.group(1)), 25) if match else None


def requested_period(question: str) -> str:
    match = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", question, flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else "the next planning period"


def retrieve_context(bundle: Dict, question: str, records: List[Dict[str, Any]], use_case: str, scores: Dict[str, float] | None = None) -> str:
    """Structured facts for the business-analyst response layer."""
    frame = pd.DataFrame(records)
    metrics = bundle.get("metrics", {}).get("targets", {}).get(use_case, {})
    lines = [
        f"Analysis focus: {USE_CASES[use_case]['label']}",
        f"Business objective: {USE_CASES[use_case]['target']}",
        f"Available organization profiles: {bundle.get('metrics', {}).get('rows_used', len(frame))}",
    ]
    if metrics.get("positive_rate") is not None:
        lines.append(f"Population opportunity rate: {float(metrics['positive_rate']) * 100:.1f}%")
    if scores and use_case in scores:
        lines.append(f"Individual opportunity assessment: {scores[use_case]:.4f}")
    if len(records) == 1:
        record = records[0]
        profile = [
            f"Customer name: {record.get('full_name')}" if record.get("full_name") else None,
            f"Customer ID: {record.get('id')}" if record.get("id") is not None else None,
            "Vehicle: " + " ".join(str(record.get(field)) for field in ("vehicle_year", "vehicle_make", "vehicle_model") if record.get(field) not in (None, "")),
        ]
        lines.extend(item for item in profile if item and not item.endswith("Vehicle: "))
    for field, label in (
        ("vehicle_mileage", "Typical vehicle mileage"),
        ("days_since_sale", "Typical time since sale"),
        ("days_since_last_transaction", "Typical time since last transaction"),
        ("sale_amount", "Typical sale value"),
        ("total_ro_amount", "Typical service value"),
    ):
        if field in frame:
            values = pd.to_numeric(frame[field], errors="coerce").dropna()
            if not values.empty:
                lines.append(f"{label}: {values.median():.0f}")
    lines.append("User question: " + question)
    return "\n".join(lines)


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "nan", "null", "na", "n/a", "unknown", "unavailable"}:
        return ""
    return text


def get_llm_api_key() -> str | None:
    token = os.getenv("LLM_API_KEY") or os.getenv("HF_TOKEN")
    token_file = PROJECT_ROOT / ".hf_token"
    if not token and token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
    return token


def clean_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned = []
    for item in history[-8:]:
        role = item.get("role")
        content = clean_text(item.get("content") or item.get("text"))
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:1200]})
    return cleaned


def hf_polish(question: str, context: str, scores: Dict[str, Any], use_case: str, history: List[Dict[str, str]] | None = None) -> str | None:
    token = get_llm_api_key()
    if not token:
        return None
    system_prompt = (
        "You are an AI Customer Intelligence Assistant for an enterprise CRM platform. "
        "Answer only the user's question using the structured context. "
        "Use the recent conversation to understand follow-up questions and references like 'that customer', 'second one', or 'why'. "
        "Keep the answer focused: 1 short paragraph or 3-5 bullets maximum. "
        "Do not add generic sections, long introductions, implementation details, APIs, model names, raw scores, probabilities, or dataset talk. "
        "Do not show placeholder or garbage values such as None, null, nan, unavailable, unknown, empty fields, or zero scores. "
        "Use customer names, IDs, counts, and business actions only when they are present and useful. "
        "If the context and recent conversation do not support the question, say exactly what is missing and stop."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(clean_history(history or []))
    messages.append({
        "role": "user",
        "content": f"Current question: {question}\n\nStructured intelligence context:\n{context}",
    })
    verify_tls: bool | str = True
    try:
        import certifi

        ca_path = certifi.where()
        verify_tls = ca_path if Path(ca_path).exists() else False
    except Exception:
        verify_tls = False
    if verify_tls is False:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    model = os.getenv("LLM_MODEL") or os.getenv("HF_MODEL") or DEFAULT_HF_MODELS[0]
    response = requests.post(
        os.getenv("LLM_API_URL", "https://router.huggingface.co/v1/chat/completions"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": 350, "stream": False},
        verify=verify_tls,
        timeout=45,
    )
    response.raise_for_status()
    message = response.json()["choices"][0].get("message", {})
    content = message.get("content") or message.get("reasoning") or str(message)
    return content.strip()


def llm_answer(question: str, context: str, scores: Dict[str, Any], use_case: str, history: List[Dict[str, str]] | None = None) -> str:
    if not get_llm_api_key():
        raise HTTPException(
            status_code=503,
            detail="LLM API key missing. Set HF_TOKEN or LLM_API_KEY, or create a .hf_token file in the project root.",
        )
    try:
        answer = hf_polish(question, context, scores, use_case, history)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
    if not answer:
        raise HTTPException(status_code=502, detail="LLM returned an empty response.")
    return answer


def build_llm_context(bundle: Dict, question: str, use_case: str, requested_id: str | None, top_count: int | None) -> tuple[str, Dict[str, Any]]:
    records = [find_record(bundle, requested_id)] if requested_id else bundle.get("sample_records", [])
    records = [record for record in records if record]
    scores = predict_record(bundle, records[0]) if requested_id and records else {}
    context = retrieve_context(bundle, question, records, use_case, scores)
    extra_lines = []
    text = question.lower()

    if "most valuable" in text:
        use_case = "value_financial"
        top_count = top_count or 1

    if top_count:
        saved_records = bundle.get("ranked_records", {}).get(use_case)
        if saved_records:
            ranked = [(record, 0.0) for record in saved_records]
        else:
            ranked = score_records(bundle, bundle.get("sample_records", []), use_case)
            ranked.sort(key=lambda item: item[1], reverse=True)
        extra_lines.append(f"Requested ranked customer count: {min(top_count, len(ranked))}")
        for index, (record, _) in enumerate(ranked[:top_count], start=1):
            vehicle_parts = [clean_text(record.get(field)) for field in ("vehicle_year", "vehicle_make", "vehicle_model")]
            vehicle = " ".join(part for part in vehicle_parts if part)
            details = [
                f"Rank {index}",
                f"customer_id={clean_text(record.get('id'))}",
                f"name={clean_text(record.get('full_name'))}",
            ]
            if vehicle:
                details.append(f"vehicle={vehicle}")
            extra_lines.append(", ".join(item for item in details if not item.endswith("=")))

    if re.search(r"\b(count|number|how many)\b", text) and re.search(r"\b(purchase|buy|sales opportunity)\b", text):
        saved_count = bundle.get("population_prediction_counts", {}).get("purchase_sales")
        population_size = bundle.get("metrics", {}).get("rows_used")
        scored = score_records(bundle, bundle.get("sample_records", []), "purchase_sales")
        available_count = sum(score >= 0.5 for _, score in scored)
        extra_lines.append(f"Requested forecast period: {requested_period(question)}")
        extra_lines.append(f"Saved purchase-ready count: {saved_count}")
        extra_lines.append(f"Population size: {population_size}")
        extra_lines.append(f"Available sampled purchase-ready count: {available_count}")

    if extra_lines:
        context += "\n" + "\n".join(extra_lines)
    return context, scores


@app.get("/", response_class=HTMLResponse)
def home():
    if FRONTEND_INDEX.exists():
        return HTMLResponse(
            FRONTEND_INDEX.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
        )
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Automobile Intelligence Chatbot</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
    body { margin: 0; background: #f5f7fb; color: #172033; }
    main { max-width: 1120px; margin: 0 auto; padding: 28px; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 22px; }
    h1 { font-size: 28px; margin: 0; letter-spacing: 0; }
    .grid { display: grid; grid-template-columns: 320px 1fr; gap: 18px; }
    section, aside { background: white; border: 1px solid #dfe5ef; border-radius: 8px; padding: 18px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
    label { display: block; font-weight: 650; margin: 12px 0 6px; }
    input, textarea { width: 100%; box-sizing: border-box; border: 1px solid #c9d3e3; border-radius: 6px; padding: 10px 12px; font: inherit; }
    textarea { min-height: 132px; resize: vertical; }
    button { margin-top: 12px; border: 0; background: #1f6feb; color: white; border-radius: 6px; padding: 10px 14px; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    pre { white-space: pre-wrap; line-height: 1.45; background: #0f172a; color: #e5edf9; padding: 16px; border-radius: 8px; min-height: 360px; overflow: auto; }
    .metric { border-top: 1px solid #edf1f7; padding: 10px 0; font-size: 14px; }
    @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } main { padding: 18px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Automobile Intelligence Chatbot</h1>
        <div>Purchase, retention, and financial opportunity predictions with dataset-grounded justification.</div>
      </div>
    </header>
    <div class="grid">
      <aside>
        <label>Customer ID</label>
        <input id="cid" placeholder="Example: 9521" />
        <label>Question</label>
        <textarea id="q">Which customers should I prioritize for future purchase and why?</textarea>
        <button id="ask">Ask</button>
        <div class="metric">Leave Customer ID blank to use the first sampled customer.</div>
        <div class="metric">Set HF_TOKEN or LLM_API_KEY so every chat answer is generated by the LLM.</div>
      </aside>
      <section>
        <pre id="answer">Train models first, then ask a question.</pre>
      </section>
    </div>
  </main>
  <script>
    const ask = document.getElementById('ask');
    ask.onclick = async () => {
      ask.disabled = true;
      answer.textContent = 'Thinking...';
      const res = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question:q.value, customer_id:cid.value || null})});
      const data = await res.json();
      answer.textContent = data.answer;
      ask.disabled = false;
    };
  </script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "ok", "artifact_exists": ARTIFACT.exists(), "llm_configured": bool(get_llm_api_key())}


@app.post("/predict")
def predict(req: PredictRequest):
    bundle = load_bundle()
    record = req.record or find_record(bundle, req.customer_id)
    return {"record_id": record.get("id"), "scores": predict_record(bundle, record), "record": record}


@app.post("/chat")
def chat(req: ChatRequest):
    bundle = load_bundle()
    use_case = req.use_case or choose_use_case(req.question)
    top_count = requested_top_count(req.question)
    requested_id = req.customer_id or customer_id_from_question(req.question)
    context, scores = build_llm_context(bundle, req.question, use_case, requested_id, top_count)
    return {"answer": llm_answer(req.question, context, scores, use_case, req.history)}
