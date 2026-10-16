"""Public HTTP API for the C4 domain-intelligence detector (Tuned 80).

Reuses src/08_test_domain.py exactly as the batch script does.
Run locally:  python -m uvicorn api_server:app --port 7860
"""
import contextlib
import hmac
import importlib.util
import io
import ipaddress
import os
import socket
import sys
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("domain_live", ROOT / "src" / "08_test_domain.py")
live = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = live
spec.loader.exec_module(live)

COMPONENT = "C4_domain_intelligence"
API_KEY = os.environ.get("C4_API_KEY", "")
MODEL_ONLY = os.environ.get("C4_MODEL_ONLY", "") == "1"
ORIGINS = [o.strip() for o in os.environ.get("C4_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
BUNDLE = live.load_selected_model()          # (label, model, preprocessor, feature names)
gate = threading.BoundedSemaphore(1)         # collection prints are redirected; one at a time

app = FastAPI(title="R26-CS-019 C4 Domain Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "X-API-Key"])


class AnalyzeRequest(BaseModel):
    url: str


def contract(status, message=None, score=None, label=None, evidence=None):
    """Group result contract: label is null whenever status is not ok."""
    ok = status == "ok"
    return {"component": COMPONENT, "status": status,
            "label": label if ok else None, "score": score if ok else None,
            "threshold": live.THRESHOLD, "score_is_calibrated_probability": False,
            "message": message, "evidence": evidence or {}}


def resolves_only_public(host):
    """Refuse hosts that resolve to private/internal addresses (server safety)."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return True   # unresolvable domains are still analysed (DNS features record it)
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        mapped = getattr(address, "ipv4_mapped", None)
        if not address.is_global or (mapped and not mapped.is_global):
            return False
    return True


@app.get("/")
def root():
    return {"component": COMPONENT, "health": "/health", "analyze": "POST /analyze", "docs": "/docs"}


@app.get("/health")
def health():
    return {"component": COMPONENT, "status": "online", "ready": True, "model": BUNDLE[0],
            "feature_count": len(BUNDLE[3]), "reputation_check": not MODEL_ONLY}


@app.post("/analyze")
def analyze(body: AnalyzeRequest, x_api_key: str | None = Header(default=None)):
    if API_KEY and not hmac.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(401, "Missing or invalid X-API-Key.")
    raw = (body.url or "").strip()
    try:
        host = live.normalize_host(raw)
    except ValueError as e:
        return contract("invalid_input", str(e))
    if not resolves_only_public(host):
        return contract("invalid_input", "Host resolves to a private, local or reserved address.")
    if not gate.acquire(blocking=False):
        raise HTTPException(429, "An analysis is already running. Try again shortly.")
    try:
        label_name, model, preprocessor, names = BUNDLE
        features = {name: np.nan for name in names}
        with contextlib.redirect_stdout(io.StringIO()):
            status = {"DNS": live.collect_dns(host, features)}
            status["RDAP"] = live.collect_rdap(host, features)
            status["TLS"] = live.collect_tls(host, features)
            status["IP"] = live.collect_ip(status["DNS"], features, do_rdap=False)
            quality = live.feature_quality(features, status, details=False)
            snapshot = None if MODEL_ONLY else live.load_reputation_snapshot()
            score, _ = live.predict(model, preprocessor, features)
            decision = live.resolve_prediction(raw, score, snapshot)
    except Exception:
        return contract("error", "Analysis failed.")
    finally:
        gate.release()
    observed = sum(bool(live.nonmissing(features[name])) for name in names)
    evidence = {
        "host": host, "model": label_name,
        "model_prediction": decision["model_prediction"],
        "decision_source": decision["decision_source"],
        "reputation": decision["reputation"],
        "data_quality": quality,
        "observed_features": observed, "total_features": len(names),
        "available_groups": {k: bool(v.get("available")) for k, v in status.items()},
        "data_availability_note": live.prediction_data_note(
            quality, status["RDAP"].get("registry_only_fallback", False)),
    }
    return contract("ok", decision["prediction"].title(), score, decision["prediction"].lower(), evidence)
