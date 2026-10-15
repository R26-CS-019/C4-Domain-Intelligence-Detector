"""Collect live domain features and classify with the selected Tuned 80 model.

Input CSV: domain,expected_label,label_context. Labels are reference labels,
not evidence that a historically malicious domain is still malicious today.
No dataset, model or earlier evidence files are overwritten.
Successful predictions are binary even with no live features; fitted
preprocessors provide missing-value defaults and availability is recorded.
Fresh exact-host/URL OpenPhish matches produce a separate final phishing
decision. The unmodified Tuned 80 score and ML verdict remain in evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_live():
    spec = importlib.util.spec_from_file_location("domain_batch_live", ROOT / "src/08_test_domain.py")
    live = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = live
    spec.loader.exec_module(live)
    return live


def model_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "models").glob("*.joblib"))}


def load_inputs(path, live):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "domain" not in rows[0]:
        raise ValueError("Input CSV must contain a domain column and at least one domain.")
    seen = set()
    result = []
    for row in rows:
        host = live.normalize_host(row["domain"])
        if host in seen:
            raise ValueError(f"Duplicate domain: {host}")
        seen.add(host)
        label = row.get("expected_label", "").strip().lower()
        if label not in ("", "legitimate", "phishing"):
            raise ValueError(f"Unsupported reference label for {host}: {label}")
        result.append({"domain": host, "input_value": row["domain"].strip(), "expected_label": label.upper() or None,
                       "label_context": row.get("label_context", "unspecified").strip()})
    return result


def test_domain(item, live, bundles, output, with_ip_rdap=False, reputation_snapshot=None):
    host = item["domain"]
    folder = output / host
    folder.mkdir()
    features = {name: np.nan for name in bundles[0][3]}
    log = io.StringIO()
    record = {**item, "observed_at_utc": datetime.now(timezone.utc).isoformat(), "models": []}
    try:
        with contextlib.redirect_stdout(log):
            status = {}
            status["DNS"] = live.collect_dns(host, features)
            status["RDAP"] = live.collect_rdap(host, features, debug=True)
            status["TLS"] = live.collect_tls(host, features)
            status["IP"] = live.collect_ip(status["DNS"], features, do_rdap=with_ip_rdap)
            quality = live.feature_quality(features, status, details=True)
            record.update({
                "data_quality": quality,
                "available_groups": {key: bool(value["available"]) for key, value in status.items()},
                "rdap_source": status["RDAP"].get("source"),
                "registration_method": status["RDAP"].get("method"),
                "rdap_diagnostics": status["RDAP"].get("diagnostics", []),
                "rdap_registry_only_fallback": status["RDAP"].get("registry_only_fallback", False),
                "tls_certificate_verified": status["TLS"].get("certificate_verified"),
            })
            for label, model, preprocessor, names in bundles:
                observed = sum(live.nonmissing(features[name]) for name in names)
                row = {"model": label, "observed_features": observed, "total_features": len(names),
                       "missing_features": [name for name in names if not live.nonmissing(features[name])]}
                score, matrix = live.predict(model, preprocessor, features)
                decision = live.resolve_prediction(item.get("input_value", host), score, reputation_snapshot)
                prediction = decision["prediction"]
                row.update({"raw_phishing_score": score, "prediction": prediction,
                            "model_prediction": decision["model_prediction"],
                            "decision_source": decision["decision_source"],
                            "reputation": decision["reputation"],
                            "prediction_status": "CLASSIFIED",
                            "data_availability_note": live.prediction_data_note(quality, record["rdap_registry_only_fallback"]),
                            "matches_reference_label": prediction == item["expected_label"] if item["expected_label"] else None})
                live.explain_prediction(model, preprocessor, matrix, label)
                record["models"].append(row)
    except Exception as error:
        record.update({"data_quality": "ERROR", "error": f"{type(error).__name__}: {error}"})
        traceback.print_exc(file=log)
    finally:
        pd.DataFrame([features]).to_csv(folder / "features.csv", index=False)
        (folder / "collection_log.txt").write_text(log.getvalue(), encoding="utf-8")
        (folder / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return record


def flattened(records):
    rows = []
    for record in records:
        for model in record["models"] or [{"model": "NOT RUN", "prediction": None, "prediction_status": "ERROR"}]:
            rows.append({key: record.get(key) for key in ("domain", "expected_label", "label_context", "data_quality", "error")}
                        | {key: model.get(key) for key in ("model", "raw_phishing_score", "prediction", "prediction_status",
                                                         "model_prediction", "decision_source",
                                                         "observed_features", "total_features", "matches_reference_label",
                                                         "data_availability_note")})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-ip-rdap", action="store_true", help="Include optional IP RDAP enrichment")
    parser.add_argument("--model-only", action="store_true", help="Disable reputation check for ML-only research evaluation")
    args = parser.parse_args()
    live = load_live()
    inputs = load_inputs(args.input_csv, live)
    hashes = model_hashes()
    bundles = [live.load_selected_model()]
    stamp = datetime.now(ZoneInfo("Asia/Colombo")).strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "results" / f"live_domain_tests_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    live.IANA_BOOTSTRAP_CACHE = output / "rdap_bootstrap_cache.json"
    reputation_snapshot = None if args.model_only else live.load_reputation_snapshot(output / "reputation_snapshot.json")
    pd.DataFrame(inputs).to_csv(output / "input_domains.csv", index=False)
    records = []
    print(f"Results folder: {output}", flush=True)
    for index, item in enumerate(inputs, 1):
        print(f"[{index}/{len(inputs)}] Collecting {item['domain']}", flush=True)
        record = test_domain(item, live, bundles, output, args.with_ip_rdap, reputation_snapshot)
        records.append(record)
        flattened(records).to_csv(output / "comparison.csv", index=False)
        (output / "results.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
        predictions = "; ".join(f"FINAL: {row['prediction']} (source={row['decision_source']}; ML={row['model_prediction']})" for row in record["models"])
        print(f"  {record['data_quality']}: {predictions or record.get('error', 'No predictions')}", flush=True)
    unchanged = model_hashes() == hashes
    manifest = {"model_hashes_unchanged": unchanged, "model_sha256": hashes,
                "with_ip_rdap": args.with_ip_rdap, "threshold": live.THRESHOLD,
                "label_source": "Reference labels and their contexts come from the input CSV; consult per-domain provenance before interpreting agreement.",
                "classification_policy": "binary",
                "active_model": bundles[0][0],
                "active_model_file": live.TUNED_MODEL.name,
                "active_preprocessor_file": live.TUNED_PREPROCESSOR.name,
                "active_raw_feature_count": len(bundles[0][3]),
                "reputation_enabled": not args.model_only,
                "reputation_available": bool(reputation_snapshot and reputation_snapshot.get("available")),
                "reputation_source": reputation_snapshot.get("source") if reputation_snapshot else None,
                "reputation_snapshot_sha256": reputation_snapshot.get("feed_sha256") if reputation_snapshot else None,
                "interpretation": "Raw scores and model_prediction are unmodified Tuned 80 outputs. prediction is the final decision, using fresh exact-host/URL reputation matches when available, otherwise the ML verdict. Missing features use saved preprocessing defaults. Feed-based decisions evaluated against that same feed measure lookup coverage, not independent ML accuracy."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("A saved model artifact changed during the batch test.")
    print(flattened(records).to_string(index=False), flush=True)
    print("Saved model checksums unchanged.", flush=True)


if __name__ == "__main__":
    main()
