# Domain Intelligence-Based Phishing Detection

The active application uses one selected model: **Tuned 80 LightGBM**, with its matching saved preprocessor. A fresh OpenPhish feed check supplies a separate known-phishing decision when the exact input host or URL is listed. Final labels are **PHISHING** and **LEGITIMATE**. The original ML score and verdict are retained separately. Missing inputs use the fitted preprocessing defaults; data availability is reported separately.

## Run the component

From this project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\src\08_test_domain.py google.com --skip-ip-rdap
```

Test the supplied ten-domain list:

```powershell
.\.venv\Scripts\python.exe .\src\14_batch_test_domains.py .\work\domain_batch_20261014.csv
```

The batch command creates a fresh timestamped results folder by default.

## Decision evidence

For a bare domain, reputation matches only that exact hostname. A listed subdomain does not flag its parent or other subdomains. For a complete URL, the scheme, hostname, port, path and query must match; a different page on the same host is not treated as an exact URL match. Hostname case, default ports and fragments are normalized.

The published feed is fetched once per batch. A local single-domain cache can be reused for up to 12 hours. Stale or invalid snapshots cannot override the ML verdict. If the feed is unavailable, the final output uses Tuned 80 and records that availability limitation. No listed phishing page is fetched by the reputation check.

Use `--model-only` with either prediction command to disable the feed check for research evaluation. Dataset labels and input CSV reference labels never control predictions.

A final decision backed by the same feed used to select test cases measures lookup coverage. It does not demonstrate improved ML accuracy. Independently labelled examples and separate ML-only metrics are needed to evaluate general performance.

## Source files

| File | Purpose |
|---|---|
| `src/01_preprocess_dataset.py` | Extract the dataset's feature definitions; also used by alignment tests. |
| `src/02_validate_dataset.py` | Validate the prepared dataset. |
| `src/03_inspect_raw_records.py` | Inspect raw dataset records. |
| `src/04_prepare_training_data.py` | Prepare balanced training and test data. |
| `src/05_train_lightgbm.py` | Train the original research baseline. |
| `src/06_evaluate_lightgbm.py` | Evaluate the original baseline and produce research results. |
| `src/07_feature_analysis.py` | Analyse feature importance and correlations. |
| `src/08_test_domain.py` | Collect DNS, RDAP/WHOIS, TLS and IP features and predict with Tuned 80. |
| `src/10_tune_lightgbm.py` | Train and tune the selected 80-feature model. |
| `src/14_batch_test_domains.py` | Run the selected model across a domain list. |

The selected artifacts are `models/lightgbm_80_tuned_model.joblib` and `models/training_preprocessor_80_tuned.joblib`. Other saved model artifacts remain as experimental baselines.

## Archived experiments

Four unused source files are preserved in `archive/research_scripts_20261014/src`: the Tuned 40 training script, Original 80 calibration script, completed RDAP audit and old prediction backup. The archive README explains restoration and its manifest records the original paths and checksums.

Dataset preparation and research training/evaluation scripts remain available to reproduce and explain the component. Running training or tuning can write artifacts; use the live testing commands above for prediction only.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Recorded offline results favour Tuned 80 for accuracy, phishing recall and F1. Broader independently labelled evaluation is still needed to assess live performance. A binary label produced from entirely missing inputs is based on preprocessing defaults.

The model receives features for the exact input hostname. DNS and TLS retain that hostname; registration lookup may use its parent. Removing `www` changes the tested target and is not a general false-positive fix. In the balanced training data, 8,066 of 8,067 `www` hostnames are labelled phishing, leaving only one legitimate example in that subgroup. The name itself is not an ML feature, but coverage of legitimate hostname profiles needs improvement.

WHOIS error messages cannot supply registration data. Current NIC Chile replies without a Domain Name field are accepted only when the registry's own lookup link identifies the exact queried domain; otherwise registration remains missing or lookup continues to a parent. This validation fix does not guarantee that the model will correctly identify reported phishing pages.

For `.de`, a verified DENIC server referral avoids depending on IANA's port-43 service for each lookup. The client uses DENIC's documented `-T dn` technical-record query to obtain domain status, nameservers and the last change date. Creation and expiry dates remain missing when the public reply does not contain them.

Feature exports use unique timestamped filenames, preserving previous single-domain evidence.
