
r"""11 - Train and Optuna-tune a separate 40-raw-feature LightGBM model.

Run from your project root:
    python .\src\11_train_40_feature_lightgbm.py --trials 15 --threads 4

Requires:
    pip install optuna lightgbm scikit-learn pandas numpy joblib

Uses the original balanced 50/50 train/test CSVs without changing them.

Feature ranking uses a temporary 80-feature LightGBM trained ONLY on the
internal tuning-training split (not the validation or test rows).

Unsupported or questionable live-domain fields are excluded before
choosing exactly 40 features.

The 40-feature preprocessor is fitted independently. No previously
trained model or preprocessor is overwritten.

Test data is NOT used for feature selection or Optuna tuning.

IMPORTANT:
The original test set has already been examined during earlier
experiments. A fresh independent live or temporal dataset is needed
before making final model-selection claims.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

TRAIN_CSV = PROCESSED / "train_balanced_raw.csv"
TEST_CSV = PROCESSED / "test_balanced_raw.csv"


# ============================================================
# 2. EXISTING MODELS
# ============================================================

REFERENCE_MODEL = (
    MODELS / "lightgbm_80_tuned_model.joblib"
)

REFERENCE_PREP = (
    MODELS / "training_preprocessor_80_tuned.joblib"
)

ORIGINAL_MODEL = (
    MODELS / "lightgbm_phishing_model.joblib"
)

ORIGINAL_PREP = (
    MODELS / "training_preprocessor.joblib"
)


# ============================================================
# 3. NEW 40-FEATURE MODEL OUTPUTS
# ============================================================

NEW_MODEL = (
    MODELS / "lightgbm_40_tuned_model.joblib"
)

NEW_PREP = (
    MODELS / "training_preprocessor_40_tuned.joblib"
)

FEATURE_LIST = (
    RESULTS / "lightgbm_40_selected_features.json"
)

FEATURE_RANKING = (
    RESULTS / "lightgbm_40_feature_ranking.csv"
)

TRIALS_CSV = (
    RESULTS / "lightgbm_40_tuning_trials.csv"
)

COMPARISON_CSV = (
    RESULTS / "lightgbm_40_test_comparison.csv"
)

SUMMARY_JSON = (
    RESULTS / "lightgbm_40_tuning_summary.json"
)


# ============================================================
# 4. SETTINGS
# ============================================================

SEED = 42

LABEL = "label"

N_FEATURES = 40

CATEGORICAL = {
    "tls_protocol",
    "tls_cipher",
}


# ============================================================
# 5. FEATURES EXCLUDED FROM THIS EXPERIMENT
# ============================================================

# These features are not necessarily bad predictors.
#
# They are excluded because the current live-domain
# extractor cannot reproduce them reliably, or because
# their training/live definitions need further investigation.
#
# This is an experimental selection policy.

UNRELIABLE_LIVE = {

    "dnssec_information_present",
    "dns_has_zone_soa",
    "dns_source_count",
    "dns_ttl_information_count",

    "tls_certificate_count",
    "tls_non_root_certificate_count",
    "tls_root_certificate_count",

    "ip_alive_count",
    "ip_alive_ratio",
    "ip_average_rtt",
    "ip_rdap_available_count",

    "asn_count",
    "asn_organization_count",

    "country_count",
    "country_code_count",
    "rir_count",

    "ip_from_cname_count",
    "ip_record_count",
    "ip_data_available",

    # Experimental RDAP exclusions.
    # We observed potential training/live mismatches.

    "rdap_registrant_count",
    "rdap_has_registrant",
    "rdap_status_count",
}


# ============================================================
# 6. MINIMUM FEATURES FROM EACH INTELLIGENCE GROUP
# ============================================================

# Keep all four intelligence groups represented.

MIN_PER_GROUP = {
    "DNS": 8,
    "RDAP": 8,
    "TLS": 6,
    "IP": 2,
}

RDAP_DATE_FIELDS = {
    "domain_age_days",
    "registration_period_days",
    "days_until_expiration",
    "days_since_last_change",
}


# ============================================================
# 7. IDENTIFY FEATURE GROUP
# ============================================================

def group_of(name: str) -> str:

    if (
        name.startswith("dns_")
        or name == "dnssec_information_present"
    ):
        return "DNS"

    if (
        name.startswith("rdap_")
        or name in RDAP_DATE_FIELDS
    ):
        return "RDAP"

    if name.startswith("tls_"):
        return "TLS"

    return "IP"


# ============================================================
# 8. EVALUATION METRICS
# ============================================================

def evaluate(y, prob):

    pred = (
        np.asarray(prob) >= 0.50
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

    return {

        "accuracy": float(
            accuracy_score(y, pred)
        ),

        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0
            )
        ),

        "specificity": (
            float(tn / (tn + fp))
            if tn + fp
            else None
        ),

        "roc_auc": float(
            roc_auc_score(y, prob)
        ),

        "pr_auc": float(
            average_precision_score(y, prob)
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 9. CHECK REQUIRED FILES
# ============================================================

def require_file(path: Path):

    if not path.is_file():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# 10. MAP TRANSFORMED FEATURES TO RAW FEATURES
# ============================================================

def raw_feature_for_transformed(
    transformed: str,
    originals: list[str]
):

    # Examples of transformed feature names:
    #
    # numeric__domain_age_days
    # numeric__missingindicator_domain_age_days
    # categorical__tls_protocol_TLSv1.3

    if transformed.startswith("numeric__"):

        value = transformed[
            len("numeric__"):
        ]

        if value.startswith(
            "missingindicator_"
        ):

            value = value[
                len("missingindicator_"):
            ]

        return (
            value
            if value in originals
            else None
        )

    if transformed.startswith(
        "categorical__"
    ):

        value = transformed[
            len("categorical__"):
        ]

        matches = [
            name
            for name in originals
            if value.startswith(name + "_")
        ]

        return (
            max(matches, key=len)
            if matches
            else None
        )

    return None


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

def feature_importances(
    model,
    preprocessor,
    originals: list[str]
) -> dict[str, float]:

    transformed_names = list(
        preprocessor.get_feature_names_out()
    )

    gains = np.asarray(

        model.booster_.feature_importance(
            importance_type="gain"
        ),

        dtype=float
    )

    if len(transformed_names) != len(gains):

        raise RuntimeError(

            "Importance/feature mismatch: "
            f"{len(gains)} gains vs "
            f"{len(transformed_names)} "
            "preprocessed features"
        )

    scores = {
        name: 0.0
        for name in originals
    }

    unknown = []

    for transformed, gain in zip(
        transformed_names,
        gains
    ):

        raw = raw_feature_for_transformed(
            str(transformed),
            originals
        )

        if raw is None:

            unknown.append(
                str(transformed)
            )

        else:

            scores[raw] += max(
                0.0,
                float(gain)
            )

    if unknown:

        raise RuntimeError(

            "Could not attribute importance "
            "to raw feature(s): "

            + ", ".join(
                unknown[:10]
            )
        )

    return scores


# ============================================================
# 12. SELECT EXACTLY 40 FEATURES
# ============================================================

def select_40(originals, gains):

    eligible = [

        feature
        for feature in originals

        if feature not in UNRELIABLE_LIVE
    ]

    if len(eligible) < N_FEATURES:

        raise RuntimeError(

            f"Only {len(eligible)} "
            "reliable candidates for "
            f"{N_FEATURES} slots."
        )

    # Higher gain = greater model importance.

    ordered = sorted(

        eligible,

        key=lambda feature: (
            -gains[feature],
            feature
        )
    )

    selected = []

    # First, guarantee a minimum number
    # of features from all four groups.

    for group, minimum in (
        MIN_PER_GROUP.items()
    ):

        candidates = [

            feature
            for feature in ordered

            if group_of(feature) == group
        ]

        if len(candidates) < minimum:

            raise RuntimeError(

                "Not enough candidate features "
                f"for {group} quota."
            )

        selected.extend(
            candidates[:minimum]
        )

    # Fill remaining slots based on importance.

    for name in ordered:

        if len(selected) >= N_FEATURES:
            break

        if name not in selected:
            selected.append(name)

    # Preserve original column order.

    chosen = [

        feature
        for feature in originals

        if feature in set(selected)
    ]

    if len(chosen) != N_FEATURES:

        raise RuntimeError(

            f"Expected 40 features, "
            f"selected {len(chosen)}."
        )

    return chosen


# ============================================================
# 13. BUILD NEW PREPROCESSOR
# ============================================================

def make_preprocessor(
    original_preprocessor,
    selected
):

    # Reuse original pipeline structure,
    # but not the fitted statistics.

    templates = {

        name: pipeline

        for name, pipeline, columns
        in original_preprocessor.transformers

        if (
            name != "remainder"
            and pipeline != "drop"
        )
    }

    numeric = [

        feature
        for feature in selected

        if feature not in CATEGORICAL
    ]

    categorical = [

        feature
        for feature in selected

        if feature in CATEGORICAL
    ]

    if "numeric" not in templates:

        raise RuntimeError(

            "Original preprocessor has "
            "no numeric pipeline."
        )

    transformer_list = [

        (
            "numeric",

            clone(
                templates["numeric"]
            ),

            numeric
        )
    ]

    if categorical:

        if "categorical" not in templates:

            raise RuntimeError(

                "Original preprocessor has "
                "no categorical pipeline."
            )

        transformer_list.append(

            (
                "categorical",

                clone(
                    templates["categorical"]
                ),

                categorical
            )
        )

    return ColumnTransformer(

        transformers=transformer_list,

        remainder="drop",

        sparse_threshold=1.0,

        verbose_feature_names_out=True
    )


# ============================================================
# 14. OPTUNA HYPERPARAMETER SETTINGS
# ============================================================

def params_for_trial(trial, args):

    return {

        "objective": "binary",

        "n_estimators": args.max_trees,

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.015,
            0.12,
            log=True
        ),

        "num_leaves": trial.suggest_int(
            "num_leaves",
            15,
            79
        ),

        "max_depth": trial.suggest_categorical(
            "max_depth",
            [-1, 5, 7, 9, 12]
        ),

        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            20,
            150
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.65,
            1.0
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.70,
            1.0
        ),

        "subsample_freq": 1,

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-4,
            10,
            log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-3,
            15,
            log=True
        ),

        "random_state": SEED,

        "n_jobs": args.threads,

        "verbosity": -1,
    }


# ============================================================
# 15. OPTUNA FINE-TUNING
# ============================================================

def tune(
    X_train,
    y_train,
    X_val,
    y_val,
    args
):

    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )

    study = optuna.create_study(

        direction="maximize",

        sampler=optuna.samplers.TPESampler(
            seed=SEED
        )
    )

    def objective(trial):

        model = lgb.LGBMClassifier(

            **params_for_trial(
                trial,
                args
            )
        )

        model.fit(

            X_train,
            y_train,

            eval_set=[
                (X_val, y_val)
            ],

            eval_metric="binary_logloss",

            callbacks=[

                lgb.early_stopping(
                    args.early_stopping,
                    verbose=False
                ),

                lgb.log_evaluation(0)
            ]
        )

        prob = model.predict_proba(
            X_val
        )[:, 1]

        metrics = evaluate(
            y_val,
            prob
        )

        best_round = int(

            model.best_iteration_
            or args.max_trees
        )

        trial.set_user_attr(
            "best_iteration",
            best_round
        )

        trial.set_user_attr(
            "validation_metrics",
            metrics
        )

        print(

            f"Trial {trial.number + 1:02}/"
            f"{args.trials}: "

            f"F1={metrics['f1']:.5f} "

            f"accuracy={metrics['accuracy']:.5f} "

            f"FP={metrics['fp']} "

            f"FN={metrics['fn']} "

            f"trees={best_round}",

            flush=True
        )

        return metrics["f1"]

    begin = time.perf_counter()

    study.optimize(

        objective,

        n_trials=args.trials,

        gc_after_trial=True
    )

    elapsed = (
        time.perf_counter() - begin
    ) / 60

    print(

        "Tuning elapsed time:",
        f"{elapsed:.2f} min"
    )

    return study


# ============================================================
# 16. EVALUATE EXISTING MODEL
# ============================================================

def evaluate_existing(
    model_path,
    prep_path,
    raw_X,
    y
):

    if (
        not model_path.exists()
        or not prep_path.exists()
    ):

        return None

    old_model = joblib.load(
        model_path
    )

    old_prep = joblib.load(
        prep_path
    )

    feature_names = list(
        old_prep.feature_names_in_
    )

    prepared = old_prep.transform(

        raw_X[feature_names]
    )

    prob = old_model.predict_proba(
        prepared
    )[:, 1]

    return evaluate(
        y,
        prob
    )


# ============================================================
# 17. COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description=(
            "Select and tune 40 domain "
            "intelligence LightGBM features"
        )
    )

    parser.add_argument(

        "--trials",

        type=int,

        default=15,

        help="Number of Optuna trials"
    )

    parser.add_argument(

        "--threads",

        type=int,

        default=4,

        help="Number of CPU threads"
    )

    parser.add_argument(

        "--max-trees",

        type=int,

        default=1500,

        help="Maximum LightGBM trees"
    )

    parser.add_argument(

        "--early-stopping",

        type=int,

        default=60,

        help="Early stopping rounds"
    )

    parser.add_argument(

        "--validation-size",

        type=float,

        default=0.15,

        help="Training-only validation fraction"
    )

    parser.add_argument(

        "--skip-test",

        action="store_true",

        help="Skip official held-out test evaluation"
    )

    args = parser.parse_args()

    if (
        args.trials < 1
        or args.threads < 1
        or args.max_trees < 10
        or args.early_stopping < 1
    ):

        parser.error(

            "Trials, threads and early-stopping "
            "must be >= 1; max-trees >= 10"
        )

    if not (
        0.05 <= args.validation_size <= 0.40
    ):

        parser.error(

            "Validation size must be "
            "between 0.05 and 0.40"
        )

    return args


# ============================================================
# 18. MAIN TRAINING FUNCTION
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # CHECK INPUT FILES
    # --------------------------------------------------------

    for path in (

        TRAIN_CSV,
        REFERENCE_MODEL,
        REFERENCE_PREP

    ):

        require_file(path)

    MODELS.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 76)

    print(
        "11 - SELECT 40 RELIABLE RAW FEATURES "
        "+ TUNE NEW LIGHTGBM"
    )

    print("=" * 76)

    # --------------------------------------------------------
    # LOAD REFERENCE MODEL AND PREPROCESSOR
    # --------------------------------------------------------

    reference_model = joblib.load(
        REFERENCE_MODEL
    )

    reference_prep = joblib.load(
        REFERENCE_PREP
    )

    originals = list(
        reference_prep.feature_names_in_
    )

    if len(originals) != 80:

        raise RuntimeError(

            "Reference model expects "
            f"{len(originals)} raw features, "
            "not 80."
        )

    # --------------------------------------------------------
    # LOAD ORIGINAL BALANCED DATASET
    # --------------------------------------------------------

    print(
        "\nReading original balanced training data..."
    )

    raw = pd.read_csv(

        TRAIN_CSV,

        usecols=originals + [LABEL],

        low_memory=False
    )

    if not raw[LABEL].isin(
        [0, 1]
    ).all():

        raise RuntimeError(

            "Expected labels "
            "0=legitimate, 1=phishing."
        )

    y = raw[LABEL].to_numpy(
        dtype=np.int8
    )

    counts = np.bincount(
        y,
        minlength=2
    )

    if counts[0] != counts[1]:

        raise RuntimeError(

            "Expected balanced training data; "

            f"found {counts[0]} legitimate "
            f"and {counts[1]} phishing. "

            "No automatic resampling "
            "will be performed."
        )

    X_all = raw[originals].copy()

    # --------------------------------------------------------
    # SPLIT BEFORE FEATURE SELECTION
    # --------------------------------------------------------

    X_fit_all, X_valid_all, y_fit, y_valid = (
        train_test_split(

            X_all,
            y,

            test_size=args.validation_size,

            random_state=2026,

            stratify=y
        )
    )

    print(

        f"Official TRAIN: {len(y):,} rows; "

        f"{counts[0]:,} legitimate, "

        f"{counts[1]:,} phishing"
    )

    print(

        "Training-only selection rows:",
        f"{len(y_fit):,}"
    )

    print(

        "Untouched tuning validation rows:",
        f"{len(y_valid):,}"
    )

    # --------------------------------------------------------
    # RANK ALL 80 FEATURES USING TRAINING ROWS ONLY
    # --------------------------------------------------------

    print(
        "\nRanking feature gain "
        "using training rows ONLY..."
    )

    ranking_prep = clone(
        reference_prep
    )

    X_rank = ranking_prep.fit_transform(
        X_fit_all
    )

    selector_model = lgb.LGBMClassifier(

        objective="binary",

        n_estimators=650,

        learning_rate=0.06,

        num_leaves=31,

        min_child_samples=40,

        colsample_bytree=0.85,

        reg_lambda=1.0,

        random_state=SEED,

        n_jobs=args.threads,

        verbosity=-1,
    )

    selector_model.fit(
        X_rank,
        y_fit
    )

    gains = feature_importances(

        selector_model,

        ranking_prep,

        originals
    )

    # --------------------------------------------------------
    # SELECT EXACTLY 40 FEATURES
    # --------------------------------------------------------

    selected = select_40(
        originals,
        gains
    )

    excluded = [

        name
        for name in originals

        if name in UNRELIABLE_LIVE
    ]

    print(

        "Excluded unreliable/questionable "
        "live fields:",

        len(excluded)
    )

    print(

        "Selected raw features:",
        len(selected)
    )

    for group in MIN_PER_GROUP:

        group_features = [

            feature
            for feature in selected

            if group_of(feature) == group
        ]

        print(

            f"  {group:4s}: "
            f"{len(group_features)} selected: "

            + ", ".join(group_features)
        )

    # --------------------------------------------------------
    # SAVE FEATURE RANKING
    # --------------------------------------------------------

    ranking = pd.DataFrame(

        [

            {

                "feature": feature,

                "group": group_of(
                    feature
                ),

                "gain_80": gains[feature],

                "excluded_from_40": (
                    feature in UNRELIABLE_LIVE
                ),

                "selected_for_40": (
                    feature in selected
                ),
            }

            for feature in sorted(

                originals,

                key=lambda feature: (
                    -gains[feature],
                    feature
                )
            )
        ]
    )

    ranking.to_csv(
        FEATURE_RANKING,
        index=False
    )

    FEATURE_LIST.write_text(

        json.dumps(

            {

                "feature_count": len(selected),

                "feature_names": selected,

                "selection_method": (
                    "training-fold 80-feature "
                    "LightGBM gain + live-reliability "
                    "exclusions + group quotas"
                ),

                "excluded_fields": excluded,

                "minimum_per_group": (
                    MIN_PER_GROUP
                ),
            },

            indent=2
        ),

        encoding="utf-8"
    )

    # --------------------------------------------------------
    # PREPARE 40-FEATURE TRAIN / VALIDATION DATA
    # --------------------------------------------------------

    X = X_all[selected].copy()

    X_fit_raw = X_fit_all[selected]

    X_valid_raw = X_valid_all[selected]

    tuning_prep = make_preprocessor(
        reference_prep,
        selected
    )

    X_fit = tuning_prep.fit_transform(
        X_fit_raw
    )

    X_valid = tuning_prep.transform(
        X_valid_raw
    )

    print(

        f"Tuning train={len(y_fit):,}; "
        f"validation={len(y_valid):,}"
    )

    print(

        "40 raw features become",

        X_fit.shape[1],

        "processed columns"
    )

    # --------------------------------------------------------
    # RUN OPTUNA FINE-TUNING
    # --------------------------------------------------------

    study = tune(

        X_fit,
        y_fit,

        X_valid,
        y_valid,

        args
    )

    best = study.best_trial

    study.trials_dataframe().to_csv(
        TRIALS_CSV,
        index=False
    )

    best_round = max(

        1,

        int(
            best.user_attrs[
                "best_iteration"
            ]
        )
    )

    print(

        "\nBest validation F1:",
        f"{best.value:.5f}"
    )

    print(

        "Best parameters:",

        json.dumps(
            best.params,
            indent=2
        )
    )

    print(
        "Best trees:",
        best_round
    )

    # --------------------------------------------------------
    # TRAIN FINAL NEW MODEL
    # --------------------------------------------------------

    print(

        "\nTraining final 40-feature model "
        "on ALL official train rows..."
    )

    final_prep = make_preprocessor(
        reference_prep,
        selected
    )

    X_final = final_prep.fit_transform(
        X
    )

    params = {

        **best.params,

        "objective": "binary",

        "n_estimators": best_round,

        "subsample_freq": 1,

        "random_state": SEED,

        "n_jobs": args.threads,

        "verbosity": -1,
    }

    new_model = lgb.LGBMClassifier(
        **params
    )

    new_model.fit(
        X_final,
        y
    )

    # Save separately.
    # Existing 80-feature models remain unchanged.

    joblib.dump(
        new_model,
        NEW_MODEL
    )

    joblib.dump(
        final_prep,
        NEW_PREP
    )

    print(
        "Saved new model:",
        NEW_MODEL
    )

    print(
        "Saved new preprocessor:",
        NEW_PREP
    )

    # --------------------------------------------------------
    # COMPARE ALL THREE MODELS
    # --------------------------------------------------------

    test_comparison = None

    if not args.skip_test:

        require_file(
            TEST_CSV
        )

        print(

            "\nComparing ALL THREE models "
            "on the same held-out dataset..."
        )

        test = pd.read_csv(

            TEST_CSV,

            usecols=originals + [LABEL],

            low_memory=False
        )

        X_test = test[originals]

        y_test = test[LABEL].to_numpy(
            dtype=np.int8
        )

        # New 40-feature model

        p40 = new_model.predict_proba(

            final_prep.transform(
                X_test[selected]
            )
        )[:, 1]

        rows = [

            {

                "model": "40_feature_tuned",

                **evaluate(
                    y_test,
                    p40
                )
            }
        ]

        # Compare with existing 80-feature models

        for label, model_path, prep_path in (

            (
                "80_feature_original",
                ORIGINAL_MODEL,
                ORIGINAL_PREP
            ),

            (
                "80_feature_tuned",
                REFERENCE_MODEL,
                REFERENCE_PREP
            ),

        ):

            result = evaluate_existing(

                model_path,

                prep_path,

                X_test,

                y_test
            )

            if result is not None:

                rows.append(

                    {

                        "model": label,

                        **result
                    }
                )

        comparison = pd.DataFrame(
            rows
        )

        comparison = comparison.sort_values(
            "model"
        ).reset_index(
            drop=True
        )

        comparison.to_csv(
            COMPARISON_CSV,
            index=False
        )

        print(

            comparison.to_string(

                index=False,

                float_format=lambda value: (
                    f"{value:.5f}"
                )
            )
        )

        test_comparison = comparison.to_dict(
            orient="records"
        )

        print(

            "Saved test comparison:",
            COMPARISON_CSV
        )

    # --------------------------------------------------------
    # SAVE RESEARCH SUMMARY
    # --------------------------------------------------------

    summary = {

        "selected_features": selected,

        "feature_count": len(selected),

        "training_samples": int(
            len(y)
        ),

        "legitimate_samples": int(
            counts[0]
        ),

        "phishing_samples": int(
            counts[1]
        ),

        "optuna_trials": args.trials,

        "best_validation_f1": float(
            best.value
        ),

        "best_validation_metrics": (
            best.user_attrs[
                "validation_metrics"
            ]
        ),

        "best_trees": best_round,

        "best_parameters": best.params,

        "test_comparison": test_comparison,

        "important_note": (

            "40-feature selection excluded "
            "historically unreliable live fields. "

            "This is an offline experiment, "
            "not proof of improved live detection. "

            "Check live input consistency before "
            "comparing operational accuracy. "

            "The official test has been inspected "
            "during earlier experiments."
        ),
    }

    SUMMARY_JSON.write_text(

        json.dumps(
            summary,
            indent=2
        ),

        encoding="utf-8"
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()

    print(
        "=" * 76
    )

    print(
        "40-FEATURE LIGHTGBM TRAINING COMPLETED"
    )

    print(
        "=" * 76
    )

    print(
        "Original and tuned 80-feature "
        "models were NOT overwritten."
    )

    print(
        "40 selected features:",
        FEATURE_LIST
    )

    print(
        "Feature ranking:",
        FEATURE_RANKING
    )

    print(
        "Tuning trials:",
        TRIALS_CSV
    )

    print(
        "Summary:",
        SUMMARY_JSON
    )

    print()

    print(

        "Reminder: current 08_test_domain.py "
        "compares ONLY the two 80-feature models."
    )

    print(

        "It must be adapted to load these "
        "40 selected raw feature names later."
    )


# ============================================================
# 19. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
