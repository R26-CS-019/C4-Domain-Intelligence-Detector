
"""
10 - FINE-TUNE 80-FEATURE LIGHTGBM

Research component: Domain Intelligence-Based Phishing Detection

What this script does:
1. Load the existing balanced training dataset.
2. Keep all original 80 raw features.
3. Create a separate training/validation split.
4. Fine-tune LightGBM using Optuna.
5. Train a new tuned model on all training data.
6. Compare the original and tuned models on the held-out test set.
7. Save the tuned model without overwriting the original.

Run:
    python .\\src\\10_tune_lightgbm.py --trials 15 --threads 4

Required packages:
    optuna
    lightgbm
    pandas
    numpy
    scipy
    scikit-learn
    joblib

Important:
- Keeps the original balanced dataset.
- Does not reduce the original 80 features.
- Fits preprocessing only on the tuning-training split.
- Does not use the final test set to select hyperparameters.
- Does not modify your existing model.
- The original probability calibrator must not be reused
  with the newly tuned model.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# 2. INPUT FILES
# ============================================================

TRAIN_RAW_FILE = (
    PROCESSED_DIR / "train_balanced_raw.csv"
)

TEST_RAW_FILE = (
    PROCESSED_DIR / "test_balanced_raw.csv"
)

ORIGINAL_PREPROCESSOR_FILE = (
    MODELS_DIR / "training_preprocessor.joblib"
)

ORIGINAL_MODEL_FILE = (
    MODELS_DIR / "lightgbm_phishing_model.joblib"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

TUNED_MODEL_FILE = (
    MODELS_DIR / "lightgbm_80_tuned_model.joblib"
)

TUNED_PREPROCESSOR_FILE = (
    MODELS_DIR / "training_preprocessor_80_tuned.joblib"
)

TUNING_TRIALS_FILE = (
    RESULTS_DIR / "lightgbm_80_tuning_trials.csv"
)

TUNING_SUMMARY_FILE = (
    RESULTS_DIR / "lightgbm_80_tuning_summary.json"
)

COMPARISON_FILE = (
    RESULTS_DIR / "lightgbm_80_tuned_test_comparison.csv"
)


# ============================================================
# 4. SETTINGS
# ============================================================

RANDOM_STATE = 42

LABEL_COLUMN = "label"

PHISHING_THRESHOLD = 0.50


# ============================================================
# 5. PRINT SECTION
# ============================================================

def print_section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 6. CALCULATE EVALUATION METRICS
# ============================================================

def calculate_metrics(y_true, probabilities):

    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= PHISHING_THRESHOLD
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    results = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "specificity": (
            float(tn / (tn + fp))
            if (tn + fp) > 0
            else None
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        ),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return results


# ============================================================
# 7. LOAD ORIGINAL BALANCED TRAINING DATA
# ============================================================

def load_training_data(original_preprocessor):

    print_section(
        "LOADING ORIGINAL BALANCED TRAINING DATA"
    )

    if not hasattr(
        original_preprocessor,
        "feature_names_in_"
    ):
        raise RuntimeError(
            "The original preprocessor does not contain "
            "feature_names_in_. Cannot identify the "
            "original 80 features safely."
        )

    feature_names = list(
        original_preprocessor.feature_names_in_
    )

    if len(feature_names) != 80:
        raise RuntimeError(
            "Expected 80 raw features, but found "
            f"{len(feature_names)}."
        )

    if not TRAIN_RAW_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_RAW_FILE}"
        )

    print("Training file:", TRAIN_RAW_FILE)

    dataset = pd.read_csv(
        TRAIN_RAW_FILE,
        usecols=feature_names + [LABEL_COLUMN],
        low_memory=False
    )

    X = dataset[feature_names].copy()

    y = dataset[LABEL_COLUMN].to_numpy(
        dtype=np.int8
    )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            "Labels must be 0=LEGITIMATE and 1=PHISHING."
        )

    counts = np.bincount(
        y,
        minlength=2
    )

    if counts.min() == 0:
        raise ValueError(
            "Both legitimate and phishing samples "
            "are required."
        )

    print()
    print("Total samples:", f"{len(y):,}")
    print("Legitimate samples:", f"{counts[0]:,}")
    print("Phishing samples:", f"{counts[1]:,}")
    print("Number of raw features:", len(feature_names))
    print()

    return X, y, feature_names


# ============================================================
# 8. OPTUNA HYPERPARAMETER SEARCH SPACE
# ============================================================

def get_trial_parameters(
    trial,
    max_trees,
    threads
):

    parameters = {

        "objective": "binary",

        "n_estimators": int(max_trees),

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
            0.0001,
            10.0,
            log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0.001,
            15.0,
            log=True
        ),

        "random_state": RANDOM_STATE,

        "n_jobs": threads,

        "verbosity": -1
    }

    return parameters


# ============================================================
# 9. RUN OPTUNA TUNING
# ============================================================

def tune_lightgbm(
    X_train,
    y_train,
    X_validation,
    y_validation,
    args
):

    print_section(
        "STARTING OPTUNA LIGHTGBM HYPERPARAMETER TUNING"
    )

    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    def objective(trial):

        parameters = get_trial_parameters(
            trial,
            args.max_trees,
            args.threads
        )

        model = lgb.LGBMClassifier(
            **parameters
        )

        model.fit(

            X_train,
            y_train,

            eval_set=[
                (
                    X_validation,
                    y_validation
                )
            ],

            eval_metric="binary_logloss",

            callbacks=[

                lgb.early_stopping(
                    stopping_rounds=args.early_stopping,
                    verbose=False
                ),

                lgb.log_evaluation(
                    period=0
                )
            ]
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        metrics = calculate_metrics(
            y_validation,
            probabilities
        )

        best_iteration = int(
            model.best_iteration_
            or args.max_trees
        )

        trial.set_user_attr(
            "best_iteration",
            best_iteration
        )

        trial.set_user_attr(
            "validation_metrics",
            metrics
        )

        print(
            f"Trial {trial.number + 1}/{args.trials} | "
            f"F1: {metrics['f1']:.5f} | "
            f"Accuracy: {metrics['accuracy']:.5f} | "
            f"ROC-AUC: {metrics['roc_auc']:.5f} | "
            f"FP: {metrics['false_positive']} | "
            f"FN: {metrics['false_negative']} | "
            f"Trees: {best_iteration}",
            flush=True
        )

        # Optimize validation F1 score.
        return metrics["f1"]

    start_time = time.perf_counter()

    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=False,
        gc_after_trial=True
    )

    elapsed_minutes = (
        time.perf_counter() - start_time
    ) / 60

    print()
    print(
        "Tuning completed in:",
        f"{elapsed_minutes:.2f} minutes"
    )

    best_trial = study.best_trial

    print()
    print(
        "Best validation F1:",
        f"{best_trial.value:.5f}"
    )

    print(
        "Best number of trees:",
        best_trial.user_attrs["best_iteration"]
    )

    print()
    print("Best hyperparameters:")

    print(
        json.dumps(
            best_trial.params,
            indent=4
        )
    )

    return study


# ============================================================
# 10. TRAIN FINAL TUNED MODEL
# ============================================================

def train_final_tuned_model(
    X,
    y,
    original_preprocessor,
    best_trial,
    args
):

    print_section(
        "TRAINING FINAL TUNED 80-FEATURE LIGHTGBM"
    )

    # Build a NEW preprocessor with the same structure.
    # Fit only on the official training dataset.

    tuned_preprocessor = clone(
        original_preprocessor
    )

    X_prepared = tuned_preprocessor.fit_transform(
        X
    )

    best_trees = max(
        1,
        int(
            best_trial.user_attrs[
                "best_iteration"
            ]
        )
    )

    final_parameters = {

        **best_trial.params,

        "objective": "binary",

        "n_estimators": best_trees,

        "subsample_freq": 1,

        "random_state": RANDOM_STATE,

        "n_jobs": args.threads,

        "verbosity": -1
    }

    tuned_model = lgb.LGBMClassifier(
        **final_parameters
    )

    print(
        "Training samples:",
        f"{len(y):,}"
    )

    print(
        "Raw input features:",
        X.shape[1]
    )

    print(
        "Features after preprocessing:",
        X_prepared.shape[1]
    )

    print(
        "Number of trees:",
        best_trees
    )

    tuned_model.fit(
        X_prepared,
        y
    )

    joblib.dump(
        tuned_model,
        TUNED_MODEL_FILE
    )

    joblib.dump(
        tuned_preprocessor,
        TUNED_PREPROCESSOR_FILE
    )

    print()
    print(
        "Tuned model saved:",
        TUNED_MODEL_FILE
    )

    print(
        "Tuned preprocessor saved:",
        TUNED_PREPROCESSOR_FILE
    )

    return tuned_model, tuned_preprocessor


# ============================================================
# 11. COMPARE ORIGINAL VS TUNED MODEL
# ============================================================

def compare_models(
    tuned_model,
    tuned_preprocessor,
    original_preprocessor
):

    print_section(
        "COMPARING ORIGINAL AND TUNED LIGHTGBM"
    )

    if not TEST_RAW_FILE.exists():
        print(
            "Test CSV not found. Comparison skipped."
        )
        return None

    if not ORIGINAL_MODEL_FILE.exists():
        print(
            "Original model not found. Comparison skipped."
        )
        return None

    feature_names = list(
        original_preprocessor.feature_names_in_
    )

    test_data = pd.read_csv(
        TEST_RAW_FILE,
        usecols=feature_names + [LABEL_COLUMN],
        low_memory=False
    )

    X_test = test_data[feature_names]

    y_test = test_data[
        LABEL_COLUMN
    ].to_numpy(dtype=np.int8)

    # --------------------------------------------------------
    # TUNED MODEL
    # --------------------------------------------------------

    X_test_tuned = (
        tuned_preprocessor.transform(
            X_test
        )
    )

    tuned_probabilities = (
        tuned_model.predict_proba(
            X_test_tuned
        )[:, 1]
    )

    tuned_metrics = calculate_metrics(
        y_test,
        tuned_probabilities
    )

    # --------------------------------------------------------
    # ORIGINAL MODEL
    # --------------------------------------------------------

    original_model = joblib.load(
        ORIGINAL_MODEL_FILE
    )

    X_test_original = (
        original_preprocessor.transform(
            X_test
        )
    )

    original_probabilities = (
        original_model.predict_proba(
            X_test_original
        )[:, 1]
    )

    original_metrics = calculate_metrics(
        y_test,
        original_probabilities
    )

    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    comparison = pd.DataFrame([

        {
            "model": "original_80",
            **original_metrics
        },

        {
            "model": "tuned_80",
            **tuned_metrics
        }
    ])

    print()
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.5f}"
            )
        )
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False
    )

    print()
    print(
        "Comparison saved:",
        COMPARISON_FILE
    )

    return comparison.to_dict(
        orient="records"
    )


# ============================================================
# 12. COMMAND LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the existing 80-feature LightGBM "
            "model using Optuna."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Number of Optuna trials."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads."
    )

    parser.add_argument(
        "--max-trees",
        type=int,
        default=1500,
        help="Maximum LightGBM trees per trial."
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=60,
        help="Early stopping rounds."
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
        help="Fraction of training data for validation."
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip final held-out test evaluation."
    )

    args = parser.parse_args()

    if args.trials < 1:
        parser.error(
            "--trials must be at least 1."
        )

    if args.threads < 1:
        parser.error(
            "--threads must be at least 1."
        )

    if args.max_trees < 10:
        parser.error(
            "--max-trees must be at least 10."
        )

    if args.early_stopping < 1:
        parser.error(
            "--early-stopping must be at least 1."
        )

    if not (
        0.05 <= args.validation_size <= 0.40
    ):
        parser.error(
            "--validation-size must be between "
            "0.05 and 0.40."
        )

    return args


# ============================================================
# 13. MAIN FUNCTION
# ============================================================

def main():

    args = parse_arguments()

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print_section(
        "10 - FINE-TUNING 80-FEATURE LIGHTGBM"
    )

    if not ORIGINAL_PREPROCESSOR_FILE.exists():
        raise FileNotFoundError(
            "Original preprocessor not found:\n"
            f"{ORIGINAL_PREPROCESSOR_FILE}"
        )

    # --------------------------------------------------------
    # LOAD ORIGINAL PREPROCESSOR
    # --------------------------------------------------------

    original_preprocessor = joblib.load(
        ORIGINAL_PREPROCESSOR_FILE
    )

    # --------------------------------------------------------
    # LOAD ORIGINAL BALANCED DATA
    # --------------------------------------------------------

    X_raw, y, feature_names = load_training_data(
        original_preprocessor
    )

    # --------------------------------------------------------
    # CREATE INTERNAL TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    print_section(
        "CREATING TUNING TRAIN / VALIDATION SPLIT"
    )

    indices = np.arange(
        len(y)
    )

    train_indices, validation_indices = (
        train_test_split(

            indices,

            test_size=args.validation_size,

            random_state=2026,

            stratify=y
        )
    )

    X_training_raw = X_raw.iloc[
        train_indices
    ]

    X_validation_raw = X_raw.iloc[
        validation_indices
    ]

    y_training = y[
        train_indices
    ]

    y_validation = y[
        validation_indices
    ]

    print(
        "Tuning training samples:",
        f"{len(y_training):,}"
    )

    print(
        "Tuning validation samples:",
        f"{len(y_validation):,}"
    )

    # --------------------------------------------------------
    # PREPROCESS WITHOUT VALIDATION LEAKAGE
    # --------------------------------------------------------

    print_section(
        "PREPROCESSING TUNING DATA"
    )

    tuning_preprocessor = clone(
        original_preprocessor
    )

    X_training = (
        tuning_preprocessor.fit_transform(
            X_training_raw
        )
    )

    X_validation = (
        tuning_preprocessor.transform(
            X_validation_raw
        )
    )

    print(
        "Prepared feature count:",
        X_training.shape[1]
    )

    # --------------------------------------------------------
    # OPTUNA FINE-TUNING
    # --------------------------------------------------------

    study = tune_lightgbm(

        X_training,
        y_training,

        X_validation,
        y_validation,

        args
    )

    best_trial = study.best_trial

    # Save all trials.
    study.trials_dataframe().to_csv(
        TUNING_TRIALS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # TRAIN BEST CONFIGURATION ON FULL TRAIN DATA
    # --------------------------------------------------------

    tuned_model, tuned_preprocessor = (
        train_final_tuned_model(

            X_raw,
            y,

            original_preprocessor,

            best_trial,
            args
        )
    )

    # --------------------------------------------------------
    # COMPARE ON HELD-OUT TEST DATA
    # --------------------------------------------------------

    comparison_results = None

    if not args.skip_test:

        comparison_results = compare_models(

            tuned_model,

            tuned_preprocessor,

            original_preprocessor
        )

    # --------------------------------------------------------
    # SAVE RESEARCH SUMMARY
    # --------------------------------------------------------

    summary = {

        "feature_count": len(feature_names),

        "balanced_training": True,

        "training_samples": int(len(y)),

        "tuning_training_samples": int(
            len(y_training)
        ),

        "tuning_validation_samples": int(
            len(y_validation)
        ),

        "optuna_trials": int(
            args.trials
        ),

        "optimization_metric": "validation_f1",

        "best_validation_f1": float(
            best_trial.value
        ),

        "best_validation_metrics": (
            best_trial.user_attrs[
                "validation_metrics"
            ]
        ),

        "best_number_of_trees": int(
            best_trial.user_attrs[
                "best_iteration"
            ]
        ),

        "best_hyperparameters": (
            best_trial.params
        ),

        "raw_feature_names": (
            feature_names
        ),

        "tuned_model_file": str(
            TUNED_MODEL_FILE
        ),

        "tuned_preprocessor_file": str(
            TUNED_PREPROCESSOR_FILE
        ),

        "test_comparison": (
            comparison_results
        ),

        "notes": (
            "Hyperparameters were selected using "
            "training-only validation data. "
            "The final test set was not used for tuning. "
            "The original model was preserved. "
            "The existing probability calibrator must not "
            "be used with this tuned model. "
            "Live feature collection still requires "
            "training-serving consistency."
        )
    }

    TUNING_SUMMARY_FILE.write_text(

        json.dumps(
            summary,
            indent=4
        ),

        encoding="utf-8"
    )

    print_section(
        "FINE-TUNING COMPLETED"
    )

    print(
        "Original model: UNCHANGED"
    )

    print(
        "New tuned model:",
        TUNED_MODEL_FILE
    )

    print(
        "New tuned preprocessor:",
        TUNED_PREPROCESSOR_FILE
    )

    print(
        "Tuning trials:",
        TUNING_TRIALS_FILE
    )

    print(
        "Tuning summary:",
        TUNING_SUMMARY_FILE
    )

    print(
        "Model comparison:",
        COMPARISON_FILE
        if comparison_results is not None
        else "Not performed"
    )

    print()
    print(
        "NEXT STEP: Review original vs tuned metrics "
        "before testing the tuned model live."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
