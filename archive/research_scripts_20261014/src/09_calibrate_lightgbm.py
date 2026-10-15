from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold


# ============================================================
# 09 - CALIBRATE LIGHTGBM MODEL
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\SLIIT\final reserch chamath"
)

PROCESSED_DIR = PROJECT_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# INPUT FILES
# ============================================================

X_TRAIN_FILE = (
    PROCESSED_DIR
    / "X_train_prepared.npz"
)

X_TEST_FILE = (
    PROCESSED_DIR
    / "X_test_prepared.npz"
)

Y_TRAIN_FILE = (
    PROCESSED_DIR
    / "y_train.npy"
)

Y_TEST_FILE = (
    PROCESSED_DIR
    / "y_test.npy"
)

BASE_MODEL_FILE = (
    MODELS_DIR
    / "lightgbm_phishing_model.joblib"
)


# ============================================================
# OUTPUT FILES
# ============================================================

CALIBRATOR_FILE = (
    MODELS_DIR
    / "lightgbm_probability_calibrator.joblib"
)

COMPARISON_FILE = (
    RESULTS_DIR
    / "lightgbm_calibration_comparison.csv"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "lightgbm_calibrated_predictions.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "lightgbm_calibration_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_FOLDS = 5


# ============================================================
# HELPERS
# ============================================================

def print_section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print_section(
        "LOADING PREPARED DATA"
    )

    required_files = [
        X_TRAIN_FILE,
        X_TEST_FILE,
        Y_TRAIN_FILE,
        Y_TEST_FILE,
        BASE_MODEL_FILE
    ]

    for file in required_files:

        if not file.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )

    X_train = sparse.load_npz(
        X_TRAIN_FILE
    ).tocsr()

    X_test = sparse.load_npz(
        X_TEST_FILE
    ).tocsr()

    y_train = np.load(
        Y_TRAIN_FILE
    )

    y_test = np.load(
        Y_TEST_FILE
    )

    base_model = joblib.load(
        BASE_MODEL_FILE
    )

    print(
        f"X_train shape: {X_train.shape}"
    )

    print(
        f"X_test shape:  {X_test.shape}"
    )

    print(
        f"y_train shape: {y_train.shape}"
    )

    print(
        f"y_test shape:  {y_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        base_model
    )


# ============================================================
# CREATE OUT-OF-FOLD LIGHTGBM PROBABILITIES
# ============================================================

def create_oof_probabilities(
    X_train,
    y_train
):

    print_section(
        "CREATING OUT-OF-FOLD LIGHTGBM PROBABILITIES"
    )

    oof_probability = np.zeros(
        len(y_train),
        dtype=float
    )

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for fold_number, (
        train_index,
        valid_index
    ) in enumerate(
        splitter.split(
            X_train,
            y_train
        ),
        start=1
    ):

        print()
        print(
            f"Fold {fold_number}/{N_FOLDS}"
        )

        X_fold_train = (
            X_train[
                train_index
            ]
        )

        X_fold_valid = (
            X_train[
                valid_index
            ]
        )

        y_fold_train = (
            y_train[
                train_index
            ]
        )

        y_fold_valid = (
            y_train[
                valid_index
            ]
        )

        model = lgb.LGBMClassifier(
            objective="binary",

            n_estimators=1000,

            learning_rate=0.05,

            num_leaves=31,

            max_depth=-1,

            min_child_samples=20,

            colsample_bytree=0.8,

            subsample=0.8,

            subsample_freq=1,

            reg_alpha=0.0,

            reg_lambda=1.0,

            random_state=(
                RANDOM_STATE
                + fold_number
            ),

            n_jobs=-1,

            verbosity=-1
        )

        start = time.time()

        model.fit(
            X_fold_train,
            y_fold_train
        )

        probability = (
            model.predict_proba(
                X_fold_valid
            )[:, 1]
        )

        oof_probability[
            valid_index
        ] = probability

        fold_auc = roc_auc_score(
            y_fold_valid,
            probability
        )

        elapsed = (
            time.time()
            - start
        )

        print(
            f"Fold ROC-AUC: "
            f"{fold_auc:.6f}"
        )

        print(
            f"Runtime: "
            f"{elapsed:.2f} sec"
        )

    overall_auc = roc_auc_score(
        y_train,
        oof_probability
    )

    print()
    print(
        f"Overall OOF ROC-AUC: "
        f"{overall_auc:.6f}"
    )

    return oof_probability


# ============================================================
# TRAIN ISOTONIC CALIBRATOR
# ============================================================

def train_calibrator(
    oof_probability,
    y_train
):

    print_section(
        "TRAINING ISOTONIC PROBABILITY CALIBRATOR"
    )

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip"
    )

    calibrator.fit(
        oof_probability,
        y_train
    )

    joblib.dump(
        calibrator,
        CALIBRATOR_FILE
    )

    print(
        "Calibration model trained successfully."
    )

    print()

    print(
        f"Calibrator saved to:\n"
        f"{CALIBRATOR_FILE}"
    )

    return calibrator


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    model_name: str,
    y_true,
    probability
):

    prediction = (
        probability
        >= 0.50
    ).astype(
        np.int8
    )

    accuracy = accuracy_score(
        y_true,
        prediction
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probability
    )

    pr_auc = average_precision_score(
        y_true,
        probability
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1]
        ).ravel()
    )

    return {
        "model":
            model_name,

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_positive":
            int(tp)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "09 - LIGHTGBM PROBABILITY CALIBRATION"
    )
    print("=" * 80)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    overall_start = time.time()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        base_model
    ) = load_data()

    # --------------------------------------------------------
    # OOF PROBABILITIES
    # --------------------------------------------------------

    oof_probability = (
        create_oof_probabilities(
            X_train,
            y_train
        )
    )

    # --------------------------------------------------------
    # TRAIN CALIBRATOR
    # --------------------------------------------------------

    calibrator = train_calibrator(
        oof_probability,
        y_train
    )

    # --------------------------------------------------------
    # ORIGINAL TEST PROBABILITY
    # --------------------------------------------------------

    print_section(
        "GENERATING TEST PREDICTIONS"
    )

    raw_probability = (
        base_model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # CALIBRATED TEST PROBABILITY
    # --------------------------------------------------------

    calibrated_probability = (
        calibrator.predict(
            raw_probability
        )
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    original_metrics = (
        calculate_metrics(
            "Original LightGBM",
            y_test,
            raw_probability
        )
    )

    calibrated_metrics = (
        calculate_metrics(
            "Calibrated LightGBM",
            y_test,
            calibrated_probability
        )
    )

    comparison_df = pd.DataFrame(
        [
            original_metrics,
            calibrated_metrics
        ]
    )

    comparison_df.to_csv(
        COMPARISON_FILE,
        index=False
    )

    print_section(
        "ORIGINAL VS CALIBRATED LIGHTGBM"
    )

    print(
        comparison_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc",
                "false_positive",
                "false_negative"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "actual_label":
                y_test,

            "raw_probability":
                raw_probability,

            "calibrated_probability":
                calibrated_probability,

            "original_prediction":
                (
                    raw_probability
                    >= 0.5
                ).astype(
                    np.int8
                ),

            "calibrated_prediction":
                (
                    calibrated_probability
                    >= 0.5
                ).astype(
                    np.int8
                )
        }
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # IMPROVEMENTS
    # --------------------------------------------------------

    accuracy_change = (
        calibrated_metrics[
            "accuracy"
        ]
        - original_metrics[
            "accuracy"
        ]
    )

    recall_change = (
        calibrated_metrics[
            "recall"
        ]
        - original_metrics[
            "recall"
        ]
    )

    f1_change = (
        calibrated_metrics[
            "f1_score"
        ]
        - original_metrics[
            "f1_score"
        ]
    )

    false_negative_reduction = (
        original_metrics[
            "false_negative"
        ]
        - calibrated_metrics[
            "false_negative"
        ]
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "calibration_method":
            "Isotonic Regression",

        "oof_folds":
            N_FOLDS,

        "classification_threshold":
            0.5,

        "original_lightgbm":
            original_metrics,

        "calibrated_lightgbm":
            calibrated_metrics,

        "changes": {
            "accuracy_difference":
                float(
                    accuracy_change
                ),

            "recall_difference":
                float(
                    recall_change
                ),

            "f1_difference":
                float(
                    f1_change
                ),

            "false_negative_reduction":
                int(
                    false_negative_reduction
                )
        }
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    total_runtime = (
        time.time()
        - overall_start
    )

    print_section(
        "CALIBRATION COMPLETED"
    )

    print(
        "ORIGINAL LIGHTGBM"
    )

    print(
        f"Accuracy:  "
        f"{original_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{original_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{original_metrics['recall']:.4f}"
    )

    print(
        f"F1-score:  "
        f"{original_metrics['f1_score']:.4f}"
    )

    print()

    print(
        "CALIBRATED LIGHTGBM"
    )

    print(
        f"Accuracy:  "
        f"{calibrated_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{calibrated_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{calibrated_metrics['recall']:.4f}"
    )

    print(
        f"F1-score:  "
        f"{calibrated_metrics['f1_score']:.4f}"
    )

    print()

    print(
        "IMPROVEMENT"
    )

    print(
        f"Accuracy change: "
        f"{accuracy_change:+.6f}"
    )

    print(
        f"Recall change:   "
        f"{recall_change:+.6f}"
    )

    print(
        f"F1 change:       "
        f"{f1_change:+.6f}"
    )

    print(
        f"Phishing cases no longer missed: "
        f"{false_negative_reduction:,}"
    )

    print()

    print(
        f"Runtime: "
        f"{total_runtime / 60:.2f} minutes"
    )

    print()

    print(
        "Final calibrator:"
    )

    print(
        CALIBRATOR_FILE
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "Update 08_test_domain.py "
        "to use calibrated probabilities."
    )


if __name__ == "__main__":
    main()