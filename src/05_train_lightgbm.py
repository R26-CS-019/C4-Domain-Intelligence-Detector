from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from sklearn.model_selection import train_test_split


# ============================================================
# 05 - TRAIN LIGHTGBM PHISHING DETECTION MODEL
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\SLIIT\final reserch chamath"
)

PROCESSED_DIR = (
    PROJECT_ROOT / "processed"
)

MODELS_DIR = (
    PROJECT_ROOT / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results"
)


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


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_FILE = (
    MODELS_DIR
    / "lightgbm_phishing_model.joblib"
)

MODEL_NATIVE_FILE = (
    MODELS_DIR
    / "lightgbm_phishing_model.txt"
)

METRICS_FILE = (
    RESULTS_DIR
    / "lightgbm_metrics.csv"
)

CONFUSION_MATRIX_FILE = (
    RESULTS_DIR
    / "lightgbm_confusion_matrix.csv"
)

CLASSIFICATION_REPORT_FILE = (
    RESULTS_DIR
    / "lightgbm_classification_report.csv"
)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "lightgbm_test_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    RESULTS_DIR
    / "lightgbm_feature_importance.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "lightgbm_training_summary.json"
)

FEATURE_NAMES_FILE = (
    RESULTS_DIR
    / "prepared_feature_names.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

VALIDATION_SIZE = 0.10


# ============================================================
# HELPER
# ============================================================

def print_section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories() -> None:

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


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
        Y_TEST_FILE
    ]

    for file in required_files:

        if not file.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )

    print(
        "Loading X_train..."
    )

    X_train = sparse.load_npz(
        X_TRAIN_FILE
    ).tocsr()

    print(
        "Loading X_test..."
    )

    X_test = sparse.load_npz(
        X_TEST_FILE
    ).tocsr()

    print(
        "Loading y_train..."
    )

    y_train = np.load(
        Y_TRAIN_FILE
    )

    print(
        "Loading y_test..."
    )

    y_test = np.load(
        Y_TEST_FILE
    )

    print()
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

    print()

    unique, counts = np.unique(
        y_train,
        return_counts=True
    )

    print(
        "Training class distribution:"
    )

    for label, count in zip(
        unique,
        counts
    ):

        class_name = (
            "Benign"
            if label == 0
            else "Phishing"
        )

        print(
            f"  {class_name}: "
            f"{count:,}"
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# CREATE VALIDATION SPLIT
# ============================================================

def create_validation_split(
    X_train,
    y_train
):

    print_section(
        "CREATING INTERNAL VALIDATION SET"
    )

    (
        X_model_train,
        X_validation,
        y_model_train,
        y_validation
    ) = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    print(
        f"Model training rows: "
        f"{X_model_train.shape[0]:,}"
    )

    print(
        f"Validation rows:     "
        f"{X_validation.shape[0]:,}"
    )

    print()
    print(
        "The official test set remains "
        "untouched during training."
    )

    return (
        X_model_train,
        X_validation,
        y_model_train,
        y_validation
    )


# ============================================================
# CREATE LIGHTGBM MODEL
# ============================================================

def create_model():

    print_section(
        "LIGHTGBM CONFIGURATION"
    )

    model = lgb.LGBMClassifier(

        # Binary phishing classification
        objective="binary",

        # Number of boosting trees.
        # Early stopping can stop before 1000.
        n_estimators=1000,

        # Smaller learning rate generally
        # provides more stable learning.
        learning_rate=0.05,

        # Standard starting point for tree
        # complexity.
        num_leaves=31,

        max_depth=-1,

        min_child_samples=20,

        # Feature sampling
        colsample_bytree=0.8,

        # Row sampling
        subsample=0.8,

        subsample_freq=1,

        # Regularization
        reg_alpha=0.0,

        reg_lambda=1.0,

        # Dataset is already 50/50 balanced.
        # No class_weight is required.

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1
    )

    print(
        "Model: LightGBM LGBMClassifier"
    )

    print(
        "Objective: Binary classification"
    )

    print(
        "Learning rate: 0.05"
    )

    print(
        "Maximum estimators: 1000"
    )

    print(
        "Leaves: 31"
    )

    print(
        "Early stopping: 50 rounds"
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    X_model_train,
    y_model_train,
    X_validation,
    y_validation
):

    print_section(
        "TRAINING LIGHTGBM"
    )

    print(
        "Training started..."
    )

    start_time = time.time()

    model.fit(
        X_model_train,
        y_model_train,

        eval_set=[
            (
                X_validation,
                y_validation
            )
        ],

        eval_metric=[
            "binary_logloss",
            "auc"
        ],

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=True
            ),

            lgb.log_evaluation(
                period=50
            )
        ]
    )

    training_time = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Training completed in "
        f"{training_time:.2f} seconds."
    )

    if hasattr(
        model,
        "best_iteration_"
    ):

        print(
            f"Best iteration: "
            f"{model.best_iteration_}"
        )

    return training_time


# ============================================================
# TEST MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    training_time
):

    print_section(
        "EVALUATING LIGHTGBM ON TEST SET"
    )

    start_time = time.time()

    y_pred = model.predict(
        X_test
    )

    y_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    prediction_time = (
        time.time()
        - start_time
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        ).ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    print()
    print(
        f"Accuracy:    "
        f"{accuracy:.6f}"
    )

    print(
        f"Precision:   "
        f"{precision:.6f}"
    )

    print(
        f"Recall:      "
        f"{recall:.6f}"
    )

    print(
        f"F1-score:    "
        f"{f1:.6f}"
    )

    print(
        f"ROC-AUC:     "
        f"{roc_auc:.6f}"
    )

    print(
        f"Specificity: "
        f"{specificity:.6f}"
    )

    print()

    print(
        "CONFUSION MATRIX"
    )

    print(
        f"True Negative:  "
        f"{tn:,}"
    )

    print(
        f"False Positive: "
        f"{fp:,}"
    )

    print(
        f"False Negative: "
        f"{fn:,}"
    )

    print(
        f"True Positive:  "
        f"{tp:,}"
    )

    print()

    print(
        f"Prediction time: "
        f"{prediction_time:.2f} seconds"
    )

    metrics = {

        "model":
            "LightGBM",

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

        "specificity":
            float(specificity),

        "false_positive_rate":
            float(false_positive_rate),

        "false_negative_rate":
            float(false_negative_rate),

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_positive":
            int(tp),

        "training_time_seconds":
            float(training_time),

        "prediction_time_seconds":
            float(prediction_time)
    }

    return (
        metrics,
        y_pred,
        y_probability
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics: dict
) -> None:

    print_section(
        "SAVING MODEL METRICS"
    )

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    print(
        f"Metrics saved to:\n"
        f"{METRICS_FILE}"
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    metrics: dict
) -> None:

    confusion_df = pd.DataFrame(
        {
            "actual_class": [
                "Benign",
                "Benign",
                "Phishing",
                "Phishing"
            ],

            "predicted_class": [
                "Benign",
                "Phishing",
                "Benign",
                "Phishing"
            ],

            "count": [
                metrics[
                    "true_negative"
                ],
                metrics[
                    "false_positive"
                ],
                metrics[
                    "false_negative"
                ],
                metrics[
                    "true_positive"
                ]
            ]
        }
    )

    confusion_df.to_csv(
        CONFUSION_MATRIX_FILE,
        index=False
    )


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    y_test,
    y_pred
) -> None:

    report = classification_report(
        y_test,
        y_pred,

        target_names=[
            "Benign",
            "Phishing"
        ],

        output_dict=True,

        zero_division=0
    )

    report_df = (
        pd.DataFrame(
            report
        )
        .transpose()
    )

    report_df.to_csv(
        CLASSIFICATION_REPORT_FILE
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

def save_predictions(
    y_test,
    y_pred,
    y_probability
) -> None:

    prediction_df = pd.DataFrame(
        {
            "actual_label":
                y_test,

            "predicted_label":
                y_pred,

            "phishing_probability":
                y_probability,

            "correct_prediction":
                (
                    y_test
                    == y_pred
                )
                .astype(int)
        }
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model
) -> None:

    print_section(
        "FEATURE IMPORTANCE"
    )

    if FEATURE_NAMES_FILE.exists():

        feature_names_df = pd.read_csv(
            FEATURE_NAMES_FILE
        )

        feature_names = (
            feature_names_df[
                "feature_name"
            ]
            .tolist()
        )

    else:

        feature_names = [
            f"feature_{i}"
            for i in range(
                len(
                    model.feature_importances_
                )
            )
        ]

    importances = (
        model.feature_importances_
    )

    if len(feature_names) != len(
        importances
    ):

        feature_names = [
            f"feature_{i}"
            for i in range(
                len(importances)
            )
        ]

    importance_df = pd.DataFrame(
        {
            "feature":
                feature_names,

            "importance":
                importances
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    importance_df[
        "rank"
    ] = np.arange(
        1,
        len(importance_df) + 1
    )

    importance_df = importance_df[
        [
            "rank",
            "feature",
            "importance"
        ]
    ]

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    print(
        "Top 20 features:"
    )

    print()

    print(
        importance_df
        .head(20)
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Full importance list saved to:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model
) -> None:

    print_section(
        "SAVING TRAINED MODEL"
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        f"Joblib model saved to:\n"
        f"{MODEL_FILE}"
    )

    # Also save native LightGBM model
    booster = model.booster_

    booster.save_model(
        str(
            MODEL_NATIVE_FILE
        )
    )

    print()
    print(
        f"Native LightGBM model saved to:\n"
        f"{MODEL_NATIVE_FILE}"
    )


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

def save_summary(
    model,
    metrics,
    X_model_train,
    X_validation,
    X_test
) -> None:

    summary = {

        "model":
            "LightGBM LGBMClassifier",

        "objective":
            "binary",

        "random_state":
            RANDOM_STATE,

        "training_rows":
            int(
                X_model_train.shape[0]
            ),

        "validation_rows":
            int(
                X_validation.shape[0]
            ),

        "test_rows":
            int(
                X_test.shape[0]
            ),

        "feature_count":
            int(
                X_test.shape[1]
            ),

        "best_iteration":
            int(
                model.best_iteration_
            )
            if model.best_iteration_
            else None,

        "parameters":
            model.get_params(),

        "metrics":
            metrics
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            default=str
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 80)

    print(
        "05 - LIGHTGBM DOMAIN INTELLIGENCE "
        "PHISHING DETECTION"
    )

    print("=" * 80)

    overall_start = time.time()

    create_directories()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()

    # --------------------------------------------------------
    # INTERNAL VALIDATION SPLIT
    # --------------------------------------------------------

    (
        X_model_train,
        X_validation,
        y_model_train,
        y_validation
    ) = create_validation_split(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    training_time = train_model(
        model,
        X_model_train,
        y_model_train,
        X_validation,
        y_validation
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    (
        metrics,
        y_pred,
        y_probability
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        training_time
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_metrics(
        metrics
    )

    save_confusion_matrix(
        metrics
    )

    save_classification_report(
        y_test,
        y_pred
    )

    save_predictions(
        y_test,
        y_pred,
        y_probability
    )

    save_feature_importance(
        model
    )

    save_model(
        model
    )

    save_summary(
        model,
        metrics,
        X_model_train,
        X_validation,
        X_test
    )

    total_time = (
        time.time()
        - overall_start
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_section(
        "LIGHTGBM TRAINING COMPLETED"
    )

    print(
        f"Accuracy:  "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score:  "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{metrics['roc_auc']:.4f}"
    )

    print()

    print(
        f"Total runtime: "
        f"{total_time / 60:.2f} minutes"
    )

    print()

    print(
        "Trained model:"
    )

    print(
        MODEL_FILE
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "06_evaluate_lightgbm.py"
    )


if __name__ == "__main__":
    main()