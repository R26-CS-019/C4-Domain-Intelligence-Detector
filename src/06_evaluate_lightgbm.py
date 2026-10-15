from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)


# ============================================================
# 06 - EVALUATE LIGHTGBM MODEL
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

PLOTS_DIR = (
    RESULTS_DIR / "plots"
)


# ============================================================
# INPUT FILES
# ============================================================

MODEL_FILE = (
    MODELS_DIR
    / "lightgbm_phishing_model.joblib"
)

X_TEST_FILE = (
    PROCESSED_DIR
    / "X_test_prepared.npz"
)

Y_TEST_FILE = (
    PROCESSED_DIR
    / "y_test.npy"
)

FEATURE_IMPORTANCE_FILE = (
    RESULTS_DIR
    / "lightgbm_feature_importance.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

FINAL_METRICS_FILE = (
    RESULTS_DIR
    / "lightgbm_final_evaluation.csv"
)

CLASSIFICATION_REPORT_FILE = (
    RESULTS_DIR
    / "lightgbm_final_classification_report.csv"
)

CONFUSION_MATRIX_FILE = (
    RESULTS_DIR
    / "lightgbm_final_confusion_matrix.csv"
)

THRESHOLD_ANALYSIS_FILE = (
    RESULTS_DIR
    / "lightgbm_threshold_analysis.csv"
)

EVALUATION_SUMMARY_FILE = (
    RESULTS_DIR
    / "lightgbm_evaluation_summary.json"
)

CONFUSION_MATRIX_PLOT = (
    PLOTS_DIR
    / "lightgbm_confusion_matrix.png"
)

ROC_CURVE_PLOT = (
    PLOTS_DIR
    / "lightgbm_roc_curve.png"
)

PR_CURVE_PLOT = (
    PLOTS_DIR
    / "lightgbm_precision_recall_curve.png"
)

PROBABILITY_DISTRIBUTION_PLOT = (
    PLOTS_DIR
    / "lightgbm_probability_distribution.png"
)

FEATURE_IMPORTANCE_PLOT = (
    PLOTS_DIR
    / "lightgbm_top20_feature_importance.png"
)

METRIC_SUMMARY_PLOT = (
    PLOTS_DIR
    / "lightgbm_metric_summary.png"
)


# ============================================================
# HELPER
# ============================================================

def print_section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD MODEL AND TEST DATA
# ============================================================

def load_data():

    print_section(
        "LOADING TRAINED MODEL AND TEST DATA"
    )

    required_files = [
        MODEL_FILE,
        X_TEST_FILE,
        Y_TEST_FILE
    ]

    for file in required_files:

        if not file.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )

    print(
        "Loading LightGBM model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Loading X_test..."
    )

    X_test = (
        sparse.load_npz(
            X_TEST_FILE
        )
        .tocsr()
    )

    print(
        "Loading y_test..."
    )

    y_test = np.load(
        Y_TEST_FILE
    )

    print()
    print(
        f"X_test shape: {X_test.shape}"
    )

    print(
        f"y_test shape: {y_test.shape}"
    )

    return (
        model,
        X_test,
        y_test
    )


# ============================================================
# PREDICTIONS
# ============================================================

def make_predictions(
    model,
    X_test
):

    print_section(
        "GENERATING TEST PREDICTIONS"
    )

    y_pred = model.predict(
        X_test
    )

    y_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "Predictions generated successfully."
    )

    return (
        y_pred,
        y_probability
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_test,
    y_pred,
    y_probability
):

    print_section(
        "FINAL TEST METRICS"
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

    pr_auc = average_precision_score(
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
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    metrics = {

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
            int(tp)
    }

    print(
        f"Accuracy:      {accuracy:.6f}"
    )

    print(
        f"Precision:     {precision:.6f}"
    )

    print(
        f"Recall:        {recall:.6f}"
    )

    print(
        f"F1-score:      {f1:.6f}"
    )

    print(
        f"ROC-AUC:       {roc_auc:.6f}"
    )

    print(
        f"PR-AUC:        {pr_auc:.6f}"
    )

    print(
        f"Specificity:   {specificity:.6f}"
    )

    print()

    print(
        "CONFUSION MATRIX"
    )

    print(
        f"True Negative:  {tn:,}"
    )

    print(
        f"False Positive: {fp:,}"
    )

    print(
        f"False Negative: {fn:,}"
    )

    print(
        f"True Positive:  {tp:,}"
    )

    return metrics


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics: dict
) -> None:

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        FINAL_METRICS_FILE,
        index=False
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    y_test,
    y_pred
) -> None:

    print_section(
        "CLASSIFICATION REPORT"
    )

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

    print(
        report_df.to_string()
    )

    report_df.to_csv(
        CLASSIFICATION_REPORT_FILE
    )


# ============================================================
# CONFUSION MATRIX DATA
# ============================================================

def save_confusion_matrix(
    y_test,
    y_pred
) -> np.ndarray:

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Benign",
            "Actual Phishing"
        ],
        columns=[
            "Predicted Benign",
            "Predicted Phishing"
        ]
    )

    cm_df.to_csv(
        CONFUSION_MATRIX_FILE
    )

    return cm


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def plot_confusion_matrix(
    y_test,
    y_pred
) -> None:

    print(
        "Creating confusion matrix plot..."
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=[
            "Benign",
            "Phishing"
        ],
        values_format=",d",
        ax=ax
    )

    ax.set_title(
        "LightGBM Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_MATRIX_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(
    y_test,
    y_probability
) -> float:

    print(
        "Creating ROC curve..."
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"LightGBM (AUC = {auc:.4f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve - LightGBM Phishing Detection"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        ROC_CURVE_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return auc


# ============================================================
# PRECISION RECALL CURVE
# ============================================================

def plot_precision_recall_curve(
    y_test,
    y_probability
) -> float:

    print(
        "Creating precision-recall curve..."
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            y_probability
        )
    )

    average_precision = (
        average_precision_score(
            y_test,
            y_probability
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=(
            f"LightGBM "
            f"(AP = {average_precision:.4f})"
        )
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Precision-Recall Curve - LightGBM"
    )

    ax.legend(
        loc="lower left"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        PR_CURVE_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return average_precision


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

def plot_probability_distribution(
    y_test,
    y_probability
) -> None:

    print(
        "Creating probability distribution plot..."
    )

    benign_probabilities = (
        y_probability[
            y_test == 0
        ]
    )

    phishing_probabilities = (
        y_probability[
            y_test == 1
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        benign_probabilities,
        bins=50,
        alpha=0.6,
        label="Benign"
    )

    ax.hist(
        phishing_probabilities,
        bins=50,
        alpha=0.6,
        label="Phishing"
    )

    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=2,
        label="Decision Threshold = 0.5"
    )

    ax.set_xlabel(
        "Predicted Phishing Probability"
    )

    ax.set_ylabel(
        "Number of Domains"
    )

    ax.set_title(
        "LightGBM Prediction Probability Distribution"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PROBABILITY_DISTRIBUTION_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# METRIC SUMMARY CHART
# ============================================================

def plot_metric_summary(
    metrics: dict
) -> None:

    print(
        "Creating metric summary chart..."
    )

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "PR-AUC",
        "Specificity"
    ]

    metric_values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["specificity"]
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        metric_names,
        metric_values
    )

    ax.set_ylim(
        0.0,
        1.05
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "LightGBM Evaluation Metrics"
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    for bar, value in zip(
        bars,
        metric_values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 0.01,

            f"{value:.3f}",

            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    fig.savefig(
        METRIC_SUMMARY_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================

def plot_feature_importance() -> None:

    print(
        "Creating feature importance plot..."
    )

    if not FEATURE_IMPORTANCE_FILE.exists():

        print(
            "Feature importance CSV not found."
        )

        return

    importance_df = pd.read_csv(
        FEATURE_IMPORTANCE_FILE
    )

    top20 = (
        importance_df
        .head(20)
        .copy()
    )

    top20["feature"] = (
        top20["feature"]
        .str.replace(
            "numeric__",
            "",
            regex=False
        )
        .str.replace(
            "categorical__",
            "",
            regex=False
        )
    )

    top20 = top20.sort_values(
        by="importance",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.barh(
        top20["feature"],
        top20["importance"]
    )

    ax.set_xlabel(
        "Feature Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Top 20 LightGBM Domain Intelligence Features"
    )

    fig.tight_layout()

    fig.savefig(
        FEATURE_IMPORTANCE_PLOT,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    y_test,
    y_probability
) -> pd.DataFrame:

    print_section(
        "DECISION THRESHOLD ANALYSIS"
    )

    rows = []

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    for threshold in thresholds:

        y_pred_threshold = (
            y_probability
            >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            y_pred_threshold
        )

        precision = precision_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        tn, fp, fn, tp = (
            confusion_matrix(
                y_test,
                y_pred_threshold,
                labels=[0, 1]
            ).ravel()
        )

        rows.append(
            {
                "threshold":
                    round(
                        float(threshold),
                        2
                    ),

                "accuracy":
                    float(accuracy),

                "precision":
                    float(precision),

                "recall":
                    float(recall),

                "f1_score":
                    float(f1),

                "false_positive":
                    int(fp),

                "false_negative":
                    int(fn),

                "true_positive":
                    int(tp),

                "true_negative":
                    int(tn)
            }
        )

    threshold_df = pd.DataFrame(
        rows
    )

    threshold_df.to_csv(
        THRESHOLD_ANALYSIS_FILE,
        index=False
    )

    best_f1_row = (
        threshold_df.loc[
            threshold_df[
                "f1_score"
            ].idxmax()
        ]
    )

    print(
        "Best threshold by F1-score:"
    )

    print(
        f"Threshold: "
        f"{best_f1_row['threshold']:.2f}"
    )

    print(
        f"F1-score: "
        f"{best_f1_row['f1_score']:.6f}"
    )

    print(
        f"Precision: "
        f"{best_f1_row['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{best_f1_row['recall']:.6f}"
    )

    return threshold_df


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    metrics: dict,
    threshold_df: pd.DataFrame
) -> None:

    best_f1_row = (
        threshold_df.loc[
            threshold_df[
                "f1_score"
            ].idxmax()
        ]
    )

    summary = {

        "model":
            "LightGBM",

        "test_samples":
            int(
                metrics[
                    "true_negative"
                ]
                + metrics[
                    "false_positive"
                ]
                + metrics[
                    "false_negative"
                ]
                + metrics[
                    "true_positive"
                ]
            ),

        "metrics":
            metrics,

        "default_threshold":
            0.5,

        "best_threshold_by_f1":
            float(
                best_f1_row[
                    "threshold"
                ]
            ),

        "best_threshold_f1_score":
            float(
                best_f1_row[
                    "f1_score"
                ]
            ),

        "best_threshold_precision":
            float(
                best_f1_row[
                    "precision"
                ]
            ),

        "best_threshold_recall":
            float(
                best_f1_row[
                    "recall"
                ]
            ),

        "plots": {
            "confusion_matrix":
                str(
                    CONFUSION_MATRIX_PLOT
                ),

            "roc_curve":
                str(
                    ROC_CURVE_PLOT
                ),

            "precision_recall_curve":
                str(
                    PR_CURVE_PLOT
                ),

            "probability_distribution":
                str(
                    PROBABILITY_DISTRIBUTION_PLOT
                ),

            "feature_importance":
                str(
                    FEATURE_IMPORTANCE_PLOT
                ),

            "metric_summary":
                str(
                    METRIC_SUMMARY_PLOT
                )
        }
    }

    with EVALUATION_SUMMARY_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 80)

    print(
        "06 - FINAL LIGHTGBM MODEL EVALUATION"
    )

    print("=" * 80)

    create_directories()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        model,
        X_test,
        y_test
    ) = load_data()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    (
        y_pred,
        y_probability
    ) = make_predictions(
        model,
        X_test
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_probability
    )

    save_metrics(
        metrics
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    save_classification_report(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    save_confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    print_section(
        "GENERATING THESIS-READY PLOTS"
    )

    plot_confusion_matrix(
        y_test,
        y_pred
    )

    plot_roc_curve(
        y_test,
        y_probability
    )

    plot_precision_recall_curve(
        y_test,
        y_probability
    )

    plot_probability_distribution(
        y_test,
        y_probability
    )

    plot_metric_summary(
        metrics
    )

    plot_feature_importance()

    # --------------------------------------------------------
    # THRESHOLD ANALYSIS
    # --------------------------------------------------------

    threshold_df = threshold_analysis(
        y_test,
        y_probability
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    save_summary(
        metrics,
        threshold_df
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_section(
        "EVALUATION COMPLETED SUCCESSFULLY"
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

    print(
        f"PR-AUC:    "
        f"{metrics['pr_auc']:.4f}"
    )

    print()

    print(
        "Plots saved in:"
    )

    print(
        PLOTS_DIR
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "07_feature_analysis.py"
    )


if __name__ == "__main__":
    main()