from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# ============================================================
# 04 - PREPARE BALANCED TRAINING DATA
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
# INPUT FILE
# ============================================================

INPUT_FILE = (
    PROCESSED_DIR
    / "domain_intelligence_full.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

BALANCED_FILE = (
    PROCESSED_DIR
    / "domain_intelligence_balanced.csv"
)

TRAIN_RAW_FILE = (
    PROCESSED_DIR
    / "train_balanced_raw.csv"
)

TEST_RAW_FILE = (
    PROCESSED_DIR
    / "test_balanced_raw.csv"
)

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

PREPROCESSOR_FILE = (
    MODELS_DIR
    / "training_preprocessor.joblib"
)

FEATURE_NAMES_FILE = (
    RESULTS_DIR
    / "prepared_feature_names.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "training_data_summary.json"
)

TRAIN_CLASS_FILE = (
    RESULTS_DIR
    / "train_class_distribution.csv"
)

TEST_CLASS_FILE = (
    RESULTS_DIR
    / "test_class_distribution.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_COLUMN = "label"


# ============================================================
# COLUMNS NOT USED AS ML FEATURES
# ============================================================

IDENTIFIER_COLUMNS = [
    "domain_name",
    "source",
    "class_name"
]


# ============================================================
# CONSTANT FEATURES FOUND DURING VALIDATION
# ============================================================

CONSTANT_COLUMNS = [
    "dns_available",
    "rdap_noc_count",
    "ip_from_mx_count",
    "ip_from_ns_count"
]


# ============================================================
# KNOWN CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_COLUMNS = [
    "tls_protocol",
    "tls_cipher"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def class_distribution(
    df: pd.DataFrame
) -> pd.DataFrame:

    distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )

    distribution["class_name"] = (
        distribution["label"].map(
            {
                0: "benign",
                1: "phishing"
            }
        )
    )

    distribution["percentage"] = (
        (
            distribution["count"]
            / len(df)
        )
        * 100
    ).round(4)

    return distribution[
        [
            "label",
            "class_name",
            "count",
            "percentage"
        ]
    ]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories() -> None:

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:

    print_section(
        "LOADING FULL PROCESSED DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Dataset not found:\n"
            f"{INPUT_FILE}"
        )

    print(
        f"Input file:\n{INPUT_FILE}"
    )

    print()
    print("Loading dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        f"Rows loaded:    {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns):,}"
    )

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            "label column not found."
        )

    labels = set(
        df[TARGET_COLUMN]
        .dropna()
        .astype(int)
        .unique()
    )

    if labels != {0, 1}:

        raise ValueError(
            f"Expected labels {{0, 1}}, "
            f"but found {labels}"
        )

    return df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "DUPLICATE REMOVAL"
    )

    before = len(df)

    # Remove fully duplicated rows
    df = df.drop_duplicates(
        keep="first"
    ).copy()

    after_row_duplicates = len(df)

    removed_rows = (
        before
        - after_row_duplicates
    )

    print(
        f"Exact duplicate rows removed: "
        f"{removed_rows:,}"
    )

    # --------------------------------------------------------
    # Remove duplicate domains
    # --------------------------------------------------------

    if "domain_name" in df.columns:

        duplicate_domains = int(
            df[
                "domain_name"
            ]
            .duplicated(
                keep="first"
            )
            .sum()
        )

        df = (
            df.drop_duplicates(
                subset=[
                    "domain_name"
                ],
                keep="first"
            )
            .reset_index(
                drop=True
            )
        )

        print(
            f"Duplicate domains removed: "
            f"{duplicate_domains:,}"
        )

    else:

        print(
            "domain_name column not found."
        )

    print(
        f"Rows remaining: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# CREATE BALANCED DATASET
# ============================================================

def create_balanced_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "CREATING BALANCED DATASET"
    )

    benign = df[
        df[TARGET_COLUMN] == 0
    ].copy()

    phishing = df[
        df[TARGET_COLUMN] == 1
    ].copy()

    benign_count = len(benign)
    phishing_count = len(phishing)

    print(
        f"Available benign:   "
        f"{benign_count:,}"
    )

    print(
        f"Available phishing: "
        f"{phishing_count:,}"
    )

    minority_count = min(
        benign_count,
        phishing_count
    )

    print()
    print(
        "Balancing method:"
    )

    print(
        "Random undersampling of "
        "majority class."
    )

    print(
        f"Samples per class: "
        f"{minority_count:,}"
    )

    # --------------------------------------------------------
    # BENIGN SAMPLE
    # --------------------------------------------------------

    benign_balanced = (
        benign.sample(
            n=minority_count,
            replace=False,
            random_state=RANDOM_STATE
        )
    )

    # --------------------------------------------------------
    # PHISHING SAMPLE
    # --------------------------------------------------------

    phishing_balanced = (
        phishing.sample(
            n=minority_count,
            replace=False,
            random_state=RANDOM_STATE
        )
        if phishing_count > minority_count
        else phishing.copy()
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    balanced = pd.concat(
        [
            benign_balanced,
            phishing_balanced
        ],
        ignore_index=True
    )

    # Shuffle rows reproducibly
    balanced = (
        balanced.sample(
            frac=1,
            random_state=RANDOM_STATE
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        class_distribution(
            balanced
        ).to_string(
            index=False
        )
    )

    print()

    print(
        f"Balanced dataset total: "
        f"{len(balanced):,}"
    )

    return balanced


# ============================================================
# SELECT ML FEATURES
# ============================================================

def create_feature_table(
    balanced: pd.DataFrame
):

    print_section(
        "SELECTING MACHINE LEARNING FEATURES"
    )

    y = (
        balanced[
            TARGET_COLUMN
        ]
        .astype(
            np.int8
        )
        .copy()
    )

    columns_to_remove = []

    # Remove identifiers
    for column in IDENTIFIER_COLUMNS:

        if column in balanced.columns:

            columns_to_remove.append(
                column
            )

    # Remove target from X
    columns_to_remove.append(
        TARGET_COLUMN
    )

    # Remove validated constant features
    for column in CONSTANT_COLUMNS:

        if column in balanced.columns:

            columns_to_remove.append(
                column
            )

    columns_to_remove = list(
        dict.fromkeys(
            columns_to_remove
        )
    )

    X = balanced.drop(
        columns=columns_to_remove,
        errors="ignore"
    ).copy()

    print(
        "Removed columns:"
    )

    for column in columns_to_remove:

        print(
            f"  - {column}"
        )

    print()

    print(
        f"Raw ML feature count: "
        f"{X.shape[1]}"
    )

    if TARGET_COLUMN in X.columns:

        raise RuntimeError(
            "Target leakage detected."
        )

    return X, y


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    balanced: pd.DataFrame
):

    print_section(
        "80 / 20 STRATIFIED TRAIN TEST SPLIT"
    )

    indices = np.arange(
        len(balanced)
    )

    train_indices, test_indices = (
        train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    X_train = (
        X.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    X_test = (
        X.iloc[
            test_indices
        ]
        .reset_index(
            drop=True
        )
    )

    y_train = (
        y.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    y_test = (
        y.iloc[
            test_indices
        ]
        .reset_index(
            drop=True
        )
    )

    train_raw = (
        balanced.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    test_raw = (
        balanced.iloc[
            test_indices
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples:  "
        f"{len(X_test):,}"
    )

    print()

    print(
        "TRAINING CLASS DISTRIBUTION"
    )

    train_distribution = (
        class_distribution(
            train_raw
        )
    )

    print(
        train_distribution
        .to_string(
            index=False
        )
    )

    print()

    print(
        "TEST CLASS DISTRIBUTION"
    )

    test_distribution = (
        class_distribution(
            test_raw
        )
    )

    print(
        test_distribution
        .to_string(
            index=False
        )
    )

    train_distribution.to_csv(
        TRAIN_CLASS_FILE,
        index=False
    )

    test_distribution.to_csv(
        TEST_CLASS_FILE,
        index=False
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_raw,
        test_raw
    )


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

def identify_feature_types(
    X_train: pd.DataFrame
):

    print_section(
        "IDENTIFYING FEATURE TYPES"
    )

    categorical_columns = [
        column
        for column
        in CATEGORICAL_COLUMNS
        if column in X_train.columns
    ]

    # Detect any additional string columns
    detected_string_columns = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    for column in detected_string_columns:

        if (
            column
            not in categorical_columns
        ):

            categorical_columns.append(
                column
            )

    numeric_columns = [
        column
        for column
        in X_train.columns
        if column
        not in categorical_columns
    ]

    print(
        f"Numeric features:     "
        f"{len(numeric_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    print()

    print(
        "Categorical columns:"
    )

    for column in categorical_columns:

        print(
            f"  - {column}"
        )

    return (
        numeric_columns,
        categorical_columns
    )


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str]
):

    print_section(
        "BUILDING PREPROCESSING PIPELINE"
    )

    # --------------------------------------------------------
    # NUMERIC
    #
    # Median imputation handles missing values.
    #
    # Missing indicators preserve information
    # about whether a value was originally absent.
    #
    # RobustScaler reduces sensitivity to large
    # outliers.
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),
            (
                "scaler",
                RobustScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    transformers = [
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        )
    ]

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True
    )

    print(
        "Numeric:"
    )

    print(
        "  Median imputation"
    )

    print(
        "  Missing-value indicators"
    )

    print(
        "  RobustScaler"
    )

    print()

    print(
        "Categorical:"
    )

    print(
        "  Missing category"
    )

    print(
        "  One-hot encoding"
    )

    return preprocessor


# ============================================================
# PREPROCESS TRAIN AND TEST DATA
# ============================================================

def preprocess_data(
    preprocessor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):

    print_section(
        "PREPROCESSING TRAINING DATA"
    )

    print(
        "Fitting preprocessing ONLY "
        "on training data..."
    )

    start = time.time()

    X_train_prepared = (
        preprocessor.fit_transform(
            X_train
        )
    )

    train_time = (
        time.time()
        - start
    )

    print(
        f"Training data transformed "
        f"in {train_time:.2f} seconds."
    )

    print()

    print(
        "Transforming test data..."
    )

    start = time.time()

    X_test_prepared = (
        preprocessor.transform(
            X_test
        )
    )

    test_time = (
        time.time()
        - start
    )

    print(
        f"Test data transformed "
        f"in {test_time:.2f} seconds."
    )

    # Convert to CSR sparse matrices
    if sparse.issparse(
        X_train_prepared
    ):

        X_train_prepared = (
            X_train_prepared.tocsr()
        )

    else:

        X_train_prepared = (
            sparse.csr_matrix(
                X_train_prepared
            )
        )

    if sparse.issparse(
        X_test_prepared
    ):

        X_test_prepared = (
            X_test_prepared.tocsr()
        )

    else:

        X_test_prepared = (
            sparse.csr_matrix(
                X_test_prepared
            )
        )

    print()

    print(
        f"X_train prepared shape: "
        f"{X_train_prepared.shape}"
    )

    print(
        f"X_test prepared shape:  "
        f"{X_test_prepared.shape}"
    )

    return (
        X_train_prepared,
        X_test_prepared
    )


# ============================================================
# SAVE FEATURE NAMES
# ============================================================

def save_feature_names(
    preprocessor
):

    print_section(
        "SAVING PREPARED FEATURE NAMES"
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    feature_df = pd.DataFrame(
        {
            "feature_index":
                range(
                    len(feature_names)
                ),

            "feature_name":
                feature_names
        }
    )

    feature_df.to_csv(
        FEATURE_NAMES_FILE,
        index=False
    )

    print(
        f"Final prepared feature count: "
        f"{len(feature_names)}"
    )

    return feature_names


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

def save_outputs(
    balanced,
    train_raw,
    test_raw,
    X_train_prepared,
    X_test_prepared,
    y_train,
    y_test,
    preprocessor
):

    print_section(
        "SAVING PREPARED DATA"
    )

    print(
        "Saving balanced dataset..."
    )

    balanced.to_csv(
        BALANCED_FILE,
        index=False
    )

    print(
        "Saving raw train dataset..."
    )

    train_raw.to_csv(
        TRAIN_RAW_FILE,
        index=False
    )

    print(
        "Saving raw test dataset..."
    )

    test_raw.to_csv(
        TEST_RAW_FILE,
        index=False
    )

    print(
        "Saving prepared training matrix..."
    )

    sparse.save_npz(
        X_TRAIN_FILE,
        X_train_prepared
    )

    print(
        "Saving prepared test matrix..."
    )

    sparse.save_npz(
        X_TEST_FILE,
        X_test_prepared
    )

    print(
        "Saving y_train..."
    )

    np.save(
        Y_TRAIN_FILE,
        y_train.to_numpy(
            dtype=np.int8
        )
    )

    print(
        "Saving y_test..."
    )

    np.save(
        Y_TEST_FILE,
        y_test.to_numpy(
            dtype=np.int8
        )
    )

    print(
        "Saving preprocessing pipeline..."
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    balanced,
    X_train_prepared,
    X_test_prepared,
    y_train,
    y_test
):

    print_section(
        "FINAL VALIDATION"
    )

    checks = []

    # --------------------------------------------------------
    # BALANCED FULL DATASET
    # --------------------------------------------------------

    balanced_counts = (
        balanced[
            TARGET_COLUMN
        ]
        .value_counts()
    )

    balance_ok = (
        len(balanced_counts) == 2
        and int(
            balanced_counts.get(
                0,
                0
            )
        )
        ==
        int(
            balanced_counts.get(
                1,
                0
            )
        )
    )

    checks.append(
        (
            "Balanced dataset is 50/50",
            balance_ok
        )
    )

    # --------------------------------------------------------
    # TRAIN BALANCE
    # --------------------------------------------------------

    train_counts = (
        y_train.value_counts()
    )

    train_balance_ok = (
        abs(
            int(
                train_counts.get(
                    0,
                    0
                )
            )
            -
            int(
                train_counts.get(
                    1,
                    0
                )
            )
        )
        <= 1
    )

    checks.append(
        (
            "Training set is balanced",
            train_balance_ok
        )
    )

    # --------------------------------------------------------
    # TEST BALANCE
    # --------------------------------------------------------

    test_counts = (
        y_test.value_counts()
    )

    test_balance_ok = (
        abs(
            int(
                test_counts.get(
                    0,
                    0
                )
            )
            -
            int(
                test_counts.get(
                    1,
                    0
                )
            )
        )
        <= 1
    )

    checks.append(
        (
            "Testing set is balanced",
            test_balance_ok
        )
    )

    # --------------------------------------------------------
    # ROW COUNTS
    # --------------------------------------------------------

    checks.append(
        (
            "Training features match labels",
            X_train_prepared.shape[0]
            == len(y_train)
        )
    )

    checks.append(
        (
            "Testing features match labels",
            X_test_prepared.shape[0]
            == len(y_test)
        )
    )

    # --------------------------------------------------------
    # FEATURE COUNTS
    # --------------------------------------------------------

    checks.append(
        (
            "Train/test feature counts match",
            X_train_prepared.shape[1]
            == X_test_prepared.shape[1]
        )
    )

    # --------------------------------------------------------
    # NAN CHECK
    # --------------------------------------------------------

    train_has_nan = bool(
        np.isnan(
            X_train_prepared.data
        ).any()
    )

    test_has_nan = bool(
        np.isnan(
            X_test_prepared.data
        ).any()
    )

    checks.append(
        (
            "No NaN in training data",
            not train_has_nan
        )
    )

    checks.append(
        (
            "No NaN in testing data",
            not test_has_nan
        )
    )

    all_passed = True

    for name, result in checks:

        status = (
            "PASS"
            if result
            else "FAIL"
        )

        print(
            f"[{status}] {name}"
        )

        if not result:
            all_passed = False

    if not all_passed:

        raise RuntimeError(
            "Training-data validation failed."
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    original_rows,
    deduplicated_rows,
    balanced,
    X_train_prepared,
    X_test_prepared,
    y_train,
    y_test,
    numeric_columns,
    categorical_columns,
    feature_names
):

    summary = {

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "original_rows":
            int(
                original_rows
            ),

        "rows_after_duplicate_removal":
            int(
                deduplicated_rows
            ),

        "balanced_total_rows":
            int(
                len(balanced)
            ),

        "balanced_benign":
            int(
                (
                    balanced[
                        TARGET_COLUMN
                    ] == 0
                ).sum()
            ),

        "balanced_phishing":
            int(
                (
                    balanced[
                        TARGET_COLUMN
                    ] == 1
                ).sum()
            ),

        "training_rows":
            int(
                len(y_train)
            ),

        "training_benign":
            int(
                (
                    y_train == 0
                ).sum()
            ),

        "training_phishing":
            int(
                (
                    y_train == 1
                ).sum()
            ),

        "testing_rows":
            int(
                len(y_test)
            ),

        "testing_benign":
            int(
                (
                    y_test == 0
                ).sum()
            ),

        "testing_phishing":
            int(
                (
                    y_test == 1
                ).sum()
            ),

        "raw_numeric_features":
            int(
                len(
                    numeric_columns
                )
            ),

        "raw_categorical_features":
            int(
                len(
                    categorical_columns
                )
            ),

        "final_prepared_features":
            int(
                len(
                    feature_names
                )
            ),

        "X_train_shape":
            [
                int(
                    X_train_prepared.shape[0]
                ),
                int(
                    X_train_prepared.shape[1]
                )
            ],

        "X_test_shape":
            [
                int(
                    X_test_prepared.shape[0]
                ),
                int(
                    X_test_prepared.shape[1]
                )
            ],

        "balancing_method":
            "Random undersampling",

        "samples_per_class":
            int(
                (
                    balanced[
                        TARGET_COLUMN
                    ] == 0
                ).sum()
            ),

        "split_method":
            "Stratified 80/20 split",

        "numeric_imputation":
            "Median",

        "missing_value_indicators":
            True,

        "numeric_scaling":
            "RobustScaler",

        "categorical_imputation":
            "missing category",

        "categorical_encoding":
            "OneHotEncoder",

        "removed_constant_columns":
            CONSTANT_COLUMNS,

        "removed_identifier_columns":
            IDENTIFIER_COLUMNS
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 80)

    print(
        "04 - PREPARE BALANCED "
        "TRAINING DATA"
    )

    print("=" * 80)

    start_time = time.time()

    create_directories()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_dataset()

    original_rows = len(df)

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    df = remove_duplicates(
        df
    )

    deduplicated_rows = len(df)

    # --------------------------------------------------------
    # BALANCE
    # --------------------------------------------------------

    balanced = (
        create_balanced_dataset(
            df
        )
    )

    # Full dataset no longer needed
    del df

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    X, y = (
        create_feature_table(
            balanced
        )
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_raw,
        test_raw
    ) = split_dataset(
        X,
        y,
        balanced
    )

    # --------------------------------------------------------
    # FEATURE TYPES
    # --------------------------------------------------------

    (
        numeric_columns,
        categorical_columns
    ) = identify_feature_types(
        X_train
    )

    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

    preprocessor = (
        build_preprocessor(
            numeric_columns,
            categorical_columns
        )
    )

    # --------------------------------------------------------
    # FIT TRAIN ONLY
    # --------------------------------------------------------

    (
        X_train_prepared,
        X_test_prepared
    ) = preprocess_data(
        preprocessor,
        X_train,
        X_test
    )

    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------

    feature_names = (
        save_feature_names(
            preprocessor
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_outputs(
        balanced,
        train_raw,
        test_raw,
        X_train_prepared,
        X_test_prepared,
        y_train,
        y_test,
        preprocessor
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    final_validation(
        balanced,
        X_train_prepared,
        X_test_prepared,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    save_summary(
        original_rows,
        deduplicated_rows,
        balanced,
        X_train_prepared,
        X_test_prepared,
        y_train,
        y_test,
        numeric_columns,
        categorical_columns,
        feature_names
    )

    runtime = (
        time.time()
        - start_time
    )

    print_section(
        "TRAINING DATA PREPARATION COMPLETED"
    )

    print(
        f"Original dataset: "
        f"{original_rows:,}"
    )

    print(
        f"After duplicate removal: "
        f"{deduplicated_rows:,}"
    )

    print(
        f"Balanced dataset: "
        f"{len(balanced):,}"
    )

    print()

    print(
        f"Training samples: "
        f"{len(y_train):,}"
    )

    print(
        f"Testing samples:  "
        f"{len(y_test):,}"
    )

    print()

    print(
        f"Prepared ML features: "
        f"{X_train_prepared.shape[1]:,}"
    )

    print(
        f"Total runtime: "
        f"{runtime / 60:.2f} minutes"
    )

    print()

    print(
        "Balanced dataset saved to:"
    )

    print(
        BALANCED_FILE
    )

    print()

    print(
        "Prepared training data:"
    )

    print(
        X_TRAIN_FILE
    )

    print()

    print(
        "Prepared test data:"
    )

    print(
        X_TEST_FILE
    )

    print()

    print(
        "Preprocessing model:"
    )

    print(
        PREPROCESSOR_FILE
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "05_train_models.py"
    )


if __name__ == "__main__":
    main()