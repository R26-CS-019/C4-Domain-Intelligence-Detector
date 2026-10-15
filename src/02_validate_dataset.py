from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 02 - VALIDATE PROCESSED DOMAIN INTELLIGENCE DATASET
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

RESULTS_DIR = (
    PROJECT_ROOT / "results"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "domain_intelligence_full.csv"
)

DATASET_SUMMARY_FILE = (
    RESULTS_DIR
    / "dataset_summary.csv"
)

MISSING_VALUES_FILE = (
    RESULTS_DIR
    / "missing_values.csv"
)

FEATURE_STATISTICS_FILE = (
    RESULTS_DIR
    / "feature_statistics.csv"
)

CLASS_DISTRIBUTION_FILE = (
    RESULTS_DIR
    / "class_distribution.csv"
)

CONSTANT_COLUMNS_FILE = (
    RESULTS_DIR
    / "constant_columns.csv"
)

DUPLICATE_DOMAINS_FILE = (
    RESULTS_DIR
    / "duplicate_domains.csv"
)

CATEGORICAL_SUMMARY_FILE = (
    RESULTS_DIR
    / "categorical_summary.csv"
)

FEATURE_GROUP_SUMMARY_FILE = (
    RESULTS_DIR
    / "feature_group_summary.csv"
)

INVALID_VALUE_REPORT_FILE = (
    RESULTS_DIR
    / "invalid_value_report.csv"
)


# ============================================================
# SETTINGS
# ============================================================

IDENTIFIER_COLUMNS = {
    "domain_name",
    "source",
    "class_name"
}

CATEGORICAL_FEATURE_COLUMNS = {
    "tls_protocol",
    "tls_cipher"
}


# ============================================================
# BASIC HELPERS
# ============================================================

def print_section(
    title: str
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def percentage(
    part: int | float,
    total: int | float
) -> float:

    if total == 0:
        return 0.0

    return round(
        (part / total) * 100,
        4
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:

    print_section(
        "LOADING PROCESSED DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Processed dataset not found:\n"
            f"{INPUT_FILE}"
        )

    file_size_mb = (
        INPUT_FILE.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"Input file: "
        f"{INPUT_FILE}"
    )

    print(
        f"File size:  "
        f"{file_size_mb:,.2f} MB"
    )

    print()
    print("Loading CSV...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Rows:    "
        f"{len(df):,}"
    )

    print(
        f"Columns: "
        f"{len(df.columns):,}"
    )

    return df


# ============================================================
# DATASET SUMMARY
# ============================================================

def create_dataset_summary(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "DATASET SUMMARY"
    )

    total_rows = len(df)
    total_columns = len(df.columns)

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_domains = 0

    if "domain_name" in df.columns:

        duplicate_domains = int(
            df["domain_name"]
            .duplicated()
            .sum()
        )

    total_missing = int(
        df.isna()
        .sum()
        .sum()
    )

    total_cells = (
        total_rows
        * total_columns
    )

    missing_percentage = percentage(
        total_missing,
        total_cells
    )

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "total_rows",
                "value":
                    total_rows
            },
            {
                "metric":
                    "total_columns",
                "value":
                    total_columns
            },
            {
                "metric":
                    "duplicate_rows",
                "value":
                    duplicate_rows
            },
            {
                "metric":
                    "duplicate_domains",
                "value":
                    duplicate_domains
            },
            {
                "metric":
                    "total_missing_values",
                "value":
                    total_missing
            },
            {
                "metric":
                    "missing_percentage",
                "value":
                    missing_percentage
            }
        ]
    )

    print(
        summary.to_string(
            index=False
        )
    )

    summary.to_csv(
        DATASET_SUMMARY_FILE,
        index=False
    )

    return summary


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def validate_class_distribution(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "CLASS DISTRIBUTION"
    )

    if "label" not in df.columns:

        raise ValueError(
            "Dataset does not contain "
            "'label' column."
        )

    labels_found = (
        df["label"]
        .dropna()
        .unique()
        .tolist()
    )

    labels_found = sorted(
        int(x)
        for x in labels_found
    )

    print(
        f"Unique labels found: "
        f"{labels_found}"
    )

    if set(labels_found) == {0, 1}:

        print(
            "Label validation passed: "
            "0 = benign, 1 = phishing"
        )

    else:

        print(
            "WARNING: Expected labels "
            "{0, 1}"
        )

    distribution = (
        df["label"]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .rename_axis(
            "label"
        )
        .reset_index(
            name="count"
        )
    )

    distribution[
        "percentage"
    ] = distribution[
        "count"
    ].apply(
        lambda x:
            percentage(
                x,
                len(df)
            )
    )

    def get_class_name(
        label
    ) -> str:

        if label == 0:
            return "benign"

        if label == 1:
            return "phishing"

        return "unknown"

    distribution[
        "class_name"
    ] = distribution[
        "label"
    ].apply(
        get_class_name
    )

    distribution = distribution[
        [
            "label",
            "class_name",
            "count",
            "percentage"
        ]
    ]

    print()
    print(
        distribution.to_string(
            index=False
        )
    )

    distribution.to_csv(
        CLASS_DISTRIBUTION_FILE,
        index=False
    )

    return distribution


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

def analyze_missing_values(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "MISSING VALUE ANALYSIS"
    )

    missing_count = (
        df.isna()
        .sum()
    )

    report = pd.DataFrame(
        {
            "column":
                df.columns,

            "missing_count":
                [
                    int(
                        missing_count[
                            column
                        ]
                    )
                    for column
                    in df.columns
                ],

            "missing_percentage":
                [
                    percentage(
                        missing_count[
                            column
                        ],
                        len(df)
                    )
                    for column
                    in df.columns
                ]
        }
    )

    report = (
        report
        .sort_values(
            by="missing_percentage",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    report.to_csv(
        MISSING_VALUES_FILE,
        index=False
    )

    print(
        report.head(30)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Top 30 columns with the "
        "highest missing percentages "
        "shown above."
    )

    return report


# ============================================================
# NUMERIC FEATURE STATISTICS
# ============================================================

def analyze_numeric_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "NUMERIC FEATURE STATISTICS"
    )

    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    numeric_columns = [
        column
        for column
        in numeric_columns
        if column != "label"
    ]

    if not numeric_columns:

        print(
            "No numeric feature "
            "columns found."
        )

        return pd.DataFrame()

    statistics = (
        df[
            numeric_columns
        ]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.95,
                0.99
            ]
        )
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index":
                    "feature"
            }
        )
    )

    statistics[
        "missing_count"
    ] = [
        int(
            df[column]
            .isna()
            .sum()
        )
        for column
        in numeric_columns
    ]

    statistics[
        "missing_percentage"
    ] = [
        percentage(
            df[column]
            .isna()
            .sum(),
            len(df)
        )
        for column
        in numeric_columns
    ]

    statistics[
        "unique_values"
    ] = [
        int(
            df[column]
            .nunique(
                dropna=True
            )
        )
        for column
        in numeric_columns
    ]

    statistics.to_csv(
        FEATURE_STATISTICS_FILE,
        index=False
    )

    display_columns = [
        "feature",
        "count",
        "mean",
        "std",
        "min",
        "max",
        "missing_percentage",
        "unique_values"
    ]

    print(
        statistics[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Numeric features found: "
        f"{len(numeric_columns)}"
    )

    return statistics


# ============================================================
# CONSTANT / LOW INFORMATION COLUMNS
# ============================================================

def detect_constant_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "CONSTANT / LOW-INFORMATION COLUMNS"
    )

    rows = []

    for column in df.columns:

        non_null_unique = int(
            df[column]
            .nunique(
                dropna=True
            )
        )

        total_unique = int(
            df[column]
            .nunique(
                dropna=False
            )
        )

        rows.append(
            {
                "column":
                    column,

                "unique_values_including_nan":
                    total_unique,

                "unique_non_null_values":
                    non_null_unique,

                "constant":
                    non_null_unique <= 1
            }
        )

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        CONSTANT_COLUMNS_FILE,
        index=False
    )

    constant_columns = (
        report[
            report[
                "constant"
            ]
            == True
        ]
    )

    if constant_columns.empty:

        print(
            "No constant columns found."
        )

    else:

        print(
            constant_columns
            .to_string(
                index=False
            )
        )

    return report


# ============================================================
# DUPLICATE DOMAIN ANALYSIS
# ============================================================

def analyze_duplicate_domains(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "DUPLICATE DOMAIN ANALYSIS"
    )

    if "domain_name" not in df.columns:

        print(
            "domain_name column not found."
        )

        return pd.DataFrame()

    duplicate_mask = (
        df[
            "domain_name"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicate_rows = (
        df.loc[
            duplicate_mask,
            [
                "domain_name",
                "label",
                "class_name"
            ]
        ]
        .copy()
    )

    if duplicate_rows.empty:

        print(
            "No duplicated domain "
            "names found."
        )

        pd.DataFrame(
            columns=[
                "domain_name",
                "occurrence_count",
                "unique_labels",
                "labels"
            ]
        ).to_csv(
            DUPLICATE_DOMAINS_FILE,
            index=False
        )

        return duplicate_rows

    duplicate_summary = (
        duplicate_rows
        .groupby(
            "domain_name",
            dropna=False
        )
        .agg(
            occurrence_count=(
                "domain_name",
                "size"
            ),
            unique_labels=(
                "label",
                "nunique"
            ),
            labels=(
                "label",
                lambda x:
                    ",".join(
                        sorted(
                            set(
                                map(
                                    str,
                                    x
                                )
                            )
                        )
                    )
            )
        )
        .reset_index()
        .sort_values(
            by="occurrence_count",
            ascending=False
        )
    )

    duplicate_summary.to_csv(
        DUPLICATE_DOMAINS_FILE,
        index=False
    )

    conflicting = (
        duplicate_summary[
            duplicate_summary[
                "unique_labels"
            ] > 1
        ]
    )

    print(
        f"Duplicated domain names: "
        f"{len(duplicate_summary):,}"
    )

    print(
        "Domains appearing in "
        f"BOTH classes: "
        f"{len(conflicting):,}"
    )

    if not conflicting.empty:

        print()
        print(
            "Example conflicting domains:"
        )

        print(
            conflicting
            .head(20)
            .to_string(
                index=False
            )
        )

    return duplicate_summary


# ============================================================
# CATEGORICAL FEATURE ANALYSIS
# ============================================================

def analyze_categorical_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "CATEGORICAL FEATURE ANALYSIS"
    )

    rows = []

    for column in sorted(
        CATEGORICAL_FEATURE_COLUMNS
    ):

        if column not in df.columns:
            continue

        unique_count = int(
            df[column]
            .nunique(
                dropna=True
            )
        )

        missing_count = int(
            df[column]
            .isna()
            .sum()
        )

        most_common_value = None
        most_common_count = 0

        non_null = (
            df[column]
            .dropna()
        )

        if not non_null.empty:

            counts = (
                non_null
                .value_counts()
            )

            most_common_value = (
                counts.index[0]
            )

            most_common_count = int(
                counts.iloc[0]
            )

        rows.append(
            {
                "column":
                    column,

                "unique_values":
                    unique_count,

                "missing_count":
                    missing_count,

                "missing_percentage":
                    percentage(
                        missing_count,
                        len(df)
                    ),

                "most_common_value":
                    most_common_value,

                "most_common_count":
                    most_common_count,

                "most_common_percentage":
                    percentage(
                        most_common_count,
                        len(df)
                    )
            }
        )

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        CATEGORICAL_SUMMARY_FILE,
        index=False
    )

    if report.empty:

        print(
            "No categorical features found."
        )

    else:

        print(
            report.to_string(
                index=False
            )
        )

        print()

        for column in sorted(
            CATEGORICAL_FEATURE_COLUMNS
        ):

            if column not in df.columns:
                continue

            print(
                f"Top values for "
                f"{column}:"
            )

            print(
                df[column]
                .value_counts(
                    dropna=False
                )
                .head(15)
                .to_string()
            )

            print()

    return report


# ============================================================
# FEATURE GROUP SUMMARY
# ============================================================

def analyze_feature_groups(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "FEATURE GROUP SUMMARY"
    )

    dns_columns = [
        column
        for column
        in df.columns
        if (
            column.startswith(
                "dns_"
            )
            or column.startswith(
                "dnssec_"
            )
        )
    ]

    rdap_columns = [
        column
        for column
        in df.columns
        if (
            column.startswith(
                "rdap_"
            )
            or column
            in {
                "domain_age_days",
                "registration_period_days",
                "days_until_expiration",
                "days_since_last_change"
            }
        )
    ]

    tls_columns = [
        column
        for column
        in df.columns
        if column.startswith(
            "tls_"
        )
    ]

    ip_columns = [
        column
        for column
        in df.columns
        if (
            column.startswith(
                "ip_"
            )
            or column
            in {
                "unique_ip_count",
                "asn_count",
                "asn_organization_count",
                "country_count",
                "country_code_count",
                "rir_count"
            }
        )
    ]

    groups = {
        "DNS":
            dns_columns,

        "RDAP_WHOIS":
            rdap_columns,

        "TLS":
            tls_columns,

        "IP_HOSTING":
            ip_columns
    }

    rows = []

    for group_name, columns in (
        groups.items()
    ):

        if not columns:
            continue

        missing_cells = int(
            df[
                columns
            ]
            .isna()
            .sum()
            .sum()
        )

        total_cells = (
            len(df)
            * len(columns)
        )

        constant_count = sum(
            1
            for column
            in columns
            if (
                df[column]
                .nunique(
                    dropna=True
                )
                <= 1
            )
        )

        rows.append(
            {
                "feature_group":
                    group_name,

                "feature_count":
                    len(columns),

                "missing_cells":
                    missing_cells,

                "missing_percentage":
                    percentage(
                        missing_cells,
                        total_cells
                    ),

                "constant_feature_count":
                    constant_count
            }
        )

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        FEATURE_GROUP_SUMMARY_FILE,
        index=False
    )

    print(
        report.to_string(
            index=False
        )
    )

    return report


# ============================================================
# INVALID / SUSPICIOUS VALUE CHECKS
# ============================================================

def analyze_invalid_values(
    df: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "INVALID / SUSPICIOUS VALUE CHECKS"
    )

    rows = []

    def add_check(
        feature: str,
        condition,
        description: str
    ) -> None:

        if feature not in df.columns:
            return

        count = int(
            condition.sum()
        )

        rows.append(
            {
                "feature":
                    feature,

                "issue":
                    description,

                "count":
                    count,

                "percentage":
                    percentage(
                        count,
                        len(df)
                    )
            }
        )

    if "domain_age_days" in df.columns:

        add_check(
            "domain_age_days",
            df[
                "domain_age_days"
            ] < 0,
            "negative domain age"
        )

    if (
        "registration_period_days"
        in df.columns
    ):

        add_check(
            "registration_period_days",
            df[
                "registration_period_days"
            ] < 0,
            "negative registration period"
        )

    if (
        "tls_certificate_validity_days"
        in df.columns
    ):

        add_check(
            "tls_certificate_validity_days",
            df[
                "tls_certificate_validity_days"
            ] < 0,
            "negative certificate validity"
        )

    if (
        "ip_alive_ratio"
        in df.columns
    ):

        add_check(
            "ip_alive_ratio",
            (
                (
                    df[
                        "ip_alive_ratio"
                    ] < 0
                )
                |
                (
                    df[
                        "ip_alive_ratio"
                    ] > 1
                )
            ),
            "alive ratio outside 0-1"
        )

    count_columns = [
        column
        for column
        in df.columns
        if (
            column.endswith(
                "_count"
            )
        )
    ]

    for column in count_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        add_check(
            column,
            df[column] < 0,
            "negative count"
        )

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        INVALID_VALUE_REPORT_FILE,
        index=False
    )

    if report.empty:

        print(
            "No suspicious value "
            "checks available."
        )

    else:

        print(
            report.to_string(
                index=False
            )
        )

    return report


# ============================================================
# CORRECTED DNS FEATURE CHECK
# ============================================================

def check_corrected_dns_features(
    df: pd.DataFrame
) -> None:

    print_section(
        "CORRECTED DNS FEATURE CHECK"
    )

    important_columns = [
        "dns_mx_count",
        "dns_has_mx",
        "dns_ns_count",
        "dns_has_ns",
        "dns_mx_related_ip_count",
        "dns_ns_related_ip_count",
        "dns_mx_min_priority",
        "dns_mx_max_priority",
        "dns_mx_mean_priority"
    ]

    for column in important_columns:

        if column not in df.columns:

            print(
                f"[MISSING] {column}"
            )

            continue

        unique_count = int(
            df[column]
            .nunique(
                dropna=True
            )
        )

        non_zero_count = 0

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            non_zero_count = int(
                (
                    df[column]
                    .fillna(0)
                    != 0
                )
                .sum()
            )

        print(
            f"{column:<32}"
            f" unique={unique_count:<8}"
            f" non_zero={non_zero_count:,}"
        )

    print()

    required_nonconstant = [
        "dns_mx_count",
        "dns_has_mx",
        "dns_ns_count",
        "dns_has_ns"
    ]

    all_good = True

    for column in required_nonconstant:

        if column not in df.columns:

            all_good = False
            continue

        if (
            df[column]
            .nunique(
                dropna=True
            )
            <= 1
        ):

            all_good = False

            print(
                f"[FAIL] {column} "
                "is still constant."
            )

        else:

            print(
                f"[PASS] {column} "
                "contains variation."
            )

    if all_good:

        print()
        print(
            "Corrected MX/NS extraction "
            "appears successful."
        )


# ============================================================
# DATA TYPES
# ============================================================

def inspect_data_types(
    df: pd.DataFrame
) -> None:

    print_section(
        "COLUMN DATA TYPES"
    )

    report = pd.DataFrame(
        {
            "column":
                df.columns,

            "dtype":
                [
                    str(
                        df[column].dtype
                    )
                    for column
                    in df.columns
                ]
        }
    )

    print(
        report.to_string(
            index=False
        )
    )


# ============================================================
# FINAL VALIDATION CHECKS
# ============================================================

def run_validation_checks(
    df: pd.DataFrame
) -> None:

    print_section(
        "FINAL VALIDATION CHECKS"
    )

    checks = []

    checks.append(
        (
            "Dataset contains rows",
            len(df) > 0
        )
    )

    checks.append(
        (
            "Dataset contains label",
            "label" in df.columns
        )
    )

    checks.append(
        (
            "Dataset contains domain_name",
            "domain_name"
            in df.columns
        )
    )

    if "label" in df.columns:

        labels = set(
            df[
                "label"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        checks.append(
            (
                "Labels are only 0 and 1",
                labels == {0, 1}
            )
        )

    feature_groups = {

        "DNS":
            [
                column
                for column
                in df.columns
                if (
                    column.startswith(
                        "dns_"
                    )
                    or column.startswith(
                        "dnssec_"
                    )
                )
            ],

        "RDAP":
            [
                column
                for column
                in df.columns
                if (
                    column.startswith(
                        "rdap_"
                    )
                    or column
                    in {
                        "domain_age_days",
                        "registration_period_days",
                        "days_until_expiration",
                        "days_since_last_change"
                    }
                )
            ],

        "TLS":
            [
                column
                for column
                in df.columns
                if column.startswith(
                    "tls_"
                )
            ],

        "IP":
            [
                column
                for column
                in df.columns
                if (
                    column.startswith(
                        "ip_"
                    )
                    or column
                    in {
                        "unique_ip_count",
                        "asn_count",
                        "asn_organization_count",
                        "country_count",
                        "country_code_count",
                        "rir_count"
                    }
                )
            ]
    }

    for group_name, columns in (
        feature_groups.items()
    ):

        checks.append(
            (
                f"{group_name} features exist",
                len(columns) > 0
            )
        )

    corrected_dns_columns = [
        "dns_mx_count",
        "dns_ns_count",
        "dns_has_mx",
        "dns_has_ns"
    ]

    for column in corrected_dns_columns:

        if column in df.columns:

            checks.append(
                (
                    f"{column} is not constant",
                    (
                        df[column]
                        .nunique(
                            dropna=True
                        )
                        > 1
                    )
                )
            )

    for check_name, result in checks:

        status = (
            "PASS"
            if result
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{check_name}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 80)
    print(
        "02 - DOMAIN INTELLIGENCE "
        "DATASET VALIDATION"
    )
    print("=" * 80)

    df = load_dataset()

    create_dataset_summary(
        df
    )

    validate_class_distribution(
        df
    )

    analyze_missing_values(
        df
    )

    analyze_numeric_features(
        df
    )

    detect_constant_columns(
        df
    )

    analyze_duplicate_domains(
        df
    )

    analyze_categorical_features(
        df
    )

    analyze_feature_groups(
        df
    )

    analyze_invalid_values(
        df
    )

    check_corrected_dns_features(
        df
    )

    inspect_data_types(
        df
    )

    run_validation_checks(
        df
    )

    print_section(
        "VALIDATION COMPLETED"
    )

    print(
        "Generated result files:"
    )

    print(
        f"1. {DATASET_SUMMARY_FILE}"
    )

    print(
        f"2. {MISSING_VALUES_FILE}"
    )

    print(
        f"3. {FEATURE_STATISTICS_FILE}"
    )

    print(
        f"4. {CLASS_DISTRIBUTION_FILE}"
    )

    print(
        f"5. {CONSTANT_COLUMNS_FILE}"
    )

    print(
        f"6. {DUPLICATE_DOMAINS_FILE}"
    )

    print(
        f"7. {CATEGORICAL_SUMMARY_FILE}"
    )

    print(
        f"8. {FEATURE_GROUP_SUMMARY_FILE}"
    )

    print(
        f"9. {INVALID_VALUE_REPORT_FILE}"
    )

    print()

    print(
        "Next step:"
    )

    print(
        "Review this validation output "
        "before creating "
        "04_prepare_training_data.py."
    )


if __name__ == "__main__":
    main()