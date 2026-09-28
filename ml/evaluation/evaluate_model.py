from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)

from sklearn.model_selection import train_test_split


# ============================================================
# CYBERGUARD-AI
# XGBOOST MODEL EVALUATION
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

# Current file:
# D:\CyberGuard-AI\ml\evaluation\evaluate_model.py
#
# parents[0] = D:\CyberGuard-AI\ml\evaluation
# parents[1] = D:\CyberGuard-AI\ml
# parents[2] = D:\CyberGuard-AI
#
# Therefore parents[2] is the PROJECT ROOT.

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "ml"
    / "models"
)


RESULT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "evaluation"
    / "results"
)


# Create result directory if it doesn't exist
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 300_000

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATH = (
    MODEL_DIR
    / "xgboost_model.joblib"
)


FEATURE_PATH = (
    MODEL_DIR
    / "feature_names.joblib"
)


# ============================================================
# DATA LOADER
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING FEATURE DATASETS")
    print("=" * 70)

    files = sorted(
        DATA_DIR.glob("*_features.csv")
    )

    if not files:

        raise FileNotFoundError(
            f"\nNo feature CSV files found in:\n"
            f"{DATA_DIR}"
        )

    dataframes = []

    total_rows = 0

    for file in files:

        print(
            f"\nLoading: {file.name}"
        )

        df = pd.read_csv(
            file,
            low_memory=False
        )

        print(
            f"Rows: {len(df):,} | "
            f"Columns: {len(df.columns)}"
        )

        dataframes.append(df)

        total_rows += len(df)

    print(
        "\n" + "-" * 70
    )

    print(
        f"Total available rows: "
        f"{total_rows:,}"
    )

    print(
        "\nCombining datasets..."
    )

    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    del dataframes

    print(
        f"Combined dataset shape: "
        f"{data.shape}"
    )

    return data


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print(
        "\nPreparing features..."
    )

    # --------------------------------------------------------
    # TARGET CHECK
    # --------------------------------------------------------

    if "Target" not in df.columns:

        raise ValueError(
            "Target column not found."
        )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y = df["Target"].astype(int)

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = df.drop(
        columns=["Target"],
        errors="ignore"
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # This is EXACTLY the same operation
    # used in train_models.py
    # --------------------------------------------------------

    X = X.select_dtypes(
        include=["int64", "float64"]
    )

    # --------------------------------------------------------
    # REPLACE INFINITY
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # MEDIAN IMPUTATION
    #
    # Same as training code
    # --------------------------------------------------------

    X = X.fillna(
        X.median()
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if X.isnull().sum().sum() > 0:

        raise ValueError(
            "Missing values still exist "
            "after preprocessing."
        )

    if not np.isfinite(
        X.to_numpy()
    ).all():

        raise ValueError(
            "Infinite values still exist "
            "after preprocessing."
        )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    print(
        f"Number of features: "
        f"{X.shape[1]}"
    )

    print(
        f"Normal samples: "
        f"{(y == 0).sum():,}"
    )

    print(
        f"Attack samples: "
        f"{(y == 1).sum():,}"
    )

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "CYBERGUARD-AI - XGBOOST EVALUATION"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # CHECK PATHS
    # ========================================================

    print("\nProject root:")

    print(PROJECT_ROOT)

    print("\nModel path:")

    print(MODEL_PATH)

    print("\nFeature names path:")

    print(FEATURE_PATH)

    print("\nData directory:")

    print(DATA_DIR)


    # ========================================================
    # VERIFY MODEL FILE
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nXGBoost model not found:\n"
            f"{MODEL_PATH}"
        )


    # ========================================================
    # VERIFY FEATURE FILE
    # ========================================================

    if not FEATURE_PATH.exists():

        raise FileNotFoundError(
            f"\nFeature names file not found:\n"
            f"{FEATURE_PATH}"
        )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading XGBoost model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "XGBoost model loaded successfully."
    )


    # ========================================================
    # LOAD FEATURE NAMES
    # ========================================================

    print(
        "\nLoading feature names..."
    )

    feature_names = joblib.load(
        FEATURE_PATH
    )

    print(
        f"Number of expected features: "
        f"{len(feature_names)}"
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data()


    # ========================================================
    # STRATIFIED SAMPLE
    #
    # EXACTLY SAME AS TRAINING
    # ========================================================

    if len(df) > SAMPLE_SIZE:

        print(
            f"\nSampling {SAMPLE_SIZE:,} records "
            f"using stratified sampling..."
        )

        df, _ = train_test_split(
            df,
            train_size=SAMPLE_SIZE,
            stratify=df["Target"],
            random_state=RANDOM_STATE
        )

    print(
        f"\nEvaluation dataset size: "
        f"{len(df):,}"
    )


    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    print(
        "\nSample target distribution:"
    )

    print(
        df["Target"]
        .value_counts()
        .sort_index()
    )


    print(
        "\nTarget percentages:"
    )

    print(
        (
            df["Target"]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )


    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    X, y = prepare_data(
        df
    )


    # ========================================================
    # VERIFY FEATURE COUNT
    # ========================================================

    print(
        "\nChecking feature structure..."
    )

    print(
        f"Current feature count: "
        f"{len(X.columns)}"
    )

    print(
        f"Saved feature count: "
        f"{len(feature_names)}"
    )


    # ========================================================
    # CHECK FEATURE NAMES
    # ========================================================

    current_features = list(
        X.columns
    )


    if set(current_features) != set(
        feature_names
    ):

        print(
            "\nWARNING:"
        )

        print(
            "Current features and saved "
            "model features do not match."
        )

        missing = [
            feature
            for feature in feature_names
            if feature not in current_features
        ]

        extra = [
            feature
            for feature in current_features
            if feature not in feature_names
        ]

        if missing:

            print(
                "\nMissing features:"
            )

            for feature in missing:

                print(
                    " -",
                    feature
                )

        if extra:

            print(
                "\nExtra features:"
            )

            for feature in extra:

                print(
                    " -",
                    feature
                )

        raise ValueError(
            "Feature mismatch detected."
        )


    # ========================================================
    # FORCE EXACT FEATURE ORDER
    # ========================================================

    X = X[
        feature_names
    ]


    print(
        "\nFeature order verified."
    )


    # ========================================================
    # TRAIN / TEST SPLIT
    #
    # EXACTLY SAME AS TRAINING
    # ========================================================

    print(
        "\nCreating 80/20 stratified split..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE
        )
    )


    print(
        f"\nTraining samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test):,}"
    )


    # ========================================================
    # IMPORTANT XGBOOST DETAIL
    # ========================================================
    #
    # Your original train_models.py contains:
    #
    # xgb_model.fit(
    #     X_train,
    #     y_train
    # )
    #
    # NOT:
    #
    # xgb_model.fit(
    #     X_train_scaled,
    #     y_train
    # )
    #
    # Therefore XGBoost must be evaluated
    # using X_test directly.
    #
    # ========================================================

    print(
        "\nXGBoost was trained on "
        "UNSCALED features."
    )

    print(
        "Using X_test directly..."
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    print(
        "\nGenerating predictions..."
    )

    y_pred = model.predict(
        X_test
    )


    # ========================================================
    # PREDICT PROBABILITIES
    # ========================================================

    print(
        "Generating prediction probabilities..."
    )

    y_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    # ========================================================
    # BASIC METRICS
    # ========================================================

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


    # ========================================================
    # PERFORMANCE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "XGBOOST PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAccuracy : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall   : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score : "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"ROC-AUC  : "
        f"{roc_auc:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CLASSIFICATION REPORT"
    )

    print(
        "=" * 70
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Attack"
        ],
        zero_division=0
    )

    print(
        report
    )


    report_path = (
        RESULT_DIR
        / "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "\nGenerating confusion matrix..."
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )


    fig, ax = plt.subplots(
        figsize=(7, 6)
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Normal",
            "Attack"
        ]
    )


    display.plot(
        ax=ax
    )


    plt.title(
        "CyberGuard-AI - XGBoost Confusion Matrix"
    )

    plt.tight_layout()


    confusion_path = (
        RESULT_DIR
        / "confusion_matrix.png"
    )


    plt.savefig(
        confusion_path,
        dpi=300
    )

    plt.close()


    print(
        f"Saved: "
        f"{confusion_path}"
    )


    # ========================================================
    # ROC CURVE
    # ========================================================

    print(
        "\nGenerating ROC curve..."
    )


    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )


    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(
        fpr,
        tpr,
        label=(
            f"XGBoost "
            f"(AUC = {roc_auc:.4f})"
        )
    )


    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        "ROC Curve - CyberGuard-AI"
    )


    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()


    roc_path = (
        RESULT_DIR
        / "roc_curve.png"
    )


    plt.savefig(
        roc_path,
        dpi=300
    )

    plt.close()


    print(
        f"Saved: "
        f"{roc_path}"
    )


    # ========================================================
    # PRECISION-RECALL CURVE
    # ========================================================

    print(
        "\nGenerating "
        "Precision-Recall curve..."
    )


    precision_values, recall_values, _ = (
        precision_recall_curve(
            y_test,
            y_probability
        )
    )


    pr_auc = auc(
        recall_values,
        precision_values
    )


    print(
        f"PR-AUC: "
        f"{pr_auc:.4f}"
    )


    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(
        recall_values,
        precision_values,
        label=(
            f"XGBoost "
            f"(AUC = {pr_auc:.4f})"
        )
    )


    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )


    plt.title(
        "Precision-Recall Curve - CyberGuard-AI"
    )


    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()


    pr_path = (
        RESULT_DIR
        / "precision_recall_curve.png"
    )


    plt.savefig(
        pr_path,
        dpi=300
    )

    plt.close()


    print(
        f"Saved: "
        f"{pr_path}"
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print(
        "\nCalculating feature importance..."
    )


    if not hasattr(
        model,
        "feature_importances_"
    ):

        print(
            "Model does not provide "
            "feature_importances_."
        )

    else:

        importance_df = pd.DataFrame({

            "Feature":
                feature_names,

            "Importance":
                model.feature_importances_

        })


        importance_df = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
        )


        importance_path = (
            RESULT_DIR
            / "feature_importance.csv"
        )


        importance_df.to_csv(
            importance_path,
            index=False
        )


        print(
            f"Saved: "
            f"{importance_path}"
        )


        # ----------------------------------------------------
        # TOP 20
        # ----------------------------------------------------

        print(
            "\nTop 20 important features:"
        )


        print(
            importance_df
            .head(20)
            .to_string(
                index=False
            )
        )


        # ----------------------------------------------------
        # FEATURE IMPORTANCE PLOT
        # ----------------------------------------------------

        top_features = (
            importance_df
            .head(20)
        )


        plt.figure(
            figsize=(10, 8)
        )


        plt.barh(
            top_features["Feature"][::-1],
            top_features["Importance"][::-1]
        )


        plt.xlabel(
            "Importance"
        )


        plt.ylabel(
            "Feature"
        )


        plt.title(
            "Top 20 Important Features - XGBoost"
        )


        plt.tight_layout()


        feature_plot_path = (
            RESULT_DIR
            / "feature_importance.png"
        )


        plt.savefig(
            feature_plot_path,
            dpi=300
        )


        plt.close()


        print(
            f"Saved: "
            f"{feature_plot_path}"
        )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_df = pd.DataFrame([

        {
            "Model":
                "XGBoost",

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "ROC_AUC":
                roc_auc,

            "PR_AUC":
                pr_auc
        }

    ])


    metrics_path = (
        RESULT_DIR
        / "xgboost_evaluation_metrics.csv"
    )


    metrics_df.to_csv(
        metrics_path,
        index=False
    )


    print(
        f"\nSaved: "
        f"{metrics_path}"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )


    print(
        "\n"
        + metrics_df.to_string(
            index=False
        )
    )


    # ========================================================
    # GENERATED FILES
    # ========================================================

    print(
        "\nGenerated files:"
    )


    for file in sorted(
        RESULT_DIR.iterdir()
    ):

        if file.is_file():

            print(
                f" - {file.name}"
            )


    print(
        "\n" + "=" * 70
    )

    print(
        "CYBERGUARD-AI EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()