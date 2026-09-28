from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

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


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 300_000
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():
    """
    Load all feature-engineered CIC-IDS2017 files.
    """

    files = sorted(
        DATA_DIR.glob("*_features.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No feature files found in: {DATA_DIR}"
        )

    dataframes = []

    print("Loading feature datasets...")

    for file in files:

        print(f"Loading: {file.name}")

        df = pd.read_csv(
            file,
            low_memory=False
        )

        dataframes.append(df)

    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(
        f"\nTotal available rows: "
        f"{len(data):,}"
    )

    return data


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(df):
    """
    Separate features and target.

    Target:
        0 = BENIGN
        1 = ATTACK
    """

    print("\nPreparing features...")

    if "Target" not in df.columns:
        raise ValueError(
            "Target column not found."
        )

    # Target
    y = df["Target"].astype(int)

    # Remove target from features
    X = df.drop(
        columns=["Target"],
        errors="ignore"
    )

    # Keep numeric columns only
    X = X.select_dtypes(
        include=["int64", "float64"]
    )

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values using median
    X = X.fillna(
        X.median()
    )

    # Final safety check
    if X.isnull().sum().sum() > 0:
        raise ValueError(
            "Missing values still exist after preprocessing."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Infinite values still exist after preprocessing."
        )

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
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    model,
    X_test,
    y_test
):
    """
    Evaluate a trained classification model.
    """

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # Create model directory
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = load_training_data()

    # --------------------------------------------------------
    # 2. STRATIFIED SAMPLE
    # --------------------------------------------------------

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
        f"Training dataset size: "
        f"{len(df):,}"
    )

    print("\nSample target distribution:")

    print(
        df["Target"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 3. PREPARE FEATURES
    # --------------------------------------------------------

    X, y = prepare_data(df)

    # Save feature names
    feature_names = list(
        X.columns
    )

    feature_names_path = (
        MODEL_DIR
        / "feature_names.joblib"
    )

    joblib.dump(
        feature_names,
        feature_names_path
    )

    print(
        f"\nFeature names saved to:\n"
        f"{feature_names_path}"
    )

    # --------------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # 5. STANDARDIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # Save scaler
    scaler_path = (
        MODEL_DIR
        / "scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print(
        f"\nScaler saved to:\n"
        f"{scaler_path}"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    # ========================================================
    # MODEL 1 — LOGISTIC REGRESSION
    # ========================================================

    print(
        "\nTraining Logistic Regression..."
    )

    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    logistic_model.fit(
        X_train_scaled,
        y_train
    )

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic_model,
            X_test_scaled,
            y_test
        )
    )

    # ========================================================
    # MODEL 2 — DECISION TREE
    # ========================================================

    print(
        "\nTraining Decision Tree..."
    )

    decision_tree = DecisionTreeClassifier(
        max_depth=20,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    decision_tree.fit(
        X_train,
        y_train
    )

    results.append(
        evaluate_model(
            "Decision Tree",
            decision_tree,
            X_test,
            y_test
        )
    )

    # ========================================================
    # MODEL 3 — RANDOM FOREST
    # ========================================================

    print(
        "\nTraining Random Forest..."
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    random_forest.fit(
        X_train,
        y_train
    )

    results.append(
        evaluate_model(
            "Random Forest",
            random_forest,
            X_test,
            y_test
        )
    )

    # ========================================================
    # MODEL 4 — XGBOOST
    # ========================================================

    print(
        "\nTraining XGBoost..."
    )

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    xgb_model.fit(
        X_train,
        y_train
    )

    results.append(
        evaluate_model(
            "XGBoost",
            xgb_model,
            X_test,
            y_test
        )
    )

    # --------------------------------------------------------
    # SAVE XGBOOST MODEL
    # --------------------------------------------------------

    xgb_model_path = (
        MODEL_DIR
        / "xgboost_model.joblib"
    )

    joblib.dump(
        xgb_model,
        xgb_model_path
    )

    print(
        f"\nXGBoost model saved to:\n"
        f"{xgb_model_path}"
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    results_path = (
        MODEL_DIR
        / "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"\nModel comparison saved to:\n"
        f"{results_path}"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved model files:"
    )

    print(
        f"1. {xgb_model_path.name}"
    )

    print(
        f"2. {scaler_path.name}"
    )

    print(
        f"3. {feature_names_path.name}"
    )

    print(
        f"4. {results_path.name}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()