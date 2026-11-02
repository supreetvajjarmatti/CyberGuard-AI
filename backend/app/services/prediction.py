from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"
FEATURE_PATH = MODEL_DIR / "feature_names.joblib"


class CyberGuardPredictor:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        if not FEATURE_PATH.exists():
            raise FileNotFoundError(
                f"Feature names not found: {FEATURE_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)
        self.feature_names = joblib.load(FEATURE_PATH)

        print("CyberGuard-AI model loaded successfully.")
        print(f"Features expected: {len(self.feature_names)}")

    def predict(self, input_features):

        # Build input in the exact training feature order.
        row = {}

        for feature in self.feature_names:
            row[feature] = input_features.get(feature, 0.0)

        data = pd.DataFrame(
            [row],
            columns=self.feature_names
        )

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        if data.isnull().values.any():
            raise ValueError(
                "Input contains missing or invalid feature values."
            )

        if not np.isfinite(data.to_numpy(dtype=float)).all():
            raise ValueError(
                "Input contains non-finite values."
            )

        prediction = int(
            self.model.predict(data)[0]
        )

        probabilities = self.model.predict_proba(data)[0]

        confidence = float(
            probabilities[prediction]
        ) * 100

        if prediction == 1:
            result = "ATTACK"

            if confidence >= 90:
                risk = "HIGH"
            elif confidence >= 70:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            message = "Potential malicious network traffic detected."

        else:
            result = "BENIGN"
            risk = "LOW"
            message = "Network traffic appears normal."

        return {
            "prediction": result,
            "target": prediction,
            "confidence": round(confidence, 2),
            "risk_level": risk,
            "message": message
        }


# Load model once when the API starts.
predictor = CyberGuardPredictor()