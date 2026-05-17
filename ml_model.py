import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

FEATURE_COLS = ["vibration_rms", "temperature_c", "rpm", "load"]


class BearingHealthModel:
    def __init__(self, model_type: str = "svm"):
        model_type = model_type.lower()
        if model_type == "xgboost":
            if not _XGBOOST_AVAILABLE:
                raise ImportError("xgboost is not installed. Run: pip install xgboost")
            self.model = XGBClassifier(n_estimators=100, max_depth=4,
                                       eval_metric="logloss", random_state=42)
        elif model_type == "svm":
            self.model = SVC(kernel="rbf", probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'svm' or 'xgboost'.")

        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLS
        self._trained = False

    def _extract(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        return df[self.feature_cols].values

    def train(self, df: pd.DataFrame) -> "BearingHealthModel":
        X = self.scaler.fit_transform(self._extract(df))
        y = df["label"].values
        self.model.fit(X, y)
        self._trained = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model has not been trained yet.")
        X = self.scaler.transform(self._extract(df))
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns health score (0 = critical, 1 = healthy)."""
        if not self._trained:
            raise RuntimeError("Model has not been trained yet.")
        X = self.scaler.transform(self._extract(df))
        # column 0 = P(healthy), column 1 = P(faulty)
        return self.model.predict_proba(X)[:, 0]

    def evaluate(self, df: pd.DataFrame) -> dict:
        y_true = df["label"].values
        y_pred = self.predict(df)
        return {
            "accuracy":         round(accuracy_score(y_true, y_pred), 4),
            "f1_score":         round(f1_score(y_true, y_pred), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "model_type": self.model_type, "feature_cols": self.feature_cols}, path)

    def load(self, path: str) -> "BearingHealthModel":
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.model_type = data["model_type"]
        self.feature_cols = data["feature_cols"]
        self._trained = True
        return self
