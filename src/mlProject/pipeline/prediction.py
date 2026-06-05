import joblib
import json
import numpy as np
import pandas as pd
import scorecardpy as sc
from pathlib import Path

from src.mlProject.constants import TARGET_COLUMN
from src.mlProject.logging import logger


# ── Artifact paths ────────────────────────────────────────────────────────────
MODEL_PATH         = Path("artifacts/model_trainer/model.pkl")
BINS_PATH          = Path("artifacts/data_preprocessing/woe_bins.pkl")
SCORECARD_PATH     = Path("artifacts/model_trainer/scorecard.pkl")
FEATURE_NAMES_PATH = Path("artifacts/model_trainer/feature_names.json")
METRICS_PATH       = Path("artifacts/model_evaluation/metrics.json")
TRAIN_METRICS_PATH = Path("artifacts/model_trainer/train_metrics.json")


class PredictionPipeline:
    """
    Pipeline untuk prediksi single nasabah dari raw input.

    Alur:
      raw input → DataFrame → WOE transform → select features → predict → score
    """

    _instance = None   # singleton cache

    def __init__(self):
        self.model        = None
        self.bins         = None
        self.scorecard    = None
        self.feature_cols = None
        self._loaded      = False

    # ── Singleton loader ──────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls):
        if cls._instance is None or not cls._instance._loaded:
            cls._instance = cls()
            cls._instance._load_artifacts()
        return cls._instance

    def _load_artifacts(self):
        missing = []

        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded: {MODEL_PATH}")
        else:
            missing.append(str(MODEL_PATH))

        if BINS_PATH.exists():
            self.bins = joblib.load(BINS_PATH)
            logger.info(f"Bins loaded : {BINS_PATH}")
        else:
            missing.append(str(BINS_PATH))

        if FEATURE_NAMES_PATH.exists():
            with open(FEATURE_NAMES_PATH) as f:
                self.feature_cols = json.load(f)
            logger.info(f"Features   : {len(self.feature_cols)} WOE features")
        else:
            missing.append(str(FEATURE_NAMES_PATH))

        if SCORECARD_PATH.exists():
            self.scorecard = joblib.load(SCORECARD_PATH)
            logger.info(f"Scorecard  : {SCORECARD_PATH}")
        else:
            logger.info("Scorecard tidak ditemukan — credit score tidak tersedia")

        if missing:
            raise FileNotFoundError(
                f"Artifacts berikut tidak ditemukan: {missing}. "
                f"Jalankan training pipeline terlebih dahulu."
            )

        self._loaded = True

    def is_ready(self) -> bool:
        return self._loaded and self.model is not None and self.bins is not None

    # ── Input cleaning ────────────────────────────────────────────────────────
    def _prepare_input(self, raw: dict) -> pd.DataFrame:
        """
        Bersihkan dan bentuk DataFrame dari raw dict input.
        Kolom yang tidak ada diisi NaN (bins/WOE akan handle).
        """
        df = pd.DataFrame([raw])

        # Numeric coerce
        numeric_cols = [
            "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
            "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
            "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
            "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
            "Total_EMI_per_month", "Amount_invested_monthly",
            "Monthly_Balance", "Credit_History_Age",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, raw_input: dict) -> dict:
        if not self.is_ready():
            raise RuntimeError("Artifacts belum di-load. Pastikan training sudah selesai.")

        # 1. Prepare
        df = self._prepare_input(raw_input)

        # 2. WOE transform
        df_woe = sc.woebin_ply(df, self.bins)

        # 3. Align features
        missing_feats = [c for c in self.feature_cols if c not in df_woe.columns]
        if missing_feats:
            logger.info(f"Kolom WOE tidak ada, diisi 0: {missing_feats}")
            for col in missing_feats:
                df_woe[col] = 0.0

        X = df_woe[self.feature_cols]

        # 4. Predict
        prob       = self.model.predict_proba(X)[0]
        prediction = int(self.model.predict(X)[0])
        prob_bad   = float(prob[1])
        prob_good  = float(prob[0])

        # 5. Credit score dari scorecard
        credit_score = None
        if self.scorecard is not None:
            try:
                score_df     = sc.scorecard_ply(df, self.scorecard, only_total_score=True)
                credit_score = int(score_df["score"].values[0])
            except Exception as e:
                logger.info(f"Credit score gagal dihitung: {e}")

        # 6. Risk level
        if prob_bad < 0.3:
            risk_level = "Low Risk"
        elif prob_bad < 0.6:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        label_map = {0: "Good / Standard", 1: "Poor"}

        return {
            "prediction":   prediction,
            "label":        label_map[prediction],
            "risk_level":   risk_level,
            "probability": {
                "bad":  round(prob_bad,  4),
                "good": round(prob_good, 4),
            },
            "credit_score": credit_score,
        }