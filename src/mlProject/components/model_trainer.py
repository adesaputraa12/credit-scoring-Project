import joblib
import json
import numpy as np
import pandas as pd
import scorecardpy as sc
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.mlProject.entity.config_entity import ModelTrainerConfig
from src.mlProject.constants import TARGET_COLUMN
from src.mlProject.logging import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _load_data(self):
        train = pd.read_csv(self.config.train_data_path)
        test  = pd.read_csv(self.config.test_data_path)
        logger.info(f"Train: {train.shape} | Test: {test.shape}")
        return train, test

    def _check_class_distribution(self, y: pd.Series, label: str):
        dist = y.value_counts(normalize=True).round(4) * 100
        logger.info(f"Class distribution [{label}]:")
        for cls, pct in dist.items():
            logger.info(f"  Kelas {cls}: {pct:.2f}%")

        if dist.min() < 20 and self.config.class_weight is not None:
            logger.info("  ⚠ Imbalance terdeteksi — class_weight aktif")

    def _log_coefficients(self, model: LogisticRegression, features: list):
        coef_df = pd.DataFrame({
            "feature": features,
            "coefficient": model.coef_[0],
            "odds_ratio": np.exp(model.coef_[0]),
        }).sort_values("coefficient", ascending=False)

        logger.info("Top 5 bad-risk features:")
        for _, row in coef_df.head(5).iterrows():
            logger.info(
                f"  {row['feature']:35s} "
                f"coef={row['coefficient']:+.4f}  OR={row['odds_ratio']:.4f}"
            )
        return coef_df

    def initiate_model_training(self):
        train, test = self._load_data()

        if TARGET_COLUMN not in train.columns:
            raise ValueError(f"TARGET '{TARGET_COLUMN}' tidak ada di train")
        if train[TARGET_COLUMN].isna().all():
            raise ValueError("Target kosong di train")

        feature_cols = [c for c in train.columns if c != TARGET_COLUMN]
        X_train, y_train = train[feature_cols], train[TARGET_COLUMN]
        X_test,  y_test  = test[feature_cols],  test[TARGET_COLUMN]

        logger.info(f"WOE features: {len(feature_cols)}")
        self._check_class_distribution(y_train, "train")

        logger.info("Training Logistic Regression ...")
        model = LogisticRegression(
            C=self.config.C,
            max_iter=self.config.max_iter,
            solver=self.config.solver,
            class_weight=self.config.class_weight,
            random_state=42,
        )
        model.fit(X_train, y_train)
        logger.info(f"Training selesai ✅ iterasi={model.n_iter_[0]}")

        auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        auc_test  = roc_auc_score(y_test,  model.predict_proba(X_test)[:, 1])
        logger.info(f"AUC Train={auc_train:.4f} | AUC Test={auc_test:.4f} | Gap={auc_train-auc_test:.4f}")

        coef_df = self._log_coefficients(model, feature_cols)

        root = Path(self.config.root_dir)
        root.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, root / self.config.model_name)
        logger.info(f"Model    : {root / self.config.model_name}")

        with open(root / "feature_names.json", "w") as f:
            json.dump(feature_cols, f, indent=2)

        coef_df.to_csv(root / "coefficients.csv", index=False)

        try:
            bins = joblib.load(self.config.woe_bins_path)
            card = sc.scorecard(bins, model, feature_cols)
            joblib.dump(card, root / "scorecard.pkl")
            logger.info(f"Scorecard: {root / 'scorecard.pkl'}")
        except Exception as e:
            logger.info(f"Scorecard tidak dapat dibuat: {e}")

        return {
            "model_path": str(root / self.config.model_name),
            "n_features": len(feature_cols),
            "n_iter":     int(model.n_iter_[0]),
            "auc_train":  round(float(auc_train), 4),
            "auc_test":   round(float(auc_test),  4),
        }