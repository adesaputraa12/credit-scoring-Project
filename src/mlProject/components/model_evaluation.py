import joblib
import json
import numpy as np
import pandas as pd
import scorecardpy as sc
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.constants import TARGET_COLUMN
from src.mlProject.logging import logger


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _load_artifacts(self):
        train = pd.read_csv(self.config.train_data_path)
        test = pd.read_csv(self.config.test_data_path)
        train_raw = pd.read_csv(self.config.train_raw_path)
        test_raw = pd.read_csv(self.config.test_raw_path)
        model = joblib.load(self.config.model_path)
        bins = joblib.load(self.config.woe_bins_path)

        scorecard_path = Path(self.config.scorecard_path)
        card = joblib.load(scorecard_path) if scorecard_path.exists() else None

        with open(self.config.feature_names_path) as f:
            feature_cols = json.load(f)

        missing_cols = [c for c in feature_cols if c not in train.columns]
        if missing_cols:
            raise ValueError(f"Feature tidak ditemukan pada dataset: {missing_cols}")

        logger.info(f"Train: {train.shape} | Test: {test.shape}")
        logger.info(f"Features: {len(feature_cols)} | Scorecard: {'✅' if card else '❌'}")
        return train, test, train_raw, test_raw, model, bins, card, feature_cols

    def _get_scores(self, model, X):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)

            if hasattr(model, "classes_") and len(model.classes_) > 1:
                classes = list(model.classes_)
                if 1 in classes:
                    pos_idx = classes.index(1)
                else:
                    pos_idx = 1 if proba.shape[1] > 1 else 0
            else:
                pos_idx = 1 if proba.shape[1] > 1 else 0

            return proba[:, pos_idx]

        if hasattr(model, "decision_function"):
            return model.decision_function(X)

        raise ValueError(f"Model {type(model).__name__} tidak punya predict_proba atau decision_function")

    def _compute_metrics(self, y_true, y_score, label: str) -> dict:
        auc = roc_auc_score(y_true, y_score)
        gini = 2 * auc - 1
        ap = average_precision_score(y_true, y_score)

        try:
            perf = sc.perf_eva(y_true, y_score, title=label, show_plot=False)
            ks = float(perf.get("KS", 0))
        except Exception as e:
            logger.info(f"KS tidak dapat dihitung [{label}]: {e}")
            ks = 0.0

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        best_idx = np.argmax(tpr - fpr)
        best_thresh = float(thresholds[best_idx])

        y_pred = (y_score >= best_thresh).astype(int)

        acc = float((y_pred == y_true).mean())
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        logger.info(
            f"[{label}] AUC={auc:.4f} | KS={ks:.4f} | Gini={gini:.4f} | "
            f"Threshold={best_thresh:.4f} | F1={f1:.4f}"
        )

        return {
            "auc": round(auc, 4),
            "ks": round(ks, 4),
            "gini": round(gini, 4),
            "ap_score": round(ap, 4),
            "optimal_threshold": round(best_thresh, 4),
            "accuracy": round(acc, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "n_samples": int(len(y_true)),
            "n_bad": int(y_true.sum()),
            "bad_rate": round(float(y_true.mean()), 4),
        }

    def _compute_psi(self, train_score, test_score, y_train, y_test) -> float | None:
        try:
            psi_result = sc.perf_psi(
                score={"train": train_score, "test": test_score},
                label={"train": y_train, "test": y_test},
                show_plot=False,
            )
            psi_val = float(psi_result["psi"]["PSI"].iloc[0])
            status = "Normal ✅" if psi_val < 0.1 else "Perhatian ⚠" if psi_val < 0.25 else "Don't Normal ❌"
            logger.info(f"PSI = {psi_val:.4f} → {status}")
            return psi_val
        except Exception as e:
            logger.info(f"PSI tidak dapat dihitung: {e}")
            return None

    def _get_iv_summary(self, bins: dict) -> dict:
        iv_dict = {}
        for feat, df_bin in bins.items():
            try:
                if "total_iv" in df_bin.columns:
                    iv_dict[feat] = round(float(df_bin["total_iv"].iloc[0]), 4)
                elif "bin_iv" in df_bin.columns:
                    iv_dict[feat] = round(float(df_bin["bin_iv"].sum()), 4)
            except Exception:
                pass
        return iv_dict

    def _build_prediction_artifact(self, df, y_true, y_score, split_name: str, threshold: float) -> pd.DataFrame:
        pred_df = pd.DataFrame({
            "split": split_name,
            "actual": y_true.values,
            "probability": y_score,
            "prediction": (y_score >= threshold).astype(int),
        })
        return pred_df

    def initiate_model_evaluation(self):
        train, test, train_raw, test_raw, model, bins, card, feature_cols = self._load_artifacts()

        X_train, y_train = train[feature_cols], train[TARGET_COLUMN]
        X_test, y_test = test[feature_cols], test[TARGET_COLUMN]

        train_score = self._get_scores(model, X_train)
        test_score = self._get_scores(model, X_test)

        train_metrics = self._compute_metrics(y_train, train_score, "train")
        test_metrics = self._compute_metrics(y_test, test_score, "test")

        overfit_gap = round(abs(train_metrics["auc"] - test_metrics["auc"]), 4)
        if overfit_gap > 0.05:
            logger.info(f"⚠ Overfit gap = {overfit_gap} — pertimbangkan turunkan C")

        psi_val = None
        if card is not None:
            try:
                train_scorecard = sc.scorecard_ply(train_raw, card, only_total_score=True)
                test_scorecard = sc.scorecard_ply(test_raw, card, only_total_score=True)
                psi_val = self._compute_psi(train_scorecard, test_scorecard, y_train, y_test)
            except Exception as e:
                logger.error(f"Scorecard scoring gagal: {e}")

        iv_summary = self._get_iv_summary(bins)

        root = Path(self.config.root_dir)
        root.mkdir(parents=True, exist_ok=True)

        threshold = test_metrics["optimal_threshold"]

        train_pred_df = self._build_prediction_artifact(
            train, y_train, train_score, "train", threshold
        )
        test_pred_df = self._build_prediction_artifact(
            test, y_test, test_score, "test", threshold
        )

        prediction_df = pd.concat([train_pred_df, test_pred_df], ignore_index=True)
        prediction_path = root / "prediction_results.csv"
        prediction_df.to_csv(prediction_path, index=False)
        logger.info(f"Prediction artifact disimpan: {prediction_path}")

        full_metrics = {
            "model_name": type(model).__name__,
            "train": train_metrics,
            "test": test_metrics,
            "overfit_gap": overfit_gap,
            "psi": round(psi_val, 4) if psi_val is not None else None,
            "iv_summary": iv_summary,
            "model_params": dict(self.config.all_params),
            "prediction_artifact": str(prediction_path),
        }

        with open(self.config.metrics_file_path, "w") as f:
            json.dump(full_metrics, f, indent=4)
        logger.info(f"Metrics disimpan: {self.config.metrics_file_path}")

        report_df = pd.DataFrame([
            {"split": "train", **train_metrics},
            {"split": "test", **test_metrics},
        ])
        report_path = root / "evaluation_report.csv"
        report_df.to_csv(report_path, index=False)
        logger.info(f"Report disimpan: {report_path}")

        logger.info(
            f"── EVALUATION SUMMARY ──\n"
            f"  AUC       : {test_metrics['auc']}\n"
            f"  KS        : {test_metrics['ks']}\n"
            f"  Gini      : {test_metrics['gini']}\n"
            f"  Accuracy  : {test_metrics['accuracy']}\n"
            f"  Precision : {test_metrics['precision']}\n"
            f"  Recall    : {test_metrics['recall']}\n"
            f"  F1 Score  : {test_metrics['f1_score']}\n"
            f"  PSI       : {psi_val}\n"
            f"  Overfit   : {overfit_gap}"
        )

        return full_metrics