import numpy as np
import pandas as pd
import scorecardpy as sc
import joblib
from pathlib import Path

from src.mlProject.constants import (
    DROP_COLUMNS,
    FLOAT_COLUMNS,
    INTEGER_COLUMNS,
    TARGET_COLUMN,
)
from src.mlProject.entity.config_entity import DataPreprocessingConfig
from src.mlProject.utils.common import (
    clean_numeric_series,
    convert_credit_history_to_months,
    fill_missing_categorical,
    fill_missing_numeric,
    save_dataframe,
)
from src.mlProject.logging import logger


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
        )
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=DROP_COLUMNS, errors="ignore")

    def _convert_numeric_like_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_like_cols = [
            "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
            "Changed_Credit_Limit", "Outstanding_Debt",
            "Amount_invested_monthly", "Monthly_Balance",
        ]
        for col in numeric_like_cols:
            if col in df.columns:
                df[col] = clean_numeric_series(df[col])

        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age"] = convert_credit_history_to_months(df["Credit_History_Age"])

        return df

    def _cast_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in FLOAT_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in INTEGER_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if TARGET_COLUMN in numeric_cols:
            numeric_cols.remove(TARGET_COLUMN)
        df = fill_missing_numeric(df, numeric_cols)

        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if TARGET_COLUMN in categorical_cols:
            categorical_cols.remove(TARGET_COLUMN)
        df = fill_missing_categorical(df, categorical_cols, fill_value="Missing")
        return df

    def _clean_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Payment_of_Min_Amount" in df.columns:
            df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace({"NM": "Missing"})
        return df

    def _map_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if TARGET_COLUMN in df.columns:
            target_map = {
                "Poor": 1,
                "Standard": 0,
                "Good": 0,
            }

            # Kalau sudah numerik, biarkan; kalau belum, map dari string
            if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
                df[TARGET_COLUMN] = (
                    df[TARGET_COLUMN]
                    .astype(str)
                    .str.strip()
                    .map(target_map)
                )

                if df[TARGET_COLUMN].isna().all():
                    raise ValueError(
                        f"TARGET_COLUMN '{TARGET_COLUMN}' tidak cocok dengan nilai target di data"
                    )

                df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0)

            df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

        return df

    def _final_numeric_cast(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in INTEGER_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
        for col in FLOAT_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        if TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._standardize_columns(df)
        df = self._drop_columns(df)
        df = self._convert_numeric_like_columns(df)
        df = self._clean_categoricals(df)
        df = self._map_target(df)
        df = self._cast_numeric_columns(df)
        df = self._handle_missing_values(df)
        df = self._final_numeric_cast(df)
        return df

    def _ensure_target_exists(self, part: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
        """
        Kadang scorecardpy split_df mengembalikan dataframe yang targetnya tidak ikut terbawa
        di versi tertentu / kondisi tertentu. Kalau itu terjadi, kita re-attach dari source
        berdasarkan index.
        """
        if TARGET_COLUMN not in part.columns:
            if part.index.isin(source.index).all():
                part = part.copy()
                part[TARGET_COLUMN] = source.loc[part.index, TARGET_COLUMN].values
            else:
                raise ValueError(
                    f"TARGET_COLUMN '{TARGET_COLUMN}' tidak ditemukan setelah split. "
                    f"Kolom yang tersedia: {part.columns.tolist()}"
                )
        return part

    def initiate_data_preprocessing(self):
        df = pd.read_csv(self.config.data_path, low_memory=False)
        logger.info(f"Raw data loaded: {df.shape}")

        df = self._preprocess(df)
        logger.info(f"Preprocessing done: {df.shape}")

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"TARGET_COLUMN '{TARGET_COLUMN}' tidak ditemukan pada dataframe hasil preprocess. "
                f"Kolom tersedia: {df.columns.tolist()}"
            )

        split = sc.split_df(df, y=TARGET_COLUMN, ratio=0.7, seed=42)
        train, test = split["train"], split["test"]

        train = self._ensure_target_exists(train, df)
        test = self._ensure_target_exists(test, df)

        logger.info(f"Train: {train.shape}, Test: {test.shape}")

        if TARGET_COLUMN not in train.columns:
            raise ValueError(
                f"TARGET_COLUMN '{TARGET_COLUMN}' tetap tidak ada di train. "
                f"Kolom train: {train.columns.tolist()}"
            )

        if train[TARGET_COLUMN].isna().all():
            raise ValueError("Target kosong setelah preprocessing")
        
        train = sc.var_filter(train, y=TARGET_COLUMN, iv_limit=0.02)
        keep_cols = [c for c in train.columns if c in test.columns]
        test = test[keep_cols]
        logger.info(f"Setelah filter IV: {train.shape}")

        bins = sc.woebin(train, y=TARGET_COLUMN)
        logger.info("WoE binning done")

        train_woe = sc.woebin_ply(train, bins)
        test_woe = sc.woebin_ply(test, bins)

        logger.info("WoE transformation applied")

        save_dataframe(train_woe, self.config.processed_train_path)
        save_dataframe(test_woe, self.config.processed_test_path)

        bins_path = Path(self.config.root_dir) / "woe_bins.pkl"
        joblib.dump(bins, bins_path)

        logger.info(f"WoE bins saved at {bins_path}")

        return {
            "processed_train_path": str(self.config.processed_train_path),
            "processed_test_path": str(self.config.processed_test_path),
        }