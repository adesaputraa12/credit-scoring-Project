from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_path: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_path: Path  
    all_schema: dict

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    data_path: Path
    processed_train_path: Path
    processed_test_path: Path
    train_raw_path: Path   # ← tambah
    test_raw_path: Path    # ← tambah

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    woe_bins_path: Path
    model_name: str
    C: float
    max_iter: int
    solver: str
    class_weight: str
    target_column: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    train_raw_path: Path        # ← tambah
    test_raw_path: Path         # ← tambah
    model_path: Path
    woe_bins_path: Path
    scorecard_path: Path
    feature_names_path: Path
    all_params: dict
    metrics_file_path: Path
    target_column: str