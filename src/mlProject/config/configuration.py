from src.mlProject.constants import *
from src.mlProject.utils.common import read_yaml, create_directories   
from mlProject.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig,
                                            DataPreprocessingConfig,
                                            )

class ConfigurationManager:
    def __init__(
        self,
        config_file_path: Path = CONFIG_FILE_PATH,
        params_file_path: Path = PARAMS_FILE_PATH,
        schema_file_path: Path = SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
    
        return DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_path=config.local_data_path,
        )
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.report_file,
            data_path=config.data_path,  # ← tambah ini
            all_schema=self.schema.COLUMNS
        )
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        preprocessing_dir = Path(self.config.artifacts_root) / "data_preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)

        return DataPreprocessingConfig(
            root_dir=preprocessing_dir,
            data_path=Path(config.data_path),
            processed_train_path=preprocessing_dir / "train.csv",
            processed_test_path=preprocessing_dir / "test.csv",
        )