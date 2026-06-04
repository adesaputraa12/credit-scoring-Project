from src.mlProject.constants import *
from src.mlProject.utils.common import read_yaml, create_directories   
from mlProject.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig,
                                            DataTransformationConfig,
                                            ModelTrainerConfig,
                                            ModelEvaluationConfig)

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

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_path=config.local_data_path,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.report_file,
            unzip_dir=config.unzip_dir,
            all_schema=self.schema.COLUMNS
        )
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir = config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            alpha = params.alpha,
            l1_ratio = params.l1_ratio,
            target_column = schema.name
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            all_params=params,
            metrics_file_path=Path(config.metrics_file_path),
            target_column=schema.name
        )
        return model_evaluation_config
    
