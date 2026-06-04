from src.mlProject.logging import logger
from pathlib import Path
from src.mlProject.components.data_preprocessing import DataPreprocessing
from src.mlProject.config.configuration import ConfigurationManager

STAGE_NAME = "Data Transformation Stage"

class DataPreprocessingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def main(self):
        config = self.config_manager.get_data_preprocessing_config()
        preprocessing = DataPreprocessing(config)
        artifact = preprocessing.initiate_data_preprocessing()
        print(f"Data preprocessing selesai: {artifact}")