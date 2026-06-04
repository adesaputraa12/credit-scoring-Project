import shutil
from src.mlProject.entity.config_entity import DataIngestionConfig
from src.mlProject.logging import logger

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def copy_data(self):
        shutil.copy(self.config.local_data_path, self.config.root_dir)  # ← local_data_path
        logger.info(f"Data copied to {self.config.root_dir}")