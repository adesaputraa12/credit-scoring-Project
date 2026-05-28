import os
import urllib.request as request
import zipfile
from mlProject.entity.config_entity import DataIngestionConfig
from mlProject.logging import logger
from mlProject.utils.common import get_size
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            file_name, headers = request.urlretrieve(
                url=self.config.source_url,
                filename=self.config.local_data_file
            )
            logger.info(f"{file_name} downloaded with following info: {headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self) -> None:
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_file_path = self.config.unzip_dir
        os.makedirs(unzip_file_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_file_path)