import os
import shutil                          # ← tambah import ini
import zipfile
from mlProject.entity.config_entity import DataIngestionConfig
from mlProject.logging import logger
from mlProject.utils.common import get_size
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):           # nama method bisa tetap sama
        if not os.path.exists(self.config.local_data_file):
            shutil.copy(               # ← ganti urlretrieve → shutil.copy
                self.config.local_data_path,
                self.config.local_data_file
            )
            logger.info(f"File copied from {self.config.local_data_path} to {self.config.local_data_file}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self) -> None:  # ← tidak perlu diubah
        unzip_file_path = self.config.unzip_dir
        os.makedirs(unzip_file_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_file_path)