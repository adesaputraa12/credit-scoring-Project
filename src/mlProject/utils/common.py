import os
from box.exceptions import BoxValueError
import yaml
from mlProject.logging import logger
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
import re
from pathlib import Path
import numpy as np
import pandas as pd

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input
    
        Raises:
            ValueError: if yaml file is empty
            e: empty file
        
        Returns:
           ConfigBox: ConfigBox type
        """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
def create_directories(path_to_directories: list[Path], verbose=True):
    """create list of directories

    Args:
        path_to_directories (list[Path]): list of path of directories to create
        verbose (bool, optional): whether to log info messages. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")
    
def save_json(path: Path, data: Any) -> None:
    """save json data to path

    Args:
        path (str): path to json file
        data (Any): data to save in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")

def load_json(path: Path) -> Any:
    """load json data from path

    Args:
        path (str): path to json file
    
    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        data = json.load(f)

    logger.info(f"json file loaded from: {path}")
    return ConfigBox(data)

def save_bin(data: Any, path: Path):
    """save binary data to path

    Args:
        data (Any): data to save in binary file
        path (str): path to binary file
    """
    joblib.dump(data, path)
    logger.info(f"binary file saved at: {path}")

def load_bin(path: Path) -> Any:
    """load binary data from path

    Args:
        path (str): path to binary file
    
    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (str): path to file
    
    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    return f"{size_in_kb} KB"

def read_csv_file(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def save_dataframe(df: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    cleaned = (
        series.astype("string")
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, "<NA>": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def convert_credit_history_to_months(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    def _convert(value):
        if pd.isna(value):
            return np.nan

        numbers = re.findall(r"\d+", str(value))
        if len(numbers) >= 2:
            years = int(numbers[0])
            months = int(numbers[1])
            return years * 12 + months
        if len(numbers) == 1:
            return int(numbers[0]) * 12
        return np.nan

    return series.apply(_convert)


def fill_missing_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def fill_missing_categorical(df: pd.DataFrame, columns: list[str], fill_value: str = "Missing") -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            s = df[col].astype("string").str.strip()
            s = s.replace(r"^_+$", np.nan, regex=True)
            s = s.replace({"<NA>": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
            df[col] = s.fillna(fill_value)
    return df