# src/data_processing/read_dataset.py

import os
from src.utils.util import utility, SchemaError

logger = utility.get_logger(__name__)

def read_raw_dataset(csv_path: str):
    try:
        logger.info(f"Reading raw dataset from {csv_path}")
        df = utility.load_csv(csv_path)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        return df
    except SchemaError as se:
        logger.error(f"Schema validation failed: {se}")
        raise
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        raise

def read_config(json_path: str):
    try:
        logger.info(f"Loading config from {json_path}")
        config = utility.load_json(json_path)
        logger.debug(f"Config loaded: {config}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise
