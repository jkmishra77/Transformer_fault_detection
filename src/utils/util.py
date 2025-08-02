# src/utils/util.py
import hashlib
import json
import joblib
import logging
import pandas as pd
from pathlib import Path


class SchemaError(Exception):
    """Dedicated validation error type"""
    pass


class utility:
    @staticmethod
    def get_logger(name=None) -> logging.Logger:
        """Standardized logger setup"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logging.getLogger(__name__)

    @staticmethod
    def create_dir(path: str) -> None:
        """Create directory at given path"""
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        """Load CSV into DataFrame"""
        return pd.read_csv(path)

    @staticmethod
    def save_csv(df: pd.DataFrame, path: str) -> None:
        """Save DataFrame to CSV with path creation"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def load_json(path: str) -> dict:
        """Standardized JSON loading"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_json(data: dict, path: str) -> None:
        """Validated JSON saving with path creation"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def save_model(model, path: str) -> None:
        """Save model object using joblib"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        utility.get_logger().info(f"Model saved to {path}")  # Added confirmation log

    @staticmethod
    def load_model(path: str):
        """Load model object using joblib"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file missing: {path}")  # Added validation
        return joblib.load(path)

    @staticmethod
    def compute_schema_hash(df: pd.DataFrame) -> str:
        """Generate MD5 hash from DataFrame's column-dtype schema"""
        schema = df.dtypes.astype(str).to_dict()
        schema_str = json.dumps(schema, sort_keys=True).encode()
        return hashlib.md5(schema_str).hexdigest()

    @staticmethod
    def save_metrics(metrics: dict, path: str) -> None:
        """Write evaluation or training metrics to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    @staticmethod
    def split_train_val(df: pd.DataFrame, test_size: float, random_state: int, target_column: str):
        """Deterministically split full DataFrame into train and test sets."""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)