# src/utils/util.py
import logging
from datetime import datetime
from pathlib import Path

class utility:
    @staticmethod
    def create_dir(path: str) -> Path:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj

    @staticmethod
    def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
        log_path = UT.create_dir(log_dir)
        log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    # Other methods (io, exceptions) remain unchanged...