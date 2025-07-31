# File: src/data_processing/build_features.py

import os
import argparse
import pandas as pd
import json
import hashlib
from datetime import datetime
import logging

def setup_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(levelname)s %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def compute_schema_hash(df: pd.DataFrame) -> str:
    schema = df.dtypes.astype(str).to_dict()
    schema_str = json.dumps(schema, sort_keys=True).encode()
    return hashlib.md5(schema_str).hexdigest()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: insert your domain feature transforms here
    # Example placeholder:
    # df["sensor_diff"] = df["sensor_reading"].diff().fillna(0)
    return df

def main(input_file: str, output_dir: str, manifest_path: str):
    logger = setup_logger()
    start_time = datetime.utcnow().isoformat()

    logger.info(f"Reading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Applying feature transformations")
    df_feat = build_features(df)

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "features.csv")
    df_feat.to_csv(out_csv, index=False)
    row_count = len(df_feat)
    schema_hash = compute_schema_hash(df_feat)
    end_time = datetime.utcnow().isoformat()

    manifest = {
        "output_file": out_csv,
        "row_count": row_count,
        "schema_hash": schema_hash,
        "start_time": start_time,
        "end_time": end_time
    }
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    logger.info(f"Wrote features ({row_count} rows) to {out_csv}")
    logger.info(f"Wrote manifest to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build features from interim merged CSV"
    )
    parser.add_argument(
        "--input-file", required=True,
        help="path to data/interim/merged.csv"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="directory for data/processed"
    )
    parser.add_argument(
        "--manifest", default="manifest_features.json",
        help="path for JSON manifest"
    )
    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.manifest)
