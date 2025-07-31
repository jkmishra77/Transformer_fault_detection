#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def main(input_dir, output_dir):
    """
    Reads CurrentVoltage.csv and Overview.csv from input_dir,
    merges on DeviceTimeStamp, drops NA, sanitizes column names,
    and writes merged.csv to output_dir.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data merge from %s to %s", input_dir, output_dir)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    try:
        # Read raw files
        df_cv = pd.read_csv(input_path / "CurrentVoltage.csv")
        df_ov = pd.read_csv(input_path / "Overview.csv")

        # Timestamp formatting
        df_cv["DeviceTimeStamp"] = pd.to_datetime(df_cv["DeviceTimeStamp"])
        df_ov["DeviceTimeStamp"] = pd.to_datetime(df_ov["DeviceTimeStamp"])

        # Merge and sanitize
        df_merged = pd.merge(df_cv, df_ov, on="DeviceTimeStamp", how="inner")
        df_merged.dropna(inplace=True)
        df_merged.columns = [
            c.strip().lower().replace(" ", "_")
            for c in df_merged.columns
        ]

        # Write to output
        output_path.mkdir(parents=True, exist_ok=True)
        merged_path = output_path / "merged.csv"
        df_merged.to_csv(merged_path, index=False)
        logger.info("Merged data stored at %s", merged_path)

    except Exception as e:
        logger.error("Merge failed: %s", e)
        raise RuntimeError(f"Data processing failed: {e}")

if __name__ == "__main__":
    # load .env if present
    load_dotenv(find_dotenv())

    # basic logging
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # simple CLI: two positional args
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output_dir>")
        sys.exit(1)

    _, input_dir, output_dir = sys.argv
    main(input_dir, output_dir)
