#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd
from utils.util import utility as UT

def main(input_dir, output_dir):
    """
    Data loading pipeline using utility methods:
    - UT.load_csv() for standardized reading
    - UT.save_csv() for consistent output
    - UT.create_dir() for path creation
    - UT.SchemaError for validation failures
    """
    logger = UT.get_logger(__name__)
    logger.info("Merging data from %s to %s", input_dir, output_dir)

    try:
        # UT-managed file operations
        input_path = Path(input_dir)
        df_cv = UT.load_csv(input_path / "CurrentVoltage.csv")
        df_ov = UT.load_csv(input_path / "Overview.csv")

        # Processing pipeline
        df_merged = (
            pd.merge(
                df_cv.assign(DeviceTimeStamp=pd.to_datetime(df_cv["DeviceTimeStamp"])),
                df_ov.assign(DeviceTimeStamp=pd.to_datetime(df_ov["DeviceTimeStamp"])),
                on="DeviceTimeStamp",
                how="inner"
            )
            .dropna()
            .rename(columns=lambda x: x.strip().lower().replace(" ", "_"))
        )

        # UT-managed output with directory creation
        output_path = Path(output_dir)
        UT.create_dir(output_path)
        UT.save_csv(df_merged, output_path / "merged.csv")
        
        logger.info("Merged data saved to %s/merged.csv", output_path)
        logger.debug("Schema hash: %s", UT.compute_schema_hash(df_merged))

    except Exception as e:
        logger.error("Data merge failed: %s", e)
        raise UT.SchemaError(f"Processing error: {e}") from e

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output_dir>")
        sys.exit(1)
        
    main(*sys.argv[1:])