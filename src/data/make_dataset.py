# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Raw data directory')
@click.option('--output', '-o', required=True, type=click.Path(), help='Processed data output directory')
def main(input, output):
    """Reads raw CSVs from input path, merges by timestamp, stores in output path"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data merge from %s to %s", input, output)

    input_path = Path(input)
    output_path = Path(output)

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
        df_merged.columns = [c.strip().lower().replace(" ", "_") for c in df_merged.columns]

        # Write to output
        output_path.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(output_path / "merged.csv", index=False)
        logger.info("Merged data stored at %s", output_path / "merged.csv")

    except Exception as e:
        logger.error("Merge failed: %s", str(e))
        raise RuntimeError(f"Data processing failed: {e}")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()
