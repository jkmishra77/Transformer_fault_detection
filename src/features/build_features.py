#!/usr/bin/env python3
import argparse
import pandas as pd
from utils.util import utility as UT


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic feature engineering placeholder"""
    df = df.copy()
    # Example: convert timestamp to datetime features if present
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["datetime"].dt.hour
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df.drop(columns=["datetime"], inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Feature engineering script")
    parser.add_argument("--input-file", required=True, help="Path to input CSV file")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    parser.add_argument("--manifest", required=True, help="Path to output manifest JSON")
    args = parser.parse_args()

    logger = UT.get_logger()
    logger.info(f"Reading data from {args.input_file}")

    df = UT.load_csv(args.input_file)

    logger.info("Applying feature transformations")
    processed_df = build_features(df)

    output_path = f"{args.output_dir}/features.csv"
    UT.save_csv(processed_df, output_path)

    UT.save_json({"features": list(processed_df.columns)}, args.manifest)

    logger.info(f"Wrote features ({len(processed_df)} rows) to {output_path}")
    logger.info(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
