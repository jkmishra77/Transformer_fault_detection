#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys

import pandas as pd
from jsonschema import validate, ValidationError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a CSV file against a JSON Schema and emit a JSON report."
    )
    parser.add_argument(
        "csv",
        help="path to input CSV file",
    )
    parser.add_argument(
        "--schema",
        required=True,
        help="path to JSON Schema file",
    )
    parser.add_argument(
        "--report",
        dest="report_path",
        required=True,
        help="path to write JSON report",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="exit with code 1 if any validation issues are found",
    )
    return parser.parse_args()


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_schema(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_rows(
    df: pd.DataFrame, schema: dict, logger: logging.Logger
) -> list[dict]:
    issues = []
    for idx, row in df.iterrows():
        record = row.to_dict()
        try:
            validate(instance=record, schema=schema)
        except ValidationError as e:
            issues.append({"row_index": idx, "message": e.message})
            logger.debug(f"Row {idx} validation error: {e.message}")
    return issues


def write_report(report: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    args = parse_args()
    logger = setup_logger()

    logger.info(f"🔍 Validating data\\{args.csv}")
    df = load_data(args.csv)
    schema = load_schema(args.schema)

    issues = validate_rows(df, schema, logger)
    report = {
        "num_rows": len(df),
        "num_issues": len(issues),
        "issues": issues,
    }

    write_report(report, args.report_path)
    logger.info(f"✅ Report saved: {args.report_path}")

    if issues:
        logger.warning("❗ Validation issues found")
        if args.fail_on_error:
            sys.exit(1)
    else:
        logger.info("✅ No validation issues found")


if __name__ == "__main__":
    main()
