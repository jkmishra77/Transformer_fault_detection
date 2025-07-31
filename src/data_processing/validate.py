#!/usr/bin/env python3
import sys
import json
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# 1. Logging setup
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

def load_schema(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)

def validate_df(df: pd.DataFrame, schema: dict) -> dict:
    report = {"missing_columns": [],
              "type_mismatches": [],
              "range_violations": []}

    required = schema["required_columns"]
    col_types = schema["columns"]
    ranges   = schema.get("value_ranges", {})

    # missing columns
    for c in required:
        if c not in df:
            report["missing_columns"].append(c)

    # dtype mismatches
    for c, expected in col_types.items():
        if c in df:
            actual = str(df[c].dtype)
            if actual != expected:
                report["type_mismatches"].append({
                    "column": c,
                    "expected": expected,
                    "actual": actual
                })

    # out-of-range values
    for c, bounds in ranges.items():
        if c in df:
            mask = (df[c] < bounds["min"]) | (df[c] > bounds["max"])
            viol = df.loc[mask, c]
            if not viol.empty:
                report["range_violations"].append({
                    "column": c,
                    "violations": int(viol.size),
                    "examples": viol.head(3).tolist()
                })

    return report

def clean_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: clean_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_numpy(v) for v in obj]
    return obj

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate a CSV against JSON schema and emit a report."
    )
    p.add_argument("csv",        type=Path, help="Path to input CSV file")
    p.add_argument("--schema",   type=Path, help="JSON schema path", required=True)
    p.add_argument("--report",   type=Path, help="Output report path", required=True)
    return p.parse_args()

def main():
    args = parse_args()

    logging.info(f"🔍 Validating {args.csv}")
    df     = pd.read_csv(args.csv)
    schema = load_schema(args.schema)

    report = validate_df(df, schema)

    # write JSON
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as f:
        json.dump(clean_numpy(report), f, indent=2)

    logging.info(f"✅ Report saved: {args.report}")

    # exit with code if any issues
    if any(report.values()):
        logging.warning("❗ Validation issues found")
        sys.exit(1)

    logging.info("✅ All checks passed")
    sys.exit(0)

if __name__ == "__main__":
    main()
