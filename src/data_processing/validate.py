#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
from jsonschema import validate, ValidationError
from utils.util import utility as UT

def validate_rows(df: pd.DataFrame, schema: dict) -> list[dict]:
    """Validate DataFrame rows against schema using UT logging"""
    issues = []
    for idx, row in df.iterrows():
        try:
            validate(instance=row.to_dict(), schema=schema)
        except ValidationError as e:
            issues.append({
                'row_index': idx,
                'message': str(e),
                'path': list(getattr(e, 'path', []))
            })
            UT.get_logger(__name__).debug(f"Row {idx} error: {str(e)}")
    return issues

def main():
    parser = argparse.ArgumentParser(
        description="Validate CSV against JSON Schema"
    )
    parser.add_argument("csv", help="Input CSV path")
    parser.add_argument("--schema", required=True, help="JSON Schema path")
    parser.add_argument("--report", required=True, help="Output report path")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    logger = UT.get_logger(__name__)
    logger.info(f"Validating {args.csv}")

    try:
        # UT-managed operations
        df = UT.load_csv(args.csv)
        schema = UT.load_json(args.schema)
        
        issues = validate_rows(df, schema)
        report = {
            'valid': len(issues) == 0,
            'row_count': len(df),
            'error_count': len(issues),
            'errors': issues,
            'schema_hash': UT.compute_schema_hash(df)  # Added schema validation
        }
        
        UT.save_json(report, args.report)  # Replaces manual json.dump
        logger.info(f"Report saved to {args.report}")

        if issues and args.fail_on_error:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise UT.SchemaError(f"Validation error: {str(e)}") from e

if __name__ == "__main__":
    main()