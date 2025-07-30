import pandas as pd, json, click, logging, numpy as np
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

def load_schema(path):
    with open(path, "r") as f:
        return json.load(f)

def validate_df(df, schema):
    report = {"missing_columns": [], "type_mismatches": [], "range_violations": []}

    expected_cols = schema["required_columns"]
    col_types = schema["columns"]
    ranges = schema.get("value_ranges", {})

    # Check required columns
    for col in expected_cols:
        if col not in df.columns:
            report["missing_columns"].append(col)

    # Check dtypes
    for col, expected in col_types.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            if actual != expected:
                report["type_mismatches"].append({"column": col, "expected": expected, "actual": actual})

    # Check ranges
    for col, bounds in ranges.items():
        if col in df.columns:
            out_of_range = df[(df[col] < bounds["min"]) | (df[col] > bounds["max"])]
            if not out_of_range.empty:
                report["range_violations"].append({
                    "column": col,
                    "violations": len(out_of_range),
                    "examples": out_of_range[col].head(3).tolist()
                })

    return report

def clean_json(obj):
    def convert(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        elif isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert(v) for v in o]
        return o
    return convert(obj)

@click.command()
@click.argument("csv", type=click.Path(exists=True))
@click.option("--schema", type=click.Path(exists=True), required=True)
@click.option("--report-path", type=click.Path(), required=True)
def main(csv, schema, report_path):
    logging.info(f"🔍 Validating: {csv}")
    df = pd.read_csv(csv)
    schema_dict = load_schema(schema)
    report = validate_df(df, schema_dict)

    with open(report_path, "w") as f:
        json.dump(clean_json(report), f, indent=2)

    logging.info(f"✅ Report saved: {report_path}")
    if any(report.values()):
        logging.warning("❗ Validation issues found")
    else:
        logging.info("✅ All checks passed")

if __name__ == "__main__":
    main()
