# Makefile

.PHONY: all make_dataset validate build_features

# Run entire pipeline
all: make_dataset validate build_features

# Dataset creation stage
make_dataset: data/interim/merged.csv
    dvc repro make_dataset

data/interim/merged.csv:
    dvc repro make_dataset

# Validation stage
validate: reports/validation_report.json
    dvc repro validate

reports/validation_report.json:
    dvc repro validate

# Feature‐building stage
build_features: data/processed/features.csv manifest_features.json
    dvc repro build_features

data/processed/features.csv manifest_features.json:
    dvc repro build_features
