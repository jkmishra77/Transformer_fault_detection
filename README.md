# Transformer_fault_detection
1. python src/data_processing/make_dataset.py data/raw data/interim
2. python src/data_processing/validate.py data/interim/merged.csv --schema config/schema.json --report reports/validation_report.json
3. python src/features/build_features.py --input-file data/interim/merged.csv --output-dir data/processed --manifest manifest_features.json # need to delete existing file
4. python src/models/train.py \
  --input-file data/processed/features.csv \
  --config config/params.json \
  --output-model-dir models/ \
  --metrics-file reports/train_metrics.json

5.python src/models/train_tune.py \
  --input-file data/processed/features.csv \
  --config config/params.json \
  --output-model-dir models/ \
  --metrics-file reports/tune_metrics.json

6. python src/models/evaluate.py \
  --model models/best_model.joblib \
  --test-data data/processed/validate.csv

7. python src/models/predict.py \
  --model models/best_model.joblib \
  --input data/processed/new_data.csv \
  --output predictions.csv