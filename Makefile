.PHONY: requirements data clean lint dvc-track dvc-run train evaluate sync_data_to_s3 sync_data_from_s3 create_environment test_environment help

PYTHON_INTERPRETER := python3
PROJECT_NAME := Transformer_fault_detection
BUCKET := [your-s3-bucket-name]   # Without s3://
PROFILE := default

ifeq (,$(shell which conda))
HAS_CONDA := False
else
HAS_CONDA := True
endif

requirements: test_environment
    $(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
    $(PYTHON_INTERPRETER) -m pip install -r requirements.txt

data: requirements
    $(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

dvc-track:
    dvc add data/raw
    dvc add data/processed
    git add data/.gitignore data/*.dvc
    git commit -m "Track raw and processed datasets via DVC"

dvc-run:
    dvc repro

train:
    $(PYTHON_INTERPRETER) src/pipelines/train_pipeline.py

test:
    $(PYTHON_INTERPRETER) -m pytest tests/


evaluate:
    $(PYTHON_INTERPRETER) src/models/evaluate_model.py models/model.pkl data/features reports/metrics.json

clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

lint:
    flake8 src

sync_data_to_s3:
ifeq ($(PROFILE),default)
    aws s3 sync data/ s3://$(BUCKET)/data/
else
    aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

sync_data_from_s3:
ifeq ($(PROFILE),default)
    aws s3 sync s3://$(BUCKET)/data/ data/
else
    aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

create_environment:
ifeq ($(HAS_CONDA),True)
    conda create --name $(PROJECT_NAME) python=3
    @echo "Activate with: conda activate $(PROJECT_NAME)"
else
    $(PYTHON_INTERPRETER) -m pip install virtualenv virtualenvwrapper
    @bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
    @echo "Activate with: workon $(PROJECT_NAME)"
endif

test_environment:
    $(PYTHON_INTERPRETER) test_environment.py

.DEFAULT_GOAL := help
help:
    @echo "Available targets:"
    @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
