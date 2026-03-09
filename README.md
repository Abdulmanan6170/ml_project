# Student Performance Prediction (End-to-End ML Project)

This project trains a regression model to predict a student's **math score** from demographic and academic context, then serves predictions through a Flask web app.

## What This Project Includes

- Data ingestion and train/test split from `Notebook/data/stud.csv`
- Data preprocessing and model training pipeline
- Serialized artifacts for inference:
  - `artifacts/model.pkl`
  - `artifacts/preprocessor.pkl`
- Flask UI for interactive predictions:
  - `/` landing page
  - `/predictdata` prediction form

## Tech Stack

- Python
- scikit-learn
- CatBoost / XGBoost (model comparison in training utilities)
- Flask
- pandas / numpy

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- setup.py
|-- artifacts/
|-- src/
|   |-- components/
|   |   |-- data_Ingestion.py
|   |   |-- data_transformation.py
|   |   `-- model_trainer.py
|   |-- Pipeline/
|   |   `-- predict_pipeline.py
|   |-- exception.py
|   |-- logger.py
|   `-- utils.py
|-- templates/
|   |-- index.html
|   `-- home.html
`-- Notebook/
    |-- data/stud.csv
    |-- EDA.ipynb
    `-- model_train.ipynb
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Train the Model

Run the ingestion + transformation + training pipeline:

```bash
python src/components/data_Ingestion.py
```

This generates/updates files under `artifacts/` (`train.csv`, `test.csv`, `data.csv`, model and preprocessor pickles).

## Run the Flask App

```bash
python app.py
```

Then open:

- `http://127.0.0.1:5000/`
- `http://127.0.0.1:5000/predictdata`

## Input Features Used for Prediction

- `gender`
- `race_ethnicity`
- `parental_level_of_education`
- `lunch`
- `test_preparation_course`
- `reading_score`
- `writing_score`

## Notes

- Inference expects trained artifacts at `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.
- Keep feature names/order aligned with `CustomData` in `src/Pipeline/predict_pipeline.py`.
