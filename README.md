# Diabetes Risk Prediction Demo

An educational machine-learning project that explores the Pima Indians Diabetes Database and presents a small interactive Streamlit application.

## Project status

This repository is a cleaned portfolio edition of a university team project. It focuses on a reproducible training pipeline and removes presentation files, personal information, downloaded reference notebooks, and unsupported medical claims.

## Features

- Loads and validates the public dataset
- Treats medically implausible zero values as missing data
- Fits imputation only on the training split to reduce data leakage
- Trains a class-balanced random forest classifier
- Reports accuracy, precision, recall, F1 score, and a confusion matrix
- Provides an interactive example prediction in Streamlit

## Dataset

The application expects the commonly used Pima Indians Diabetes Database with these columns:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

Download the dataset from its original or an authorized distribution page, review its terms, and save it as:

```text
data/diabetes.csv
```

The dataset is intentionally not included in this repository until its redistribution terms are confirmed. A commonly referenced distribution is the [Kaggle Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Methods

The project uses a stratified 80/20 train-test split. Missing values are imputed with training-set medians inside a scikit-learn pipeline. The classifier is a random forest with fixed parameters and `random_state=42` for reproducibility.

The displayed metrics are calculated at runtime. They are not copied from the original course slides.

## Limitations and responsible use

This project is an educational demonstration. It is not a medical device and must not be used to diagnose disease, estimate an individual's clinical risk, or make treatment decisions. The dataset represents a limited population and does not establish general clinical validity.

## Background

The original work was completed as a university team project. This public edition does not claim sole authorship of the original team presentation or downloaded reference materials.
