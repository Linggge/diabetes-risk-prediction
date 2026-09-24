from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path(__file__).parent / "data" / "diabetes.csv"
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + ["Outcome"]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return data


@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[FEATURES].copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessing = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), FEATURES)],
        remainder="drop",
    )
    pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=6,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
    }
    matrix = confusion_matrix(y_test, predictions)
    return pipeline, metrics, matrix


def main() -> None:
    st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="📊", layout="wide")
    st.title("Diabetes Risk Prediction")
    st.warning(
        "Educational demonstration only. This application is not a medical device "
        "and must not be used for diagnosis or treatment decisions."
    )

    if not DATA_PATH.exists():
        st.error(
            "Dataset not found. Download the Pima Indians Diabetes Database and save "
            "the CSV as data/diabetes.csv. See README.md for details."
        )
        st.stop()

    try:
        data = load_data()
        model, metrics, matrix = train_model(data)
    except (OSError, ValueError) as error:
        st.error(f"Unable to load the dataset: {error}")
        st.stop()

    overview, evaluation, prediction = st.tabs(["Data", "Evaluation", "Prediction"])

    with overview:
        st.subheader("Dataset overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(data))
        col2.metric("Input features", len(FEATURES))
        col3.metric("Positive class", f"{data['Outcome'].mean():.1%}")
        st.dataframe(data.head(20), width="stretch")

    with evaluation:
        st.subheader("Holdout evaluation")
        columns = st.columns(len(metrics))
        for column, (name, value) in zip(columns, metrics.items()):
            column.metric(name, f"{value:.3f}")
        st.caption("Metrics come from a stratified 80/20 split with random_state=42.")
        st.write("Confusion matrix")
        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual negative", "Actual positive"],
                columns=["Predicted negative", "Predicted positive"],
            )
        )

    with prediction:
        st.subheader("Try one example")
        left, right = st.columns(2)
        values = {
            "Pregnancies": left.number_input("Pregnancies", 0, 20, 1),
            "Glucose": left.number_input("Glucose", 0, 250, 120),
            "BloodPressure": left.number_input("Blood pressure", 0, 150, 70),
            "SkinThickness": left.number_input("Skin thickness", 0, 100, 20),
            "Insulin": right.number_input("Insulin", 0, 900, 80),
            "BMI": right.number_input("BMI", 0.0, 70.0, 25.0, step=0.1),
            "DiabetesPedigreeFunction": right.number_input(
                "Diabetes pedigree function", 0.0, 3.0, 0.5, step=0.01
            ),
            "Age": right.number_input("Age", 18, 100, 35),
        }
        if st.button("Run educational prediction", type="primary"):
            sample = pd.DataFrame([values], columns=FEATURES)
            probability = model.predict_proba(sample)[0, 1]
            st.metric("Model probability", f"{probability:.1%}")
            st.info(
                "This number only describes the model output for this educational "
                "dataset. It is not an estimate suitable for clinical use."
            )


if __name__ == "__main__":
    main()
