import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

DATA_CANDIDATES = ["heart.csv", "heart(1).csv"]
MODEL_FILE = "Logistic_Regression_heart.pkl"
SCALER_FILE = "scaler.pkl"
COLUMNS_FILE = "columns.pkl"
NUMERIC_COLUMNS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
TARGET = "HeartDisease"


@st.cache_resource
def load_and_train_model():
    """Train a consistent model/scaler pair from the project's dataset.

    The notebook currently saves a scaler that was fit on the test set, so this
    app deliberately refits the preprocessing correctly rather than relying on
    potentially mismatched saved artifacts.
    """
    csv_path = next((Path(p) for p in DATA_CANDIDATES if Path(p).exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "Dataset not found. Put heart.csv (or heart(1).csv) in the same folder as app.py."
        )

    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"The dataset must contain a '{TARGET}' column.")

    # Follow the notebook's treatment of impossible zero values.
    for col in ["RestingBP", "Cholesterol"]:
        if col in df.columns:
            nonzero_mean = df.loc[df[col] != 0, col].mean()
            df[col] = df[col].replace(0, nonzero_mean)

    X_raw = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].astype(int)

    # Keep the same dummy-variable approach used in the Sheryians notebook.
    X = pd.get_dummies(X_raw, drop_first=True)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    # Save a matching set of artifacts for reference/reuse.
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_columns, COLUMNS_FILE)

    return model, scaler, feature_columns, X_raw, test_accuracy


st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the health measurements below to see the prediction from a Logistic Regression model."
)
st.caption(
    "Educational project only. This prediction is not a medical diagnosis or a substitute for a clinician's advice."
)

try:
    model, scaler, feature_columns, X_reference, test_accuracy = load_and_train_model()
except Exception as exc:
    st.error(f"Could not load the project dataset or train the model: {exc}")
    st.info("Keep app.py and heart.csv in the same project folder, then run: streamlit run app.py")
    st.stop()

with st.form("prediction_form"):
    age = st.slider("Age", min_value=18, max_value=100, value=40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=50, max_value=700, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Maximum Heart Rate", min_value=50, max_value=230, value=150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Replace zero values in the same way as the training data.
    if resting_bp == 0:
        resting_bp = float(X_reference["RestingBP"].mean())
    if cholesterol == 0:
        cholesterol = float(X_reference["Cholesterol"].mean())

    user_row = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }])

    # Encode the user row using the category vocabulary of the training data.
    combined = pd.concat([X_reference, user_row], ignore_index=True)
    encoded = pd.get_dummies(combined, drop_first=True)
    input_encoded = encoded.iloc[[-1]].reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    prediction = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0]
    class_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
    risk_score = float(probabilities[class_index])

    if prediction == 1:
        st.error("Model prediction: Heart disease indicated")
    else:
        st.success("Model prediction: Heart disease not indicated")

    st.metric("Model-estimated probability for class 1", f"{risk_score:.1%}")
    st.caption(
        f"Held-out test accuracy for this training run: {test_accuracy:.1%}. "
        "Accuracy is a dataset-level metric and does not establish the reliability of an individual prediction."
    )
