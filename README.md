# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning to predict whether heart disease is indicated based on a person's health measurements and medical attributes.

It uses **Logistic Regression**, a supervised machine-learning algorithm, and provides an interactive web interface built with **Streamlit**. Users can enter health-related details and receive a model prediction.

This project was developed for educational purposes to practice data analysis, preprocessing, model training, and deployment of a machine-learning application.

## 🎯 Objectives

- Analyze heart disease data using Python.
- Clean and preprocess the dataset.
- Convert categorical variables into numerical features.
- Train a Logistic Regression classification model.
- Scale numerical features using StandardScaler.
- Build an interactive prediction interface using Streamlit.
- Display model predictions and an estimated class probability.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## 📂 Project Structure

```text
heart-disease-prediction-ml/
│
├── app.py
├── heart.csv
├── heart project by youtube video part 1 by sheryians ai school.ipynb
├── requirements.txt
├── README.md
│
├── Logistic_Regression_heart.pkl
├── scaler.pkl
└── columns.pkl
```

**Note:** The Streamlit app trains the model and creates the three `.pkl` files when it starts. You do not need to upload pre-existing model files for the app to train. Keep the dataset in the same folder as `app.py`.

## 📊 Dataset

The project uses a heart disease dataset containing patient-related attributes, including:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

The target column is `HeartDisease`.

- `0` — Heart disease not indicated in the dataset
- `1` — Heart disease indicated in the dataset

## ⚙️ Machine Learning Workflow

1. Load the dataset using Pandas.
2. Handle zero values in selected blood pressure and cholesterol measurements.
3. Separate the input features from the target variable.
4. Encode categorical features using one-hot encoding.
5. Split the data into training and testing sets.
6. Standardize the features using StandardScaler.
7. Train a Logistic Regression model.
8. Evaluate the model on held-out test data.
9. Save the trained model, scaler, and feature-column information.
10. Use Streamlit to accept user input and display predictions.

## 🌐 Streamlit Application

The application provides an interactive form where users can enter health measurements and click **Predict**.

The app displays:
- The model's heart disease prediction.
- The estimated probability for class 1.
- The model's held-out test accuracy for the current training run.

The probability is a model output, not a clinically validated estimate of an individual's actual risk.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/heart-disease-prediction-ml.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Open the project folder

```bash
cd heart-disease-prediction-ml
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser. Make sure `heart.csv` is present in the project folder.

## 📈 Model Evaluation

The application uses a train/test split and reports the accuracy on the held-out test set.

Actual results may vary depending on the dataset, preprocessing, and training configuration. The reported accuracy should not be interpreted as clinical validation.

## 🔮 Future Improvements

- Compare Logistic Regression with additional classification algorithms.
- Add a confusion matrix, precision, recall, and F1-score.
- Improve input validation and error handling.
- Deploy the application online.
- Improve model validation and assess performance on independent data.

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes only. It is not a medical device and must not be used to diagnose heart disease or make treatment decisions. Consult a qualified healthcare professional for medical advice.

## 👨‍💻 Author

**Your Name**

GitHub: https://github.com/YOUR-USERNAME

