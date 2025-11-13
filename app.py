from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import os  # added for relative paths

app = Flask(__name__)

# Get project root dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "Result")  # folder where your .pkl files are

# Load trained model and scaler using relative paths
try:
    model = pickle.load(open(os.path.join(RESULT_DIR, "churn_prediction_project.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(RESULT_DIR, "standard_scaler.pkl"), "rb"))
    feature_columns = pickle.load(open(os.path.join(RESULT_DIR, "feature_columns.pkl"), "rb"))
except FileNotFoundError:
    print("Warning: Model files not found in Result folder.")
    model, scaler, feature_columns = None, None, []

# Cache prediction to show same result until refresh
cached_result = None


@app.route('/')
def home():
    global cached_result
    cached_result = None  # reset on refresh
    return render_template('index.html', form_data={})


@app.route('/predict', methods=['POST'])
def predict():
    global cached_result
    form_data = {}
    try:
        input_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'tenure', 'MonthlyCharges', 'TotalCharges'
        ]

        data = [request.form.get(col) for col in input_cols]
        form_data = {col: request.form.get(col, '') for col in input_cols}

        if any(v is None or (isinstance(v, str) and v.strip() == "") for v in data):
            return render_template('index.html', prediction="⚠ Error: Please enter all the values", form_data=form_data)

        if model is None or scaler is None or not feature_columns:
            return render_template('index.html', prediction="⚠ Error: Model files could not be loaded.",
                                   form_data=form_data)

        df = pd.DataFrame([data], columns=input_cols)
        df['TotalCharges'] = df['TotalCharges'].replace('', 0)

        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
        df['tenure'] = df['tenure'].astype(float)
        df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
        df['TotalCharges'] = df['TotalCharges'].astype(float)

        df['tenure_log_trim'] = np.log1p(df['tenure'])
        df['TotalCharges_log_trim'] = np.log1p(df['TotalCharges'])

        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]

        df_encoded = pd.get_dummies(df, columns=categorical_cols)

        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_columns]

        scaled_features = scaler.transform(df_encoded)
        prediction = model.predict(scaled_features)[0]
        result = "✅ The customer is NOT likely to churn." if prediction == 0 else "⚠ The customer is LIKELY to churn."
        cached_result = result

        return render_template('index.html', prediction=result, form_data=form_data)

    except Exception as e:
        return render_template('index.html', prediction=f"⚠ An unexpected error occurred: {str(e)}",
                               form_data=form_data)


if __name__ == '__main__':
    app.run(debug=True)
