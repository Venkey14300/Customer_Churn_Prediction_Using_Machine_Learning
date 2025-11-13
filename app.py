from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model and scaler
# NOTE: These paths are system-specific and will cause errors in a generic environment.
# Assuming these files are available in the runtime environment for the Flask app to function.
try:
    model = pickle.load(open("C:\\Users\\LENOVO\\Documents\\PROJECT_ALL_FILES\\Result\\churn_prediction_project.pkl", "rb"))
    scaler = pickle.load(open("C:\\Users\\LENOVO\\Documents\\PROJECT_ALL_FILES\\Result\\standard_scaler.pkl", "rb"))
    feature_columns = pickle.load(open("C:\\Users\\LENOVO\\Documents\\PROJECT_ALL_FILES\\Result\\feature_columns.pkl", "rb"))
except FileNotFoundError:
    print("Warning: Model files not found at specified C:\\Users\\LENOVO\\Documents\\PROJECT_ALL_FILES\\Result\\ paths.")
    model, scaler, feature_columns = None, None, []

# Cache prediction to show same result until refresh
cached_result = None


@app.route('/')
def home():
    global cached_result
    cached_result = None  # reset on refresh
    # Pass an empty form_data dictionary for the initial load
    return render_template('index.html', form_data={})


@app.route('/predict', methods=['POST'])
def predict():
    global cached_result
    form_data = {}  # store values to show back in form
    try:
        # Check if we should serve cached result (only if form hasn't changed, though this logic is tricky
        # so we'll rely on the client-side 'Refresh' hitting home() to clear the cache)
        # if cached_result is not None:
        #     return render_template('index.html', prediction=cached_result, form_data=request.form)

        input_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'tenure', 'MonthlyCharges', 'TotalCharges'
        ]

        # Use request.form.get to safely retrieve data
        data = [request.form.get(col) for col in input_cols]
        # Store submitted values in form_data dictionary
        form_data = {col: request.form.get(col, '') for col in input_cols}

        if any(v is None or (isinstance(v, str) and v.strip() == "") for v in data):
            # Do not clear cached_result here, let the form_data persist
            return render_template('index.html', prediction="⚠ Error: Please enter all the values", form_data=form_data)

        # Handle missing model files gracefully
        if model is None or scaler is None or not feature_columns:
            return render_template('index.html', prediction="⚠ Error: Model files could not be loaded.",
                                   form_data=form_data)

        # Data processing logic
        df = pd.DataFrame([data], columns=input_cols)

        # TotalCharges needs special handling for empty string as it comes from an input field
        df['TotalCharges'] = df['TotalCharges'].replace('', 0)

        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
        df['tenure'] = df['tenure'].astype(float)
        df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
        df['TotalCharges'] = df['TotalCharges'].astype(float)

        # Feature Engineering (log transformation)
        df['tenure_log_trim'] = np.log1p(df['tenure'])
        df['TotalCharges_log_trim'] = np.log1p(df['TotalCharges'])

        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]

        # One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=categorical_cols)

        # Align columns with trained features
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_columns]

        # Scaling and Prediction
        scaled_features = scaler.transform(df_encoded)
        prediction = model.predict(scaled_features)[0]
        result = "✅ The customer is NOT likely to churn." if prediction == 0 else "⚠ The customer is LIKELY to churn."
        cached_result = result

        return render_template('index.html', prediction=result, form_data=form_data)

    except Exception as e:
        # Keep form data on error
        return render_template('index.html', prediction=f"⚠ An unexpected error occurred: {str(e)}",
                               form_data=form_data)


if __name__ == '__main__':
    app.run(debug=True)