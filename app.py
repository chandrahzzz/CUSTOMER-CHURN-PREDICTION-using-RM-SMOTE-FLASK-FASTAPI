from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Load model, encoders, scaler
with open(os.path.join(BASE_DIR, 'best_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'encoder.pkl'), 'rb') as f:
    encoders = pickle.load(f)

with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)


def make_prediction(input_data):
    df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    # Scale numeric columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return "Churn" if prediction == 1 else "No Churn", round(float(probability), 2)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            # Validate empty fields
            if "" in request.form.values():
                raise ValueError("Please fill all fields")

            input_data = {
                'gender': request.form['gender'],
                'SeniorCitizen': int(request.form['SeniorCitizen']),
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'tenure': int(request.form['tenure']),
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod'],
                'MonthlyCharges': float(request.form['MonthlyCharges']),
                'TotalCharges': float(request.form['TotalCharges']),
            }

            prediction, probability = make_prediction(input_data)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
