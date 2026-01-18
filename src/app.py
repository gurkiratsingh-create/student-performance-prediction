from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    internal_marks = float(request.form["internal_marks"])

    data = np.array([[study_hours, attendance, internal_marks]])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    result = "PASS" if prediction == 1 else "FAIL"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability * 100, 2)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)