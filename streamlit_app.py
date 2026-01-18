import streamlit as st
import pickle
import numpy as np

st.title("Student Performance Prediction")

# Load model
with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)

study_hours = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
internal_marks = st.number_input("Internal Marks", min_value=0.0)

if st.button("Predict"):
    data = np.array([[study_hours, attendance, internal_marks]])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.success(f"PASS (Probability: {probability*100:.2f}%)")
    else:
        st.error(f"FAIL (Probability: {probability*100:.2f}%)")
