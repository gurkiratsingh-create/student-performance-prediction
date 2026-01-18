import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
internal_marks = float(input("Enter internal marks: "))

data = np.array([[study_hours, attendance, internal_marks]])

prediction = model.predict(data)
probability = model.predict_proba(data)

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")

print("Probability of PASS:", probability[0][1])
print("Probability of FAIL:", probability[0][0])