import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle

# Load dataset
df = pd.read_csv("../data/student_data.csv")

# Features and target
X = df.drop("final_result", axis=1)
y = df["final_result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Training model
print("Training Logistic Regression Model...")
lrmodel = LogisticRegression()
lrmodel.fit(X_train, y_train)


#Cross-validation
cv_scores = cross_val_score(lrmodel,X,y,cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())


# Evaluating model
predictions = lrmodel.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
print("Total Samples:",X.shape[0])

# Saving model
with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(lrmodel, f)
