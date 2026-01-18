# Student Performance Prediction – Machine Learning Web App

An end-to-end Machine Learning project that predicts whether a student will **PASS or FAIL** based on academic factors such as study hours, attendance, and internal marks. The trained ML model is deployed as a **web application using Flask**.

---

## 📌 Problem Statement

Educational institutions often want early indicators of student performance. This project uses machine learning to predict student outcomes and demonstrates how an ML model can be trained, evaluated, saved, and deployed as a web application.

---

## 📊 Dataset

The dataset contains student academic information with the following features:

- `study_hours` – Number of hours a student studies per day  
- `attendance` – Attendance percentage  
- `internal_marks` – Internal assessment marks  
- `final_result` – Target variable  
  - `1` → PASS  
  - `0` → FAIL  

The dataset is stored in `data/student_data.csv`.

---

## 🧠 Machine Learning Approach

- **Problem Type:** Binary Classification  
- **Primary Model:** Logistic Regression  
- **Secondary Model (Compared):** Decision Tree Classifier  
- **Libraries Used:** Scikit-learn, Pandas, NumPy  

### ML Workflow
1. Load and preprocess the dataset  
2. Separate features (`X`) and target (`y`)  
3. Perform a stratified train–test split  
4. Train machine learning models  
5. Evaluate models using multiple metrics  
6. Compare models using cross-validation  
7. Select the final model  
8. Save the trained model for inference  

---

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  
- 5-Fold Cross-Validation  

### Model Comparison Summary

- **Logistic Regression**  
  - Mean CV Accuracy ≈ 96%  
  - Stable performance across folds  
  - Lower risk of overfitting on small datasets  

- **Decision Tree**  
  - Mean CV Accuracy ≈ 92%  
  - Higher variance across folds  
  - More prone to overfitting  

📌 **Final Model Selected:** Logistic Regression  
The selection was based on cross-validation stability and generalization performance.

---

## 🔮 Inference (Prediction)

The trained model is saved using `pickle` and reused for predictions without retraining.

Users provide:
- Study hours  
- Attendance percentage  
- Internal marks  

The model outputs:
- PASS / FAIL prediction  
- Probability of passing  

---

## 🌐 Web Deployment (Flask)

The project includes a Flask-based web application that allows users to interact with the trained ML model through a browser.

### Features
- HTML form for user input  
- Real-time predictions  
- Probability score displayed  

The application runs locally at:
```
http://127.0.0.1:5000/
```

---

## 📁 Project Structure

```
student-performance-prediction/
├── data/
│   └── student_data.csv
├── src/
│   ├── train_model.py
│   ├── predict.py
│   ├── app.py
│   ├── model.pkl
│   └── templates/
│       └── index.html
├── venv/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/student-performance-prediction.git
cd student-performance-prediction
```

### 2️⃣ Activate virtual environment
```bash
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the model
```bash
cd src
python train_model.py
```

### 5️⃣ Run the Flask app
```bash
python app.py
```

Open your browser and visit:
```
http://127.0.0.1:5000/
```

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Flask  
- HTML (Jinja Templates)  

---

## 🎯 Learning Outcomes

- Built an end-to-end Machine Learning pipeline  
- Implemented proper model evaluation techniques  
- Compared multiple ML models using cross-validation  
- Saved and reused trained ML models  
- Deployed a Machine Learning model using Flask  
- Connected backend ML logic with a frontend user interface  

---

## 🚀 Future Improvements

- Increase dataset size for better generalization  
- Add feature scaling and hyperparameter tuning  
- Deploy the application online (Render / Railway)  
- Improve user interface with CSS  
- Add authentication and input validation  

---

## 👤 Author

**Gurkirat Singh**  
B.E CSE (AI & ML) Student  
Aspiring AI/ML Engineer

