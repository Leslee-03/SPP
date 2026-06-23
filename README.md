# 🎓 Student Performance Predictor

A beginner-friendly Machine Learning project that predicts student academic performance using Logistic Regression (Pass/Fail classification) and Linear Regression (final score prediction).

**Course:** Soft Computing

---

## 📁 Project Structure

```
student-performance-predictor/
├── data/
│   └── student_data.csv          # Synthetic dataset (250 rows)
├── notebooks/
│   └── EDA.ipynb                 # Exploratory Data Analysis notebook
├── src/
│   ├── data_preprocessing.py     # Data cleaning, scaling, splitting
│   ├── train_model.py            # Model training and evaluation
│   └── predict.py                # Single-input prediction
├── model/
│   └── model.pkl                 # Saved trained model
├── app/
│   └── app.py                    # Streamlit web application
├── reports/
│   ├── project_report.docx       # Academic project report
│   └── presentation.pptx         # Project presentation slides
├── requirements.txt
└── README.md
```

---

## 🎯 Project Objectives

- Predict whether a student will **Pass or Fail** (Binary Classification)
- Predict a student's **Final Score** (Regression)
- Build a simple **web app** using Streamlit for real-time predictions

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `study_hours` | Daily study hours (1–10) |
| `attendance` | Class attendance percentage (40–100) |
| `previous_marks` | Previous exam score (30–95) |
| `assignments_completed` | Assignments submitted (0–10) |
| `final_score` | Final exam score (target for regression) |
| `pass` | 1 = Pass, 0 = Fail (target for classification) |

---

## 🛠️ Installation

### 1. Clone or download the project
```bash
cd student-performance-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the Model
```bash
python src/train_model.py
```
This will preprocess the data, train both models, print evaluation metrics, and save `model/model.pkl`.

### Step 2 — Make a Prediction (Terminal)
```bash
python src/predict.py
```
Enter student details when prompted.

### Step 3 — Run the Web App
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`

### Step 4 — Run EDA Notebook
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## 📈 Sample Results

```
Classification Accuracy : 95.00%
Confusion Matrix:
  TN=3   FP=2
  FN=1   TP=44

Regression R² Score : 0.9312
MAE                 : 4.81
RMSE                : 6.23
```

### Sample Prediction
```
Input: study_hours=7, attendance=85, previous_marks=72, assignments=8
Output:
  Predicted Score : 79.45 / 100
  Result          : PASS ✓
  Pass Probability: 96.3%
```

---

## 📸 Screenshots

> _[Add Streamlit app screenshot here]_

> _[Add EDA plots screenshot here]_

---

## 🔧 Tech Stack

- **Python 3.x**
- **scikit-learn** — ML models
- **pandas / numpy** — data processing
- **matplotlib / seaborn** — visualization
- **Streamlit** — web application

---

## 📚 Course

Soft Computing
