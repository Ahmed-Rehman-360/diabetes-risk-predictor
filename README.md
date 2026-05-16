# 🩺 Smart Diabetes Risk Predictor

An intelligent web application that predicts diabetes risk using 
Machine Learning and provides personalized health advice using 
a Large Language Model (LLM).

---

## 📌 Problem Statement

Diabetes is a growing global health crisis. Early detection is
critical but often delayed due to lack of accessible screening tools.
This project addresses that gap by building an ML-powered web app
that predicts diabetes risk from basic health parameters and explains
the result in simple, friendly language using AI.

---

## 📊 Dataset

- **Name:** Pima Indians Diabetes Dataset
- **Source:** UCI Machine Learning Repository (via Kaggle)
- **Size:** 768 patients, 9 features
- **Target:** Outcome (1 = Diabetic, 0 = Non-Diabetic)

### Features Used:
| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mmHg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (IU/mL) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes hereditary score |
| Age | Age in years |

---

## 🤖 Machine Learning Models

Four classifiers were trained and compared:

| Model | Accuracy |
|-------|----------|
| KNN | 70.78% |
| Naive Bayes | 69.48% |
| **SVM** ⭐ | **72.73%** |
| Decision Tree | 66.88% |

**Best Model: SVM (Support Vector Machine) with 72.73% accuracy**

### Preprocessing Steps:
- Replaced invalid zero values with column medians
- Applied StandardScaler normalization
- 80/20 train-test split with stratification

---

## 🧠 LLM Integration

- **Provider:** Groq API
- **Model:** LLaMA 3.1 8B Instant
- **Purpose:** Generates personalized, friendly health advice
  based on patient data and ML prediction
- **Why LLM?** Raw ML predictions are hard to understand.
  The LLM translates results into actionable human-friendly advice.

---

## 🖥️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| LLM API | Groq (LLaMA 3.1) |
| Frontend | Streamlit |
| Model Saving | Joblib |

---

## 🚀 How to Run

### 1. Clone or download this project

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add your Groq API key
Open llm_advisor.py and replace:
GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"

### 4. Run the app
streamlit run app.py

### 5. Open browser at:
http://localhost:8501

---

## 📁 Project Structure

diabetes-risk-predictor/
│
├── data/
│   └── diabetes.csv
├── models/
│   └── best_model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── model_training.ipynb
├── app.py
├── llm_advisor.py
├── requirements.txt
└── README.md

---

## 🎓 Academic Information

- **Course:** Machine Learning Fundamentals
- **Project Type:** End-to-End ML + LLM Integration
- **Algorithms Used:** KNN, Naive Bayes, SVM, Decision Tree
- **LLM Integration:** Groq LLaMA 3.1 for health advice generation