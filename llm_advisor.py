from groq import Groq
import joblib
import numpy as np
import os
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
# Works both locally and on Streamlit Cloud
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────
#  LOAD MODEL & SCALER
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ─────────────────────────────────────────
#  MAIN FUNCTION
# ─────────────────────────────────────────
def predict_and_advise(pregnancies, glucose, blood_pressure,
                        skin_thickness, insulin, bmi,
                        diabetes_pedigree, age):
    """
    Takes raw patient values → ML prediction → Groq LLM health advice.
    Returns: prediction (0 or 1), confidence (%), risk_label, advice
    """

    # Step 1 — Prepare input as DataFrame (fixes feature names warning)
    columns = ['Pregnancies','Glucose','BloodPressure',
               'SkinThickness','Insulin','BMI',
               'DiabetesPedigreeFunction','Age']

    input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure,
                               skin_thickness, insulin, bmi,
                               diabetes_pedigree, age]], columns=columns)

    # Step 2 — Scale & Predict
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]
    confidence   = round(max(probability) * 100, 2)
    risk_label   = "Diabetic" if prediction == 1 else "Non-Diabetic"

    # Step 3 — Build Prompt
    prompt = f"""
You are a friendly and professional medical AI assistant.

A patient has been analyzed by a Machine Learning model trained on the
Pima Indians Diabetes Dataset. Here are the patient details:

- Age                       : {age} years
- Pregnancies               : {pregnancies}
- Glucose Level             : {glucose} mg/dL
- Blood Pressure            : {blood_pressure} mmHg
- Skin Thickness            : {skin_thickness} mm
- Insulin Level             : {insulin} IU/mL
- BMI                       : {bmi}
- Diabetes Pedigree Function: {diabetes_pedigree}

ML Model Prediction : {risk_label}
Model Confidence    : {confidence}%

Please do the following:
1. Explain what this prediction means in simple, friendly words.
2. Point out which values look concerning and why.
3. Give 3 to 4 specific and actionable health recommendations.
4. Keep tone warm, supportive and easy to understand.
5. Keep response under 200 words.

Speak directly to the patient. Do not repeatedly say "consult a doctor".
"""

    # Step 4 — Call Groq API
    chat_response = client.chat.completions.create(
        # model    = "llama3-8b-8192",
        model = "llama-3.1-8b-instant",  # Free & fast Llama 3 model
        messages = [
            {
                "role"   : "system",
                "content": "You are a helpful and friendly medical AI assistant."
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        max_tokens  = 300,
        temperature = 0.7
    )

    advice = chat_response.choices[0].message.content.strip()

    return prediction, confidence, risk_label, advice


# ─────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 Testing Groq LLM Integration...\n")

    prediction, confidence, risk_label, advice = predict_and_advise(
        pregnancies      = 2,
        glucose          = 138,
        blood_pressure   = 62,
        skin_thickness   = 35,
        insulin          = 0,
        bmi              = 33.6,
        diabetes_pedigree= 0.127,
        age              = 47
    )

    print(f"🔮 Prediction : {risk_label}")
    print(f"📊 Confidence : {confidence}%")
    print(f"\n🤖 AI Health Advice:\n")
    print(advice)
    print("\n✅ Groq Integration working perfectly!")