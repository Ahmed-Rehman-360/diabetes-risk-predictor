import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from llm_advisor import predict_and_advise

# ─────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "Diabetes Risk Predictor",
    page_icon  = "🩺",
    layout     = "wide"
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .stApp { font-family: 'Segoe UI', sans-serif; }

    .title-box {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    .title-box h1 { font-size: 2.5em; margin: 0; }
    .title-box p  { font-size: 1.1em; margin-top: 8px; opacity: 0.9; }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        margin: 20px 0;
    }
    .diabetic {
        background-color: #fdecea;
        border: 2px solid #e53935;
        color: #b71c1c;
    }
    .non-diabetic {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
        color: #1b5e20;
    }
    .advice-box {
        background-color: #ffffff;
        border-left: 5px solid #1a73e8;
        padding: 20px 25px;
        border-radius: 10px;
        font-size: 1em;
        line-height: 1.7;
        color: #333;
        margin-top: 10px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
    <div class="title-box">
        <h1>🩺 Smart Diabetes Risk Predictor</h1>
        <p>Powered by Machine Learning (SVM) + AI Health Advisor (Groq LLaMA 3)</p>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SIDEBAR — PATIENT INPUT
# ─────────────────────────────────────────
st.sidebar.title("👤 Patient Information")
st.sidebar.markdown("Fill in the patient details below:")

pregnancies       = st.sidebar.slider("🤰 Pregnancies",           0,  17,  2)
glucose           = st.sidebar.slider("🩸 Glucose (mg/dL)",       0, 200, 120)
blood_pressure    = st.sidebar.slider("💓 Blood Pressure (mmHg)", 0, 122,  70)
skin_thickness    = st.sidebar.slider("📏 Skin Thickness (mm)",   0,  99,  20)
insulin           = st.sidebar.slider("💉 Insulin (IU/mL)",       0, 846,  80)
bmi               = st.sidebar.slider("⚖️ BMI",                  0.0, 67.1, 25.0, step=0.1)
diabetes_pedigree = st.sidebar.slider("🧬 Diabetes Pedigree",     0.0,  2.4,  0.5, step=0.01)
age               = st.sidebar.slider("🎂 Age",                   1,   81,  33)

predict_btn = st.sidebar.button("🔮 Predict Now", use_container_width=True)

# ─────────────────────────────────────────
#  MAIN AREA — TWO COLUMNS
# ─────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Patient Summary")

    summary_data = {
        "Feature"       : ["Pregnancies","Glucose","Blood Pressure",
                           "Skin Thickness","Insulin","BMI",
                           "Diabetes Pedigree","Age"],
        "Value"         : [pregnancies, glucose, blood_pressure,
                           skin_thickness, insulin, bmi,
                           diabetes_pedigree, age],
        "Normal Range"  : ["0–10","70–140 mg/dL","60–80 mmHg",
                           "10–50 mm","16–166 IU/mL","18.5–24.9",
                           "0.0–1.0","–"]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # ── Model Accuracy Comparison Chart ──
    st.subheader("📊 ML Model Comparison")
    model_names = ["KNN", "Naive Bayes", "SVM", "Decision Tree"]
    accuracies  = [70.78, 69.48, 72.73, 66.88]
    colors      = ["#3498db","#2ecc71","#e74c3c","#f39c12"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(model_names, accuracies, color=colors,
                  edgecolor="black", width=0.5)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{acc}%", ha="center",
                va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylim(60, 85)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.patch.set_facecolor("#f0f4f8")
    ax.set_facecolor("#f0f4f8")
    st.pyplot(fig)

with col2:
    st.subheader("🔮 Prediction Result")

    if predict_btn:
        with st.spinner("🤖 Analyzing patient data..."):
            try:
                prediction, confidence, risk_label, advice = predict_and_advise(
                    pregnancies, glucose, blood_pressure,
                    skin_thickness, insulin, bmi,
                    diabetes_pedigree, age
                )

                # ── Result Card ──
                if prediction == 1:
                    st.markdown(f"""
                        <div class="result-box diabetic">
                            🔴 Result: DIABETIC<br>
                            <span style="font-size:0.8em">
                                Model Confidence: {confidence}%
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-box non-diabetic">
                            🟢 Result: NON-DIABETIC<br>
                            <span style="font-size:0.8em">
                                Model Confidence: {confidence}%
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

                # ── Confidence Metrics ──
                m1, m2 = st.columns(2)
                m1.metric("🎯 Model Used", "SVM")
                m2.metric("📊 Confidence", f"{confidence}%")

                # ── AI Advice ──
                st.subheader("🤖 AI Health Advisor")
                st.markdown(f"""
                    <div class="advice-box">{advice}</div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    else:
        st.info("👈 Fill in patient details in the sidebar and click **Predict Now**")

        st.markdown("""
            ### How it works:
            1. 📝 Enter patient details in the left sidebar
            2. 🔮 Click **Predict Now**
            3. 🤖 ML model (SVM) analyzes the data
            4. 💬 AI advisor generates personalized health advice
        """)

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:grey; font-size:0.85em'>
        🎓 Academic Project — Machine Learning Fundamentals |
        Dataset: Pima Indians Diabetes | Model: SVM | LLM: Groq LLaMA 3
    </div>
""", unsafe_allow_html=True)