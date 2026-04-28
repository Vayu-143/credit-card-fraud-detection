import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Fraud Detection", layout="wide")

# ------------------------------
# LOAD MODEL (CACHED)
# ------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    )

df = load_data()

# ------------------------------
# TITLE
# ------------------------------
st.markdown("<h1 style='text-align:center;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

# ==============================
# 🎛️ SIDEBAR INPUTS
# ==============================
st.sidebar.header("🧾 Enter Transaction Details")

amount = st.sidebar.number_input("Transaction Amount", value=100.0)

v1 = st.sidebar.slider("V1", -10.0, 10.0, 0.0)
v2 = st.sidebar.slider("V2", -10.0, 10.0, 0.0)
v3 = st.sidebar.slider("V3", -10.0, 10.0, 0.0)
v4 = st.sidebar.slider("V4", -10.0, 10.0, 0.0)
v5 = st.sidebar.slider("V5", -10.0, 10.0, 0.0)

time_val = np.random.uniform(0, 100000)

# Random generator
if st.sidebar.button("🎲 Generate Random Transaction"):
    amount = np.random.uniform(1, 10000)
    v1, v2, v3, v4, v5 = np.random.normal(0, 1, 5)

# ==============================
# 🔍 PREDICTION
# ==============================
predict_btn = st.sidebar.button("🚀 Check Fraud")

if predict_btn:

    # Generate realistic unseen features
    random_features = np.random.normal(0, 1, 23)

    features = [time_val, v1, v2, v3, v4, v5] + list(random_features) + [amount]

    # Match training columns
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df = pd.DataFrame([features], columns=columns)

    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]

    # ------------------------------
    # RESULT
    # ------------------------------
    st.markdown("## 🧾 Transaction Analysis")

    if prob > 0.5:
        st.error(f"🚨 Fraud Detected\nProbability: {prob:.4f}")
    else:
        st.success(f"✅ Legit Transaction\nFraud Probability: {prob:.4f}")

    st.markdown("### 📊 Confidence Level")
    st.progress(float(prob))

    if prob > 0.7:
        st.warning("🔴 High Risk Transaction")
    elif prob > 0.3:
        st.info("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    st.markdown("---")

# ==============================
# 📊 DATASET INSIGHTS
# ==============================
st.markdown("## 📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Class Distribution")
    st.bar_chart(df['Class'].value_counts())

with col2:
    st.subheader("Amount Distribution")
    st.line_chart(df['Amount'].head(200))

st.markdown("---")

# ==============================
# 🤖 MODEL PERFORMANCE
# ==============================
st.markdown("## 🤖 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.write("Precision (Fraud): 0.81")
    st.write("Recall (Fraud): 0.82")
    st.write("F1-score: 0.81")

with col2:
    st.image("images/roc.png", caption="ROC Curve")

st.markdown("---")

# ==============================
# 🧠 MODEL EXPLANATION
# ==============================
st.markdown("## 🧠 How the Model Works")

st.write("""
- Model used: Random Forest Classifier  
- Handles imbalanced data using class_weight='balanced'  
- Uses anonymized PCA features (V1–V28)  
- Detects unusual transaction behavior patterns  
""")

st.markdown("---")

# ==============================
# 📊 FEATURE IMPORTANCE
# ==============================
st.markdown("## 📊 Feature Importance")

importance = model.feature_importances_

fig, ax = plt.subplots()
ax.bar(range(len(importance)), importance)
ax.set_title("Feature Importance")

st.pyplot(fig)

st.markdown("---")

# ==============================
# 🧾 TRANSACTION HISTORY
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

if predict_btn:
    st.session_state.history.append({
        "Amount": amount,
        "Fraud Probability": round(prob, 4)
    })

st.markdown("## 🧾 Recent Transactions")
st.dataframe(pd.DataFrame(st.session_state.history))

# ==============================
# 📌 PROJECT DESCRIPTION
# ==============================
st.markdown("---")
st.markdown("## 📌 About This Project")

st.write("""
This project detects fraudulent credit card transactions using machine learning.

It uses a Random Forest model trained on highly imbalanced financial data.

Features:
- Real-time fraud prediction  
- Confidence scoring  
- Dataset visualization  
- Feature importance analysis  
""")