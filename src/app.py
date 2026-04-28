import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_data
from preprocessing import create_input_dataframe
from model import load_model
from visualize import plot_feature_importance
from evaluate import get_metrics

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Fraud Detection", layout="wide")

# =========================
# LOAD DATA + MODEL
# =========================
df = load_data()
model, scaler = load_model()

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.title("📋 Enter Transaction Details")

amount = st.sidebar.number_input("Transaction Amount", value=100.0)

v1 = st.sidebar.slider("V1", -10.0, 10.0, 0.0)
v2 = st.sidebar.slider("V2", -10.0, 10.0, 0.0)
v3 = st.sidebar.slider("V3", -10.0, 10.0, 0.0)
v4 = st.sidebar.slider("V4", -10.0, 10.0, 0.0)
v5 = st.sidebar.slider("V5", -10.0, 10.0, 0.0)

predict_btn = st.sidebar.button("🚀 Check Fraud")

# =========================
# TITLE
# =========================
st.markdown(
    "<h1 style='text-align:center;'>💳 Credit Card Fraud Detection</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# PREDICTION
# =========================
if predict_btn:

    input_df = create_input_dataframe(amount, v1, v2, v3, v4, v5)

    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]

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

# =========================
# DATASET INSIGHTS
# =========================
st.markdown("## 📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Class Distribution")
    st.bar_chart(df['Class'].value_counts())

with col2:
    st.subheader("Amount Distribution")
    st.line_chart(df['Amount'].head(200))

st.markdown("---")

# =========================
# MODEL PERFORMANCE
# =========================
st.markdown("## 🤖 Model Performance")

metrics = get_metrics()

st.write(f"Precision (Fraud): {metrics['precision']}")
st.write(f"Recall (Fraud): {metrics['recall']}")
st.write(f"F1-score: {metrics['f1']}")

# =========================
# ROC CURVE
# =========================
st.image("images/roc.png", use_column_width=True)

st.markdown("---")

# =========================
# FEATURE IMPORTANCE
# =========================
st.markdown("## 🔍 Feature Importance")

fig = plot_feature_importance(model)
st.pyplot(fig, use_container_width=True)

st.markdown("---")

# =========================
# MODEL EXPLANATION
# =========================
st.markdown("## 🧠 How the Model Works")

st.write("""
- Model: Random Forest Classifier  
- Handles imbalanced data using class_weight='balanced'  
- Uses PCA-transformed features (V1–V28)  
- Detects unusual transaction patterns  
""")

# =========================
# PROJECT INFO
# =========================
st.markdown("---")
st.markdown("## 📌 About This Project")

st.write("""
This project detects fraudulent credit card transactions using machine learning.

Features:
- Real-time fraud prediction  
- Confidence scoring  
- Dataset visualization  
- Feature importance analysis  
""")