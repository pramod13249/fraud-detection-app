import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Fraud Detection", layout="wide")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")   # ✅ FIXED PATH
    return df.sample(20000)

df = load_data()

X = df.drop("Class", axis=1)
y = df["Class"]

# ==============================
# SESSION STATE
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

# ==============================
# TITLE
# ==============================
st.title("💳 Fraud Detection System")
st.write("Real-time fraud detection using Machine Learning")

# ==============================
# SIDEBAR INPUT
# ==============================
with st.sidebar:
    st.title("⚙️ Input Features")

    threshold = st.slider("Fraud Threshold", 0.1, 0.9, 0.3)

    input_data = {}

    # Only important features (clean UI)
    important_features = [f"V{i}" for i in range(1, 11)]

    for feature in important_features:
        input_data[feature] = st.slider(feature, -10.0, 10.0, 0.0)

    # Remaining features default
    for i in range(11, 29):
        input_data[f"V{i}"] = 0.0

    amount = st.number_input("Amount", 0.0)
    input_data["Amount"] = scaler.transform([[amount]])[0][0]

    input_data["Time"] = st.number_input("Time", 0.0)

    predict_btn = st.button("🚀 Predict")
    random_btn = st.button("🎲 Random Test")

input_df = pd.DataFrame([input_data])
input_df = input_df[X.columns]

# Random sample
if random_btn:
    input_df = X.sample(1)

# ==============================
# METRICS
# ==============================
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob > threshold).astype(int)

col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC", f"{roc_auc_score(y, y_prob):.3f}")
col2.metric("Recall (Fraud)", f"{classification_report(y, y_pred, output_dict=True)['1']['recall']:.2f}")
col3.metric("Accuracy", f"{np.mean(y_pred == y):.3f}")

# ==============================
# INPUT SUMMARY
# ==============================
st.subheader("📥 Input Data")
st.dataframe(input_df)

# ==============================
# PREDICTION
# ==============================
if predict_btn:

    prob = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")

    if prob > 0.7:
        result = "HIGH RISK"
        st.error(f"🚨 HIGH RISK ({prob:.2f})")
    elif prob > 0.4:
        result = "MEDIUM RISK"
        st.warning(f"⚠️ MEDIUM RISK ({prob:.2f})")
    else:
        result = "LOW RISK"
        st.success(f"✅ LOW RISK ({prob:.2f})")

    st.progress(float(prob))

    confidence = abs(prob - threshold) * 100
    st.write(f"🎯 Confidence: {confidence:.2f}%")

    st.session_state.history.append({
        "Probability": prob,
        "Result": result
    })

    # Feature importance
    st.subheader("🧠 Feature Importance")

    importances = model.feature_importances_
    features = X.columns

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(imp_df.set_index("Feature"))

# ==============================
# HISTORY
# ==============================
st.subheader("📜 Prediction History")

if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.write("No predictions yet.")

# ==============================
# VISUALIZATIONS (CLEAN UI)
# ==============================
with st.expander("📊 Advanced Visualizations"):

    col1, col2 = st.columns(2)

    # ROC Curve
    with col1:
        st.subheader("📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y, y_prob)
        fig1, ax1 = plt.subplots()
        ax1.plot(fpr, tpr)
        ax1.plot([0, 1], [0, 1], '--')
        st.pyplot(fig1)

    # Heatmap
    with col2:
        st.subheader("🔥 Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

# ==============================
# MODEL INFO
# ==============================
st.subheader("🤖 Model Info")

st.write("""
- Model: Random Forest  
- Handles imbalance using SMOTE  
- Optimized for fraud detection  
- Adjustable threshold for business usage  
""")