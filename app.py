import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Fraud Detection", layout="wide")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df.sample(20000)

df = load_data()

# ==============================
# TRAIN MODEL (NO .PKL NEEDED)
# ==============================
@st.cache_resource
def train_model(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train_res, y_train_res)

    return model, scaler, X_test, y_test

model, scaler, X_test, y_test = train_model(df)

X = df.drop("Class", axis=1)
y = df["Class"]

# ==============================
# UI
# ==============================
st.title("💳 Fraud Detection System")

threshold = st.sidebar.slider("🎯 Threshold", 0.1, 0.9, 0.3)

# ==============================
# METRICS
# ==============================
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob > threshold).astype(int)

col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC", f"{roc_auc_score(y, y_prob):.3f}")
col2.metric("Recall", f"{classification_report(y, y_pred, output_dict=True)['1']['recall']:.2f}")
col3.metric("Accuracy", f"{np.mean(y_pred == y):.3f}")

# ==============================
# INPUT
# ==============================
st.sidebar.header("Input")

input_data = {}

for i in range(1, 29):
    input_data[f"V{i}"] = st.sidebar.slider(f"V{i}", -10.0, 10.0, 0.0)

amount = st.sidebar.number_input("Amount", 0.0)
input_data["Amount"] = scaler.transform([[amount]])[0][0]
input_data["Time"] = st.sidebar.number_input("Time", 0.0)

input_df = pd.DataFrame([input_data])
input_df = input_df[X.columns]

# ==============================
# PREDICTION
# ==============================
if st.sidebar.button("Predict"):

    prob = model.predict_proba(input_df)[0][1]

    if prob > 0.7:
        st.error(f"🚨 HIGH RISK ({prob:.2f})")
    elif prob > 0.4:
        st.warning(f"⚠️ MEDIUM RISK ({prob:.2f})")
    else:
        st.success(f"✅ LOW RISK ({prob:.2f})")

    st.progress(float(prob))

# ==============================
# ROC CURVE
# ==============================
st.subheader("ROC Curve")

fpr, tpr, _ = roc_curve(y, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], '--')

st.pyplot(fig)