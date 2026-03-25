import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, classification_report, roc_auc_score
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
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    return df.sample(20000)

df = load_data()

# ==============================
# TRAIN MODEL
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

    return model, scaler, X_test, y_test, X, y

model, scaler, X_test, y_test, X, y = train_model(df)

# ==============================
# SHAP EXPLAINER (NO CACHE BUG)
# ==============================
def get_explainer(model):
    return shap.TreeExplainer(model)

explainer = get_explainer(model)

# ==============================
# SESSION STATE
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

# ==============================
# TITLE
# ==============================
st.title("💳 Fraud Detection System")

# ==============================
# THRESHOLD
# ==============================
threshold = st.sidebar.slider("🎯 Fraud Threshold", 0.1, 0.9, 0.3)

# ==============================
# METRICS
# ==============================
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob > threshold).astype(int)

col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC", f"{roc_auc_score(y, y_prob):.3f}")
col2.metric("Fraud Recall", f"{classification_report(y, y_pred, output_dict=True)['1']['recall']:.2f}")
col3.metric("Accuracy", f"{np.mean(y_pred == y):.3f}")

# ==============================
# SIDEBAR INPUT
# ==============================
st.sidebar.header("⚙️ Input Features")

input_data = {}

for i in range(1, 29):
    input_data[f"V{i}"] = st.sidebar.slider(f"V{i}", -10.0, 10.0, 0.0)

input_data["Amount"] = st.sidebar.number_input("Amount", 0.0)
input_data["Time"] = st.sidebar.number_input("Time", 0.0)

# ==============================
# CREATE INPUT DF
# ==============================
input_df = pd.DataFrame([input_data])

# ✅ CORRECT SCALING
input_df["Amount"] = scaler.transform(input_df[["Amount"]])

# Match training columns
input_df = input_df[X.columns]

# ==============================
# DISPLAY INPUT
# ==============================
st.subheader("📥 Input Data")
st.dataframe(input_df)

# ==============================
# PREDICTION
# ==============================
if st.sidebar.button("🚀 Predict"):

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

    st.session_state.history.append({
        "Probability": prob,
        "Result": result
    })

    # ==============================
    # SHAP EXPLANATION (FIXED)
    # ==============================
    st.subheader("🧠 Why this prediction?")

    shap_values = explainer.shap_values(input_df)

    # ✅ Handle both SHAP formats
    if isinstance(shap_values, list):
        values = shap_values[1][0]
        base = explainer.expected_value[1]
    else:
        values = shap_values[0]
        base = explainer.expected_value

    fig, ax = plt.subplots()

    shap.plots.waterfall(
        shap.Explanation(
            values=values,
            base_values=base,
            data=input_df.iloc[0]
        ),
        show=False
    )

    st.pyplot(fig)

# ==============================
# HISTORY
# ==============================
st.subheader("📜 Prediction History")

if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.write("No predictions yet.")

# ==============================
# ROC CURVE
# ==============================
st.subheader("📈 ROC Curve")

fpr, tpr, _ = roc_curve(y, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], '--')
st.pyplot(fig)

# ==============================
# HEATMAP
# ==============================
st.subheader("🔥 Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax2)
st.pyplot(fig2)