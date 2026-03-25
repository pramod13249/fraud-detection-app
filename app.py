import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)
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
# TITLE
# ==============================
st.title("💳 Fraud Detection System")
st.write("Real-time fraud detection using Machine Learning")

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
# INPUT
# ==============================
st.sidebar.header("⚙️ Input Features")

input_data = {}

for i in range(1, 29):
    input_data[f"V{i}"] = st.sidebar.slider(f"V{i}", -10.0, 10.0, 0.0)

input_data["Amount"] = st.sidebar.number_input("Amount", 0.0)
input_data["Time"] = st.sidebar.number_input("Time", 0.0)

input_df = pd.DataFrame([input_data])

# Scale amount
input_df["Amount"] = scaler.transform(input_df[["Amount"]])

# Match columns
input_df = input_df[X.columns]

st.subheader("📥 Input Data")
st.dataframe(input_df)

# ==============================
# PREDICTION
# ==============================
if st.sidebar.button("🚀 Predict"):

    prob = model.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")

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
st.subheader("📈 ROC Curve")

fpr, tpr, _ = roc_curve(y, y_prob)

fig1, ax1 = plt.subplots()
ax1.plot(fpr, tpr)
ax1.plot([0, 1], [0, 1], '--')
st.pyplot(fig1)

# ==============================
# CONFUSION MATRIX
# ==============================
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)

# ==============================
# PRECISION-RECALL CURVE
# ==============================
st.subheader("📉 Precision-Recall Curve")

precision, recall, _ = precision_recall_curve(y, y_prob)

fig3, ax3 = plt.subplots()
ax3.plot(recall, precision)
ax3.set_xlabel("Recall")
ax3.set_ylabel("Precision")

st.pyplot(fig3)

# ==============================
# FEATURE IMPORTANCE
# ==============================
st.subheader("📦 Feature Importance")

importances = model.feature_importances_

feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

fig4, ax4 = plt.subplots()
ax4.barh(feat_imp["Feature"], feat_imp["Importance"])
ax4.invert_yaxis()

st.pyplot(fig4)

# ==============================
# HEATMAP
# ==============================
st.subheader("🔥 Feature Correlation Heatmap")

fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax5)

st.pyplot(fig5)