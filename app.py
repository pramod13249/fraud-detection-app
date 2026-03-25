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
# LOAD DATA (FROM URL ✅)
# ==============================
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    return df.sample(20000)

df = load_data()

# ==============================
# TRAIN MODEL (NO PKL ✅)
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
# THRESHOLD
# ==============================
threshold = st.sidebar.slider("🎯 Fraud Threshold", 0.1, 0.9, 0.3)

st.info("""
Lower threshold → higher fraud detection (recall)  
Higher threshold → fewer false positives  
""")

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

amount = st.sidebar.number_input("Amount", 0.0)
input_data["Amount"] = scaler.transform([[amount]])[0][0]

input_data["Time"] = st.sidebar.number_input("Time", 0.0)

input_df = pd.DataFrame([input_data])
input_df = input_df[X.columns]

# ==============================
# INPUT SUMMARY
# ==============================
st.subheader("📥 Input Summary")
st.dataframe(input_df)

# ==============================
# RANDOM TEST
# ==============================
if st.sidebar.button("🎲 Random Test"):
    input_df = X.sample(1)
    st.sidebar.success("Random sample loaded!")

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

    confidence = abs(prob - threshold) * 100
    st.write(f"🎯 Confidence: {confidence:.2f}%")

    st.session_state.history.append({
        "Probability": prob,
        "Result": result
    })

    # Feature importance
    st.subheader("🧠 Top Features")

    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
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
# DISTRIBUTION
# ==============================
st.subheader("📊 Fraud vs Normal Distribution")

st.bar_chart(df["Class"].value_counts())

# ==============================
# MODEL PERFORMANCE
# ==============================
st.subheader("📊 Model Performance")

report = classification_report(y, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.text("Classification Report")
    st.text(report)

with col2:
    st.text("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))

# ==============================
# DOWNLOAD REPORT
# ==============================
st.download_button(
    label="📥 Download Report",
    data=report,
    file_name="model_report.txt"
)

# ==============================
# ROC CURVE
# ==============================
st.subheader("📈 ROC Curve")

fpr, tpr, _ = roc_curve(y, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], '--')
ax.set_title("ROC Curve")

st.pyplot(fig)

# ==============================
# HEATMAP
# ==============================
st.subheader("🔥 Feature Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax2)

st.pyplot(fig2)

# ==============================
# MODEL INFO
# ==============================
st.subheader("🤖 Model Info")

st.write("""
- Model: Random Forest  
- Uses SMOTE for imbalance  
- Optimized for fraud detection  
- Adjustable threshold  
""")