import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==============================
# LOAD DATA (SMALL SAMPLE)
# ==============================
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    df = df.sample(30000, random_state=42)  # 🔥 reduced size
    return df

df = load_data()

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# TRAIN MODELS (ONCE)
# ==============================
rf_model = RandomForestClassifier(n_estimators=20, max_depth=6)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(n_estimators=50, max_depth=4, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict(model_name, *inputs):
    input_dict = {f"V{i+1}": inputs[i] for i in range(28)}
    input_dict["Amount"] = scaler.transform([[inputs[28]]])[0][0]
    input_dict["Time"] = inputs[29]

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[X.columns]

    model = rf_model if model_name == "Random Forest" else xgb_model

    prob = model.predict_proba(input_df)[0][1]
    result = "⚠️ Fraud" if prob > 0.3 else "✅ Legit"

    return f"{result} | Probability: {prob:.4f}"

# ==============================
# UI
# ==============================
inputs = [gr.Slider(-10, 10, value=0, label=f"V{i}") for i in range(1, 29)]
inputs += [
    gr.Number(label="Amount"),
    gr.Number(label="Time")
]

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Dropdown(["Random Forest", "XGBoost"], label="Model")] + inputs,
    outputs="text",
    title="💳 Fraud Detection System (Gradio)",
    description="Predict fraudulent transactions using ML models"
)

demo.launch()