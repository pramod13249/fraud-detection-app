# 💳 Fraud Detection System (Production-Ready)

A **machine learning-powered fraud detection system** built with **Streamlit**, designed for real-time and batch prediction of fraudulent credit card transactions.

---

## 🚀 Overview

This project detects fraudulent transactions using a trained **Random Forest model** on a real-world dataset. It supports:

* 🔍 **Real-time prediction** (manual input)
* 📂 **Batch prediction** (CSV upload)
* ⚡ **Fast inference** (no retraining in UI)
* 📊 **Probability-based risk scoring**

---

## 📊 Dataset

* Source: TensorFlow-hosted version of the **Credit Card Fraud Detection dataset**
* Features:

  * `V1–V28` → PCA-transformed features
  * `Time`, `Amount`
  * `Class` → Target (0 = Normal, 1 = Fraud)

---

## 🧠 Model Details

* Algorithm: **Random Forest Classifier**
* Preprocessing:

  * Standard scaling on `Amount`
* Imbalance Handling:

  * **SMOTE (Synthetic Minority Oversampling)**
* Evaluation Metric:

  * ROC-AUC score (evaluated on test set)

---

## 🏗️ Project Structure

```
fraud-detection-app/
│
├── app.py              # Streamlit app (UI + prediction)
├── train.py            # Model training script
├── model.pkl           # Trained model
├── scaler.pkl          # Saved scaler
├── requirements.txt    # Dependencies
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/pramod13249/fraud-detection-app.git
cd fraud-detection-app
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Train the Model (Run Once)

```bash
python train.py
```

This will generate:

* `model.pkl`
* `scaler.pkl`

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🔍 Features

### ✅ Single Prediction

* Input transaction features manually
* Get fraud probability & risk level:

  * 🟢 Low Risk
  * 🟡 Medium Risk
  * 🔴 High Risk

### 📂 Batch Prediction

* Upload CSV file
* Get predictions for multiple transactions
* Download results as CSV

---

## ⚡ Production Improvements

This version is **production-ready**:

* ✅ No data leakage (proper train/test split)
* ✅ Model trained offline (not inside app)
* ✅ Fast startup (loads `.pkl` files)
* ✅ Scalable architecture
* ✅ Clean separation of concerns

---

## 📦 Requirements

* Python 3.8+
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* imbalanced-learn
* joblib

---

## 🌐 Deployment

This app can be deployed on:

* Hugging Face Spaces
* Streamlit Cloud
* AWS / GCP / Azure

---

## 📈 Future Enhancements

* XGBoost / LightGBM for better accuracy
* SHAP explanations for model interpretability
* Threshold tuning UI
* Real-time API integration

---

## 👨‍💻 Author

**Pramod Nagisetty**
GitHub: https://github.com/pramod13249

---

## ⭐ If you found this useful

Give this repo a ⭐ and share it!
