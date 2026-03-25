import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

print("Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Amount
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Training model...")
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train_res, y_train_res)

# Save model + scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model saved as model.pkl")