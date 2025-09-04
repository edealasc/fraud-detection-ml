import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, f1_score
import joblib

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
dataset = pd.read_csv("creditcard.csv")
print("Fraudulent cases:", len(dataset[dataset['Class'] == 1]))
print("Normal cases:", len(dataset[dataset['Class'] == 0]))
print("Summary of Amount:", dataset["Amount"].describe())

# -----------------------------
# 2️⃣ Standardize 'Time' and 'Amount'
# -----------------------------
scaler = RobustScaler().fit(dataset[["Time", "Amount"]])
dataset[["Time", "Amount"]] = scaler.transform(dataset[["Time", "Amount"]])

# -----------------------------
# 3️⃣ Separate features and target
# -----------------------------
y = dataset["Class"]
X = dataset.iloc[:, 0:30]

# -----------------------------
# 4️⃣ Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5️⃣ Apply NearMiss undersampling
# -----------------------------
nm = NearMiss()
X_res, y_res = nm.fit_resample(X_train, y_train)

# -----------------------------
# 6️⃣ Train RandomForestClassifier
# -----------------------------
best_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
best_model.fit(X_res, y_res)

# -----------------------------
# 7️⃣ Evaluate model
# -----------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# -----------------------------
# 8️⃣ Plot ROC curve
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"RandomForest (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - RandomForest with NearMiss")
plt.legend()
plt.show()

# -----------------------------
# 9️⃣ Save model and scaler for deployment
# -----------------------------
joblib.dump(best_model, "fraud_detector_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved!")
