# churn_models.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ----------------------------
# 2. Preprocessing
# ----------------------------
df.drop("customerID", axis=1, inplace=True)      # inplace true meaning donot make a new df 

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # convert any non  numeric to numcerc # if value cannot be numeric put in nan
df.dropna(inplace=True)             # remve missing values 

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

categorical_cols = df.select_dtypes(include=["object"]).columns   # select cols that are string/text


le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


X = df.drop("Churn", axis=1)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(            #   stratify=y keeps churn ratio same in test and train 
    X, y, test_size=0.2, random_state=42, stratify=y               # RANDOM STATE 42 SO IT REMAINS SAME EVERY TIME 
)

# ======================================================
# PART 1: DECISION TREE
# ======================================================

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)   # simple tree makes a binary tree CART based on gini index
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\n=== Decision Tree Results ===")
print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall   :", recall_score(y_test, y_pred_dt))
print("F1 Score :", f1_score(y_test, y_pred_dt))

# ----------------------------
# Visualize Decision Tree
# ----------------------------
plt.figure(figsize=(20,10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    max_depth=3  # limit depth for readability
)
plt.title("Decision Tree Visualization (Depth=3)")
plt.savefig("decision_tree.png")
plt.close()


# ======================================================
# PART 2: RANDOM FOREST
# ======================================================

rf_model = RandomForestClassifier(
    n_estimators=100,                 # builds 100 trees 
    max_depth=8,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest Results ===")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))


# ----------------------------
# Feature Importance (Random Forest)
# ----------------------------
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
plt.title("Random Forest Feature Importance")
plt.savefig("feature_importance.png")
plt.close()