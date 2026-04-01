# ============================================
# Q3 - Titanic Survival Prediction
# Simple Linear Regression (As Required)
# + Correct Method (Logistic Regression)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

# ============================================
# 1. Load Dataset
# ============================================

df = pd.read_csv("Titanic-Dataset.csv")

print("First 20 Rows:")
print(df.head(20))

# ============================================
# 2. Data Preprocessing
# ============================================

# Select relevant features (remove unnecessary ones)
df = df[['Survived', 'Sex', 'Age', 'Fare','Pclass']]

# Handle missing Age values
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Convert Sex to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

print("\nAfter Preprocessing:")
print(df.head())

# ============================================
# 3. Define Features and Target
# Simple Linear Regression requires ONE feature
# So we use Fare only
# ============================================

X = df[['Fare']]       # Independent variable
y = df['Survived']     # Dependent variable

# ============================================
# 4. Train-Test Split (75% Training)
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# ============================================
# 5. SIMPLE LINEAR REGRESSION (Required)
# ============================================

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_model.predict(X_test)

# Convert regression output to 0/1
y_pred_class = [1 if i > 0.5 else 0 for i in y_pred_linear]

# Evaluation
r2 = r2_score(y_test, y_pred_linear)
accuracy_linear = accuracy_score(y_test, y_pred_class)

print("\n===== SIMPLE LINEAR REGRESSION =====")
print("Model Equation:")
print("Survived =",
      linear_model.intercept_,
      "+",
      linear_model.coef_[0],
      "* Fare")

print("R² Score:", r2)
print("Accuracy:", accuracy_linear)

# Plot Best Fit Line
plt.scatter(X_train, y_train)
plt.plot(X_train, linear_model.predict(X_train))
plt.xlabel("Fare")
plt.ylabel("Survived")
plt.title("Simple Linear Regression - Best Fit Line")
plt.savefig("titanic_linear_regression.png")
plt.clf()

print("Linear Regression plot saved as titanic_linear_regression.png")

# ============================================
# 6. CORRECT METHOD - LOGISTIC REGRESSION
# ============================================

# Logistic regression works better for classification
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)

print("\n===== LOGISTIC REGRESSION (Correct Method) =====")
print("Accuracy:", accuracy_log)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

# Plot Logistic Curve
X_range = np.linspace(X['Fare'].min(), X['Fare'].max(), 300).reshape(-1,1)
y_prob = log_model.predict_proba(X_range)[:,1]

plt.scatter(X_train, y_train)
plt.plot(X_range, y_prob)
plt.xlabel("Fare")
plt.ylabel("Survival Probability")
plt.title("Logistic Regression Curve")
plt.savefig("titanic_logistic_regression.png")
plt.clf()

print("Logistic Regression plot saved as titanic_logistic_regression.png")

# ============================================
# END OF FILE
# ============================================