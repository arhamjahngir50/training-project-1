# Import libraries
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample dataset (2 features)
X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5],
    [6, 6]
])

# Target labels (0 or 1)
y = np.array([0, 0, 0, 1, 1, 1])

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Take user input
x1 = int(input("Enter value for x1: "))
x2 = int(input("Enter value for x2: "))

# Predict class (0 or 1)
prediction = model.predict([[x1, x2]])

# Predict probability
probability = model.predict_proba([[x1, x2]])

print("Predicted Class:", prediction)
print("Prediction Probability:", probability)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Predict on training data
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred)

# Precision (positive class = 1)
precision = precision_score(y, y_pred)

# Recall
recall = recall_score(y, y_pred)

# F1 Score
f1 = f1_score(y, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred)

print("\n--- Evaluation Metrics ---")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)