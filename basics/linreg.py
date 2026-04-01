# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dataset (2 features: x1 and x2)
# Each row = [x1, x2]
X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5]
])

# Target values
y = np.array([5, 6, 9, 10, 13])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

x= int(input("enter a value"))
y= int(input("enter another value "))
# Make prediction (example: x1=6, x2=7)
prediction = model.predict([[x, y]])
print("Prediction:", prediction)



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# ---- EVALUATION METRICS ----
# Predict on training data
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n--- Evaluation Metrics ---")
print("MAE:", mae)       # on average how much am i wrong
print("MSE:", mse)      # punishes large errors largely 
print("RMSE:", rmse)      # same unit as input so becomes easy to understand. Still is punishing large errors 
print("R2 Score:", r2)