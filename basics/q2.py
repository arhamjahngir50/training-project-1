import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Data
X = np.array([1,2,3,4,5,6,7]).reshape(-1,1)
Y = np.array([50000,55000,65000,80000,110000,150000,200000])

# Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, Y)

# Predictions
Y_pred = model.predict(X_poly)

# R2 Score
r2 = r2_score(Y, Y_pred)

# Print Equation
print("Polynomial Regression Equation:")
print("Salary =",
      model.intercept_,
      "+", model.coef_[1], "* (Experience)",
      "+", model.coef_[2], "* (Experience^2)")

print("\nR² Score:", r2)

# Plot
plt.scatter(X, Y)
plt.plot(X, Y_pred)
plt.title("Polynomial Regression (Degree 2)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.savefig("q2_polynomial_plot.png")