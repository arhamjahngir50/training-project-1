import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# ==============================
# DATASET 1
# ==============================

X1 = np.array([3,4,5,6,7,8]).reshape(-1,1)
Y1 = np.array([2.5,3.2,3.8,6.5,11.5,14.6])

# ----- Simple Linear Regression -----
linear1 = LinearRegression()
linear1.fit(X1, Y1)
Y1_pred_linear = linear1.predict(X1)

# R2
r2_linear1 = r2_score(Y1, Y1_pred_linear)

# ----- Polynomial Regression (Degree 2) -----
poly = PolynomialFeatures(degree=2)
X1_poly = poly.fit_transform(X1)

poly_model1 = LinearRegression()
poly_model1.fit(X1_poly, Y1)
Y1_pred_poly = poly_model1.predict(X1_poly)

r2_poly1 = r2_score(Y1, Y1_pred_poly)

print("===== DATASET 1 =====")
print("Linear Equation: y =", linear1.intercept_, "+", linear1.coef_[0], "x")
print("Linear R2:", r2_linear1)

print("Polynomial Equation: y =", 
      poly_model1.intercept_, "+",
      poly_model1.coef_[1], "x +",
      poly_model1.coef_[2], "x^2")
print("Polynomial R2:", r2_poly1)

# Plot Dataset 1
plt.scatter(X1, Y1)
plt.plot(X1, Y1_pred_linear)
plt.plot(X1, Y1_pred_poly)
plt.title("Dataset 1")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("q1-a.png")

# ==============================
# DATASET 2
# ==============================

X2 = np.array([1,2,3,4]).reshape(-1,1)
Y2 = np.array([2,4,9,16])

# ----- Simple Linear Regression -----
linear2 = LinearRegression()
linear2.fit(X2, Y2)
Y2_pred_linear = linear2.predict(X2)

r2_linear2 = r2_score(Y2, Y2_pred_linear)

# ----- Polynomial Regression (Degree 2) -----
X2_poly = poly.fit_transform(X2)

poly_model2 = LinearRegression()
poly_model2.fit(X2_poly, Y2)
Y2_pred_poly = poly_model2.predict(X2_poly)

r2_poly2 = r2_score(Y2, Y2_pred_poly)

print("\n===== DATASET 2 =====")
print("Linear Equation: y =", linear2.intercept_, "+", linear2.coef_[0], "x")
print("Linear R2:", r2_linear2)

print("Polynomial Equation: y =", 
      poly_model2.intercept_, "+",
      poly_model2.coef_[1], "x +",
      poly_model2.coef_[2], "x^2")
print("Polynomial R2:", r2_poly2)

# Plot Dataset 2
plt.scatter(X2, Y2)
plt.plot(X2, Y2_pred_linear)
plt.plot(X2, Y2_pred_poly)
plt.title("Dataset 2")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("q1-b.png")