# ================================
# PCA Analysis with Synthetic Data
# ================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

np.random.seed(42)

# -------------------------------
# Create synthetic dataset
# -------------------------------
n = 200

df = pd.DataFrame({
    "engine_size": np.random.normal(2.5, 0.8, n),
})

df["horsepower"] = df["engine_size"] * 40 + np.random.normal(0, 10, n)
df["curb_weight"] = df["engine_size"] * 500 + np.random.normal(0, 100, n)
df["highway_mpg"] = 50 - df["engine_size"] * 5 + np.random.normal(0, 2, n)

# Price depends on multiple factors
df["price"] = (
    df["engine_size"] * 5000 +
    df["horsepower"] * 100 +
    df["curb_weight"] * 5 +
    np.random.normal(0, 2000, n)
)

# Add categorical columns for analysis
df["make"] = np.random.choice(["toyota", "bmw", "porsche", "nissan"], n)
df["body_style"] = np.random.choice(["sedan", "wagon", "hatchback", "convertible"], n)

# -------------------------------
# Feature selection
# -------------------------------
features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]

X = df.copy()
y = X.pop('price')
X = X.loc[:, features]

# -------------------------------
# Standardize features
# -------------------------------
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# -------------------------------
# Apply PCA
# -------------------------------
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

print("\n=== PCA Components ===")
print(X_pca.head())

# -------------------------------
# Loadings
# -------------------------------
loadings = pd.DataFrame(
    pca.components_.T,
    columns=component_names,
    index=X.columns,
)

print("\n=== Loadings ===")
print(loadings)

# -------------------------------
# Explained variance plot
# -------------------------------
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance")
plt.grid()
plt.show()

# -------------------------------
# Mutual Information
# -------------------------------
mi = mutual_info_regression(X_pca, y)
mi_scores = pd.Series(mi, index=component_names)

print("\n=== MI Scores ===")
print(mi_scores.sort_values(ascending=False))

# -------------------------------
# Analyze PC3
# -------------------------------
idx = X_pca["PC3"].sort_values(ascending=False).index
cols = ["make", "body_style", "horsepower", "curb_weight"]

print("\n=== Top PC3 Rows ===")
print(df.loc[idx, cols].head(10))

# -------------------------------
# New Feature
# -------------------------------
df["sports_or_wagon"] = df["curb_weight"] / df["horsepower"]

# -------------------------------
# Plot
# -------------------------------
plt.figure()
sns.regplot(x="sports_or_wagon", y="price", data=df, order=2)
plt.title("Sports vs Wagon vs Price")
plt.show()