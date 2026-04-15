import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Mall_Customers.csv")

print("Dataset Preview:")
print(df.head())
print("\n")

# -------------------------------
# Feature Selection
# -------------------------------
# We use Income and Spending Score for segmentation
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)    # large attributes small , small attributes large 

# -------------------------------
# Elbow Method (Optional - To find best K)
# -------------------------------
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("elbow_plot.png")
plt.close()
# -------------------------------
# Apply K-Means
# -------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------
# Visualization
# -------------------------------
plt.figure()
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.savefig("clusters.png")
plt.close()
# -------------------------------
# Cluster Analysis
# -------------------------------
print("\nCluster Summary (Means):")

numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
print(df.groupby("Cluster")[numeric_cols].mean())

print("\nCluster Counts:")
print(df["Cluster"].value_counts())