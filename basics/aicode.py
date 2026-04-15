# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Sample data: X = feature, y = target
# X = np.array([[1],[2],[3],[4],[55]])
# y = np.array([2,4,6,8,10])

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# # Train model
# lr = LinearRegression()
# lr2 =LogisticRegression()

# lr.fit(X_train, y_train)

# # Predict
# y_pred = lr.predict(X_test)

# print(X_test)
# # Evaluate
# print("Predictions:", y_pred)
# print("MSE:", mean_squared_error(y_test, y_pred))




################### NAIVE BAYES 
# from sklearn.naive_bayes import GaussianNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Sample data: X = features, y = classes
# X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
# y = np.array([0,0,1,1,1])

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# # Train model
# nb = GaussianNB()
# nb.fit(X_train, y_train)

# # Predict
# y_pred = nb.predict(X_test)

# # Evaluate
# print(X_test)
# print("Predictions:", y_pred)
# print("Accuracy:", accuracy_score(y_test, y_pred))



# #################### SIMPLE RANDOM FOREST 
# # 1️⃣ Import libraries
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # 2️⃣ Load dataset
# data = load_iris()
# X = data.data
# y = data.target

# # 3️⃣ Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4️⃣ Create Random Forest Classifier
# clf = RandomForestClassifier(
#     n_estimators=100,   # number of trees
#     max_features='sqrt', # features considered at each split
#     random_state=42
# )

# # 5️⃣ Train the model
# clf.fit(X_train, y_train)

# # 6️⃣ Make predictions
# y_pred = clf.predict(X_test)

# # 7️⃣ Evaluate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # 8️⃣ Feature importance (optional)
# print("Feature Importances:", clf.feature_importances_)






############################## SVM ---------___ SUPPORT VECTOR MACHINE 

# 1️⃣ Import libraries
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# ============================
# 2️⃣ SVM Classification Example
# ============================

# Load dataset
iris = load_iris()
X_cls = iris.data
y_cls = iris.target

# Train-test split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Create SVM classifier
svm_clf = SVC(
    kernel='rbf',   # 'linear', 'poly', 'sigmoid' also possible
    C=1.0,          # Regularization parameter
    gamma='scale',  # Kernel coefficient for 'rbf', 'poly', 'sigmoid'
    random_state=42
)

# Train
svm_clf.fit(Xc_train, yc_train)

# Predict
yc_pred = svm_clf.predict(Xc_test)

# Evaluate
print("SVM Classification Accuracy:", accuracy_score(yc_test, yc_pred))


# ============================
# 3️⃣ SVM Regression Example
# ============================

# Load dataset
housing = fetch_california_housing()
X_reg = housing.data
y_reg = housing.target

# Train-test split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Create SVR model
svm_reg = SVR(
    kernel='rbf',   # 'linear', 'poly', etc.
    C=10.0,         # Regularization
    epsilon=0.2,    # Width of epsilon-insensitive tube
    gamma='scale'
)

# Train
svm_reg.fit(Xr_train, yr_train)

# Predict
yr_pred = svm_reg.predict(Xr_test)

# Evaluate
import numpy as np
mse = mean_squared_error(yr_test, yr_pred)  # returns mean squared error
rmse = np.sqrt(mse)                          # take square root
print("SVM Regression RMSE:", rmse)

