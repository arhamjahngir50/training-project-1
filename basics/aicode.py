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




# #################### RANDOM FOREST 
# # ===============================
# # 1. IMPORT LIBRARIES
# # ===============================
# import numpy as np
# import pandas as pd

# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# # ===============================
# # 2. LOAD DATA (EXAMPLE)
# # ===============================
# # Replace with your dataset
# from sklearn.datasets import load_iris, fetch_california_housing

# # Classification dataset
# data_cls = load_iris()
# X_cls = data_cls.data
# y_cls = data_cls.target

# # Regression dataset
# data_reg = fetch_california_housing()
# X_reg = data_reg.data
# y_reg = data_reg.target

# # ===============================
# # 3. TRAIN-TEST SPLIT
# # ===============================
# Xc_train, Xc_test, yc_train, yc_test = train_test_split(
#     X_cls, y_cls, test_size=0.2, random_state=42
# )

# Xr_train, Xr_test, yr_train, yr_test = train_test_split(
#     X_reg, y_reg, test_size=0.2, random_state=42
# )

# # ===============================
# # 4. PIPELINE (Handling Missing + Scaling)
# # ===============================
# pipeline_cls = Pipeline([
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler()),
#     ('rf', RandomForestClassifier(
#         n_estimators=100,
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         max_features='sqrt',   # feature randomness
#         bootstrap=True,
#         oob_score=True,
#         random_state=42,
#         n_jobs=-1
#     ))
# ])

# pipeline_reg = Pipeline([
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler()),
#     ('rf', RandomForestRegressor(
#         n_estimators=100,
#         max_depth=None,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         max_features='sqrt',
#         bootstrap=True,
#         oob_score=True,
#         random_state=42,
#         n_jobs=-1
#     ))
# ])

# # ===============================
# # 5. TRAIN MODELS
# # ===============================
# pipeline_cls.fit(Xc_train, yc_train)
# pipeline_reg.fit(Xr_train, yr_train)

# # ===============================
# # 6. PREDICTIONS
# # ===============================
# yc_pred = pipeline_cls.predict(Xc_test)
# yr_pred = pipeline_reg.predict(Xr_test)

# # ===============================
# # 7. EVALUATION
# # ===============================
# print("Classification Accuracy:", accuracy_score(yc_test, yc_pred))
# print(classification_report(yc_test, yc_pred))

# print("Regression RMSE:", np.sqrt(mean_squared_error(yr_test, yr_pred)))

# # ===============================
# # 8. OOB SCORE (Out-of-Bag)
# # ===============================
# rf_cls = pipeline_cls.named_steps['rf']
# rf_reg = pipeline_reg.named_steps['rf']

# print("OOB Score (Classification):", rf_cls.oob_score_)
# print("OOB Score (Regression):", rf_reg.oob_score_)

# # ===============================
# # 9. FEATURE IMPORTANCE
# # ===============================
# importances_cls = rf_cls.feature_importances_
# importances_reg = rf_reg.feature_importances_

# print("Feature Importance (Classification):", importances_cls)
# print("Feature Importance (Regression):", importances_reg)

# # ===============================
# # 10. CROSS VALIDATION
# # ===============================
# cv_scores = cross_val_score(
#     pipeline_cls, X_cls, y_cls, cv=5, scoring='accuracy'
# )
# print("Cross-validation Accuracy:", cv_scores.mean())

# # ===============================
# # 11. HYPERPARAMETER TUNING
# # ===============================
# param_grid = {
#     'rf__n_estimators': [50, 100, 200],
#     'rf__max_depth': [None, 10, 20],
#     'rf__max_features': ['sqrt', 'log2'],
#     'rf__min_samples_split': [2, 5],
# }

# grid_search = GridSearchCV(
#     pipeline_cls,
#     param_grid,
#     cv=3,
#     scoring='accuracy',
#     n_jobs=-1
# )

# grid_search.fit(Xc_train, yc_train)

# print("Best Params:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)

# # ===============================
# # 12. SAVE / LOAD MODEL
# # ===============================
# import joblib

# joblib.dump(pipeline_cls, "rf_model.pkl")
# loaded_model = joblib.load("rf_model.pkl")

# # Test loaded model
# print("Loaded Model Prediction:", loaded_model.predict(Xc_test[:5]))




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
rmse = mean_squared_error(yr_test, yr_pred, squared=False)           ## some problem here 
print("SVM Regression RMSE:", rmse)




