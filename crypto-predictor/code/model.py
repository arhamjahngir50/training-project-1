from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


def train_model(df):
    # Feature columns (must exist in dataframe)
    feature_columns = [
        "price",
        "ma_3",
        "ma_7",
        "price_diff",
        "pct_change"
    ]

    X = df[feature_columns].values
    y = df["next_day_price"].values

    # ---- Time Series Split (NO shuffle) ----
    split_index = int(len(df) * 0.8)          # calulate 80 percent of dataset length 

    X_train = X[:split_index]                 # first 80 percent training 
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    # ---- Train Model ----
    model = LinearRegression()
    model.fit(X_train, y_train)

    # ---- Predictions ----
    predictions = model.predict(X_test)

    # ---- Evaluation ----
    r2_score = model.score(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("Model Performance:")
    print(f"R2 Score: {r2_score}")
    print(f"RMSE: {rmse}")

    return model