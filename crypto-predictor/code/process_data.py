# import pandas as pd


# def load_and_process():
#     df = pd.read_csv("data/btc_prices.csv")

#     # Convert timestamp
#     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

#     # Sort by date
#     df = df.sort_values("timestamp")

#     # Create next-day price column
#     df["next_day_price"] = df["price"].shift(-1)

#     df = df.dropna()

#     return df




import pandas as pd


def load_and_process():
    df = pd.read_csv("data/btc_prices.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp")

    # ---- Feature Engineering ----
    df["ma_3"] = df["price"].rolling(window=3).mean()
    df["ma_7"] = df["price"].rolling(window=7).mean()   # rolling window averages used for estimation in trading 
    df["price_diff"] = df["price"].diff()               # difference price[i]- price[i-1]
    df["pct_change"] = df["price"].pct_change()         # (price[i] - price[i-1]) / price[i-1]

    # Target column
    df["next_day_price"] = df["price"].shift(-1)          # Y 

    df = df.dropna()

    return df