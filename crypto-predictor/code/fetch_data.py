import requests
import csv
import os


def fetch_bitcoin_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "30"
    }

    try:
        print("Sending request...")
        response = requests.get(url, params=params)

        print("Status Code:", response.status_code)

        if response.status_code != 200:
            print("Error:", response.text)
            return

        data = response.json()

        if "prices" not in data:
            print("Unexpected API response:", data)
            return

        prices = data["prices"]

        print(f"Total price records received: {len(prices)}")

        os.makedirs("data", exist_ok=True)

        file_path = os.path.join("data", "btc_prices.csv")

        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "price"])

            for price in prices:
                writer.writerow(price)

        print(f"Data successfully saved to {file_path}")

    except Exception as e:
        print("Error occurred:", str(e))