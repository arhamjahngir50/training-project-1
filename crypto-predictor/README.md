# Bitcoin Price Predictor

A machine learning project that predicts Bitcoin prices using historical data and engineered features. This project demonstrates data fetching, feature engineering, and linear regression for time series forecasting.

## Project Overview

This project demonstrates:
- **API Integration**: Fetches real-time Bitcoin data from CoinGecko API
- **Feature Engineering**: Creates technical indicators from price data
- **Time Series Modeling**: Uses Linear Regression for price prediction
- **Data Pipeline**: Automated workflow from data collection to model evaluation

## Project Structure

```
crypto-predictor/
├── requirements.txt               # Python package dependencies
├── README.md                      # This file
├── code/                          # Main project code
│   ├── main.py                   # Entry point - orchestrates the pipeline
│   ├── fetch_data.py             # Fetches Bitcoin data from CoinGecko API
│   ├── process_data.py           # Data preprocessing & feature engineering
│   ├── model.py                  # Model training and evaluation
│   └── data/
│       └── btc_prices.csv        # Cached Bitcoin price data
└── data/                         # Alternative data storage location
    └── btc_prices.csv            # Bitcoin price dataset
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip or conda
- Internet connection (to fetch data from API)

### Install Dependencies

1. **Navigate to the project directory:**
   ```bash
   cd crypto-predictor
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install requests pandas scikit-learn numpy
   ```

   Or create from requirements file:
   ```bash
   pip install -r requirements.txt
   ```

## File Descriptions

### `main.py` - Pipeline Orchestrator

**Purpose**: Entry point that coordinates the entire workflow.

**Workflow**:
1. Fetches Bitcoin price data for the past 30 days
2. Loads and processes the data
3. Trains the prediction model

**Usage**:
```bash
python main.py
```

### `fetch_data.py` - Data Collection

**Purpose**: Fetches historical Bitcoin price data from CoinGecko API.

**Key Features**:
- Calls CoinGecko API: `https://api.coingecko.com/api/v3/coins/bitcoin/market_chart`
- Retrieves 30 days of historical price data in USD
- Returns timestamp (milliseconds) and price per data point
- Saves data to CSV file: `data/btc_prices.csv`
- Includes error handling for API failures

**API Parameters**:
```python
{
    "vs_currency": "usd",      # Currency comparison
    "days": "30"               # Number of days of history
}
```

**Output Format** (btc_prices.csv):
```
timestamp,price
1701374400000,42500.50
1701460800000,42600.75
...
```

### `process_data.py` - Feature Engineering

**Purpose**: Loads raw data and creates features for model training.

**Data Processing Steps**:

1. **Load Data**: Reads CSV file into pandas DataFrame
2. **Timestamp Conversion**: Converts Unix milliseconds to datetime
3. **Sorting**: Ensures data is sorted chronologically
4. **Feature Engineering**:
   - `ma_3`: 3-day moving average of price
   - `ma_7`: 7-day moving average of price
   - `price_diff`: Daily price change (price[i] - price[i-1])
   - `pct_change`: Percentage change ((price[i] - price[i-1]) / price[i-1])
5. **Target Column**: Creates `next_day_price` by shifting current price forward 1 day
6. **Data Cleaning**: Removes rows with missing values (NaN)

**Output**: Processed DataFrame ready for model training

### `model.py` - Model Training & Evaluation

**Purpose**: Trains a Linear Regression model and evaluates performance.

**Model Architecture**:
- **Algorithm**: Linear Regression (sklearn)
- **Input Features**: 5 engineered features
- **Output**: Predicted next-day Bitcoin price

**Training Process**:

1. **Feature Selection**:
   ```python
   Features = ["price", "ma_3", "ma_7", "price_diff", "pct_change"]
   Target = "next_day_price"
   ```

2. **Data Split** (Time Series Split, 80/20):
   - **Training Set**: First 80% of data (chronological)
   - **Test Set**: Last 20% of data (chronological)
   - ⚠️ **Important**: No shuffling to maintain time series order

3. **Model Training**: Fits LinearRegression on training data

4. **Predictions**: Makes predictions on test set

5. **Evaluation Metrics**:
   - **R² Score**: Coefficient of determination (0-1, higher is better)
   - **RMSE**: Root Mean Squared Error (lower is better)

## How to Use

### Step 1: Run the Complete Pipeline

Execute the entire workflow in one command:

```bash
python code/main.py
```

This will:
1. Fetch 30 days of Bitcoin data from the API
2. Process data and create features
3. Train the model
4. Display performance metrics

### Step 2: View Results

After running, you'll see output similar to:
```
Sending request...
Status Code: 200
Total price records received: 720
Data successfully saved to data/btc_prices.csv

Model Performance:
R2 Score: 0.87
RMSE: 1205.34
```

### Step 3: Interpret Results

- **R² Score of 0.87**: Model explains 87% of the variance in Bitcoin price movements
- **RMSE of 1205.34**: Average prediction error is ±$1,205

## Data Pipeline Flow

```
┌──────────────────────────────────────┐
│  Step 1: Fetch Data (fetch_data.py)  │
│  - Call CoinGecko API                │
│  - Get 30 days of BTC prices         │
│  - Save to CSV                       │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Step 2: Process Data (process_data.py)│
│  - Load CSV                          │
│  - Convert timestamps                │
│  - Engineer features (MA, diff, %)   │
│  - Create target column              │
│  - Remove NaN values                 │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Step 3: Train Model (model.py)       │
│  - Split data (80/20, chronological) │
│  - Fit LinearRegression              │
│  - Make predictions                  │
│  - Evaluate (R², RMSE)               │
└──────────────────────────────────────┘
```

## Feature Engineering Explained

### Moving Averages (ma_3, ma_7)
**Purpose**: Smooths price volatility to identify trends

- **ma_3**: Average of last 3 days' prices
- **ma_7**: Average of last 7 days' prices
- **Use**: Help model identify short and medium-term trends

### Price Difference (price_diff)
**Formula**: `price[i] - price[i-1]`

**Purpose**: Captures daily price momentum
- Positive value: Price increased
- Negative value: Price decreased

### Percentage Change (pct_change)
**Formula**: `(price[i] - price[i-1]) / price[i-1]`

**Purpose**: Normalizes price changes (useful for assets with different price scales)
- Shows relative change, not absolute change

## Model Performance Interpretation

| Metric | Range | Interpretation |
|--------|-------|-----------------|
| R² Score | 0-1 | % of variance explained. 0.87 = Good fit |
| RMSE | 0+ | Avg prediction error in dollars. Lower is better |

**Example**: R² = 0.87, RMSE = 1205
- Model explains 87% of Bitcoin price movements
- Average prediction error is ±$1,205

## Time Series Considerations

### Why No Shuffling?
Time series data has temporal dependencies:
- Future prices depend on past prices
- Shuffling breaks this relationship
- **Solution**: Use chronological split (first 80% train, last 20% test)

### Data Leakage Prevention
The time split ensures:
- Training data: Days 1-24 (of 30 days)
- Testing data: Days 25-30
- Model never "sees" test data during training

## Troubleshooting

### Issue: "Connection error" when fetching data

**Cause**: CoinGecko API is unavailable or no internet connection

**Solution**:
- Check internet connection
- Verify CoinGecko API is accessible: https://api.coingecko.com/
- Pre-existing `data/btc_prices.csv` will still work

### Issue: "KeyError: 'prices'" in fetch_data.py

**Cause**: Unexpected API response format

**Solution**:
- Verify API endpoint is correct
- Check if CoinGecko has changed their API structure
- Manually verify API response: https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30

### Issue: "No such file or directory: data/btc_prices.csv"

**Cause**: data directory not created

**Solution**:
- First run `fetch_data.py` to create the directory
- Or manually create: `mkdir -p data/`

### Issue: NaN values in DataFrame

**Cause**: Insufficient data for moving averages at start of dataset

**Solution**: This is normal and handled by `df.dropna()` in process_data.py

### Issue: ModuleNotFoundError

**Cause**: Missing dependencies

**Solution**:
```bash
pip install requests pandas scikit-learn numpy
```

## Dependencies

| Package | Purpose |
|---------|---------|
| requests | HTTP requests to fetch API data |
| pandas | Data manipulation and analysis |
| scikit-learn | Machine learning (LinearRegression, metrics) |
| numpy | Numerical computations |

## Advanced Usage

### Modifying Prediction Period

Change the number of days fetched in `fetch_data.py`:

```python
params = {
    "vs_currency": "usd",
    "days": "90"  # Change from 30 to 90 days
}
```

### Adjusting Moving Average Windows

In `process_data.py`, modify window sizes:

```python
df["ma_5"] = df["price"].rolling(window=5).mean()   # 5-day MA
df["ma_14"] = df["price"].rolling(window=14).mean() # 14-day MA
```

Then update `feature_columns` in `model.py`.

### Trying Different Algorithms

Replace LinearRegression in `model.py`:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

# Option 1: Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Option 2: Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
model = LinearRegression()
```

## Performance Optimization

### Caching Data
Data is automatically cached in `data/btc_prices.csv` to avoid repeated API calls.

### API Rate Limiting
CoinGecko allows ~10 requests per second. If running multiple times rapidly, consider adding delays:

```python
import time
time.sleep(1)  # Wait 1 second between API calls
```

## Future Enhancements

- [ ] Add more technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Try ensemble models (Random Forest, Gradient Boosting)
- [ ] Implement cross-validation for better evaluation
- [ ] Add prediction confidence intervals
- [ ] Real-time prediction API endpoint
- [ ] Support for multiple cryptocurrencies
- [ ] Backtest model on historical data
- [ ] Add data visualization (matplotlib, plotly)

## References

- [CoinGecko API Documentation](https://www.coingecko.com/en/api/documentation)
- [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Time Series Forecasting Best Practices](https://otexts.com/fpp2/)

## Disclaimer

This project is for **educational purposes only**. Bitcoin price prediction is inherently uncertain and involves significant risk. Do not use this model for actual trading decisions without proper risk management and professional advice.

## License

This project is open source and available for educational purposes.
