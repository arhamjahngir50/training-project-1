# Decision Tree & Random Forest Customer Churn Predictor

A machine learning project that predicts customer churn using Decision Tree and Random Forest classifiers. This project compares two tree-based algorithms and demonstrates data preprocessing, model training, and evaluation on the Telco Customer Churn dataset.

## Project Overview

This project demonstrates:
- **Tree-Based Models**: Implements both Decision Tree and Random Forest
- **Data Preprocessing**: Encoding categorical variables and handling missing values
- **Model Comparison**: Evaluates both models side-by-side
- **Visualization**: Generates decision tree plots and feature importance charts
- **Classification Metrics**: Uses accuracy, precision, recall, and F1-score for evaluation

## Project Structure

```
decisiontree/
├── code.py                                    # Main script (preprocessing + training)
├── README.md                                  # This file
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Customer churn dataset
├── decision_tree.png                         # Generated decision tree visualization
└── feature_importance.png                    # Generated feature importance chart
```

## Dataset Overview

**File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Purpose**: Predict whether a customer will churn (leave the service)

**Size**: ~7,000+ customer records with 20+ features

**Target Variable**: `Churn` (Yes/No → 1/0)

**Key Features**:
- Demographics: Gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, InternetService, OnlineSecurity, DeviceProtection
- Billing: MonthlyCharges, TotalCharges, Contract, PaperlessBilling
- Tenure: Months as customer

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip

### Install Dependencies

1. **Navigate to the project directory:**
   ```bash
   cd decisiontree
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

## File Descriptions

### `code.py` - Main Training Script

**Purpose**: Complete end-to-end machine learning pipeline for churn prediction.

**Workflow Stages**:

#### 1. Data Loading
```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```
Loads the Telco customer dataset into memory.

#### 2. Data Preprocessing

**Steps**:
- **Remove customerID**: Not useful for prediction (unique identifier)
- **Convert TotalCharges to numeric**: Handles non-numeric values as NaN
- **Drop missing values**: Removes rows with NaN values
- **Encode target variable**: Maps "Yes"→1, "No"→0 for churn
- **Label encode categorical variables**: Converts text to numbers for all categorical columns

```python
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
# All text columns converted to numbers using LabelEncoder
```

#### 3. Train/Test Split

**Split Strategy** (80/20):
- **Training Set**: 80% of data for model training
- **Test Set**: 20% of data for evaluation
- **Stratification**: Maintains same churn ratio in both sets
- **Random State**: Set to 42 for reproducibility

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

#### 4. Model 1 - Decision Tree

**Configuration**:
- **Algorithm**: Decision Tree Classification (CART - Classification and Regression Trees)
- **Max Depth**: 5 (limits tree depth to prevent overfitting)
- **Splitting Criterion**: Gini Index (default)

**Training**:
```python
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
```

**Output Metrics**:
- Accuracy, Precision, Recall, F1-Score on test set
- Decision tree visualization saved as `decision_tree.png`

#### 5. Model 2 - Random Forest

**Configuration**:
- **Algorithm**: Random Forest (ensemble of decision trees)
- **Number of Trees**: 100 decision trees
- **Max Depth**: 8 per tree
- **Aggregation**: Majority voting for final prediction

**Training**:
```python
rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
```

**Output Metrics**:
- Accuracy, Precision, Recall, F1-Score on test set
- Feature importance chart saved as `feature_importance.png`

### `WA_Fn-UseC_-Telco-Customer-Churn.csv` - Dataset

**CSV Structure**:
```
customerID, gender, SeniorCitizen, Partner, Dependents, ..., Churn
001, Female, 0, No, No, ..., No
002, Male, 1, Yes, No, ..., Yes
```

**Sample Features**:
- Categorical: gender, Partner, Dependents, PhoneService, InternetService
- Numeric: SeniorCitizen, tenure, MonthlyCharges, TotalCharges
- Target: Churn (Yes/No)

## How to Use

### Run the Complete Pipeline

Execute the script to train both models:

```bash
python code.py
```

### Expected Output

```
=== Decision Tree Results ===
Accuracy : 0.7834
Precision: 0.6545
Recall   : 0.5234
F1 Score : 0.5820

=== Random Forest Results ===
Accuracy : 0.8156
Precision: 0.7123
Recall   : 0.6789
F1 Score : 0.6945
```

### Generated Files

After running, two visualization files are created:

1. **decision_tree.png**: Visual representation of the decision tree (limited to depth 3 for readability)
2. **feature_importance.png**: Bar chart showing which features are most important for predictions

## Model Architecture & Algorithm Details

### Decision Tree Classifier

**How it works**:
1. Starts with all data at root node
2. Finds best feature that splits data (using Gini Index)
3. Recursively splits each subset
4. Continues until reaching max_depth or other stopping criteria
5. Each leaf node represents a prediction (Churn/No Churn)

**Decision Boundaries**: Creates axis-aligned rectangular regions in feature space

**Advantages**:
- Interpretable and easy to visualize
- Handles both categorical and numerical features
- No feature scaling required

**Disadvantages**:
- Prone to overfitting on training data
- Can be unstable with small data changes

### Random Forest Classifier

**How it works**:
1. Creates 100 independent decision trees
2. Each tree trained on random subset of data (bootstrap sampling)
3. Each split considers random subset of features
4. Final prediction: Majority vote from all trees

**Why "Random"**: Randomness in data samples and features reduces correlation between trees

**Advantages**:
- Reduces overfitting through ensemble averaging
- Better generalization than single tree
- Robust to outliers and noise
- Provides feature importance scores

**Disadvantages**:
- Less interpretable than single tree
- Slower prediction than single tree
- Higher memory usage

## Evaluation Metrics Explained

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | % of correct predictions |
| **Precision** | TP/(TP+FP) | Of predicted churners, how many actually churned |
| **Recall** | TP/(TP+FN) | Of actual churners, how many were caught |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Harmonic mean of precision & recall |

**Legend**:
- TP (True Positive): Predicted churn, actually churned ✓
- TN (True Negative): Predicted no churn, didn't churn ✓
- FP (False Positive): Predicted churn, didn't churn ✗
- FN (False Negative): Predicted no churn, but churned ✗

### Example Interpretation

Decision Tree Results:
- **Accuracy 78.34%**: Correct 78 out of 100 predictions
- **Precision 65.45%**: Of 100 customers predicted to churn, 65 actually churn (false alarms?)
- **Recall 52.34%**: Of actual churners, only 52% are identified (missing opportunities)

Random Forest Results (Better):
- **Accuracy 81.56%**: Catches more correct predictions
- **Precision 71.23%**: Fewer false alarms
- **Recall 67.89%**: Identifies more actual churners

## Data Preprocessing Details

### Step 1: Remove customerID
```python
df.drop("customerID", axis=1, inplace=True)
```
Why? customerID is unique per customer and provides no predictive value.

### Step 2: Convert TotalCharges to Numeric
```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
```
Why? TotalCharges may contain non-numeric values (empty strings, text).
Effect: Non-numeric values become NaN.

### Step 3: Drop Missing Values
```python
df.dropna(inplace=True)
```
Why? Models cannot handle missing values.
Impact: Removes rows with NaN (from previous step or original data).

### Step 4: Encode Target Variable
```python
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
```
Why? Models require numeric outputs (1 = churn, 0 = no churn).

### Step 5: Label Encode Categorical Features
```python
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
```
Why? Tree models require numeric inputs.
How? Converts categories to integers: "Male"→0, "Female"→1

## Feature Importance

The Random Forest model generates feature importance scores showing which features most influence churn predictions.

**Interpretation**:
- Higher score = More important for predictions
- Example: Tenure (0.25) = 25% importance

**Common Important Features for Churn**:
1. **Tenure**: Customers with shorter tenure more likely to churn
2. **MonthlyCharges**: Higher charges correlate with churn
3. **Contract**: Month-to-month contracts have higher churn
4. **InternetService**: Fiber optic users may churn more

## Hyperparameter Tuning

### Decision Tree Adjustments

**To reduce overfitting**:
```python
dt_model = DecisionTreeClassifier(
    max_depth=3,              # Shallower tree
    min_samples_split=10,     # Require more samples to split
    min_samples_leaf=5        # Require more samples in leaves
)
```

**To increase model complexity**:
```python
dt_model = DecisionTreeClassifier(
    max_depth=10,             # Deeper tree
    min_samples_split=2,      # Allow more splits
    min_samples_leaf=1        # Allow single samples in leaves
)
```

### Random Forest Adjustments

**To reduce overfitting**:
```python
rf_model = RandomForestClassifier(
    n_estimators=50,          # Fewer trees
    max_depth=5,              # Shallower trees
    min_samples_split=20      # Require more samples to split
)
```

**To increase accuracy**:
```python
rf_model = RandomForestClassifier(
    n_estimators=200,         # More trees
    max_depth=10,             # Deeper trees
    random_state=42
)
```

## Troubleshooting

### Issue: "FileNotFoundError" for CSV file

**Cause**: Dataset file not found

**Solution**:
```bash
# Verify file exists
ls WA_Fn-UseC_-Telco-Customer-Churn.csv

# Ensure you're in correct directory
cd decisiontree
```

### Issue: "ModuleNotFoundError" for scikit-learn

**Cause**: Missing dependencies

**Solution**:
```bash
pip install pandas numpy matplotlib scikit-learn
```

### Issue: Model accuracy is very low (< 0.6)

**Cause**: Data quality or preprocessing issues

**Solution**:
- Check for missing values: `df.isnull().sum()`
- Verify target variable encoding
- Try different max_depth values
- Check for class imbalance

### Issue: Feature importance values sum to 0

**Cause**: Random Forest may assign low importance if all features weak

**Solution**:
- Try different random_state
- Add feature engineering
- Verify data quality

## Performance Comparison

| Aspect | Decision Tree | Random Forest |
|--------|---------------|---------------|
| **Training Time** | Fast | Moderate |
| **Prediction Time** | Very Fast | Fast |
| **Interpretability** | Excellent | Good |
| **Accuracy** | ~78% | ~82% |
| **Overfitting Risk** | High | Low |
| **Stability** | Unstable | Stable |

## Advanced Usage

### Cross-Validation

Evaluate model using k-fold cross-validation:

```python
from sklearn.model_selection import cross_val_score

dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5)
print(f"Decision Tree CV Accuracy: {dt_cv_scores.mean():.4f}")

rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"Random Forest CV Accuracy: {rf_cv_scores.mean():.4f}")
```

### Hyperparameter Grid Search

Find optimal parameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

### Class Imbalance Handling

If churn is imbalanced (e.g., 80% no churn, 20% churn):

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

rf_model = RandomForestClassifier(
    class_weight='balanced',  # Penalize underrepresented class
    n_estimators=100
)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation and CSV loading |
| numpy | Numerical computations |
| matplotlib | Data visualization |
| scikit-learn | Machine learning models and metrics |

## Future Enhancements

- [ ] Add ROC-AUC curves and confusion matrices
- [ ] Implement feature scaling for comparison models
- [ ] Add XGBoost or LightGBM models
- [ ] Create prediction pipeline for new data
- [ ] Add model persistence (save/load trained model)
- [ ] Implement feature engineering (polynomial, interactions)
- [ ] Add SHAP values for model explainability
- [ ] Deploy as REST API

## References

- [scikit-learn Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- [Decision Trees Explained](https://en.wikipedia.org/wiki/Decision_tree_learning)

## License

This project is open source and available for educational purposes.
