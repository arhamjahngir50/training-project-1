# ANN Hyperparameter Tester (Flask + PyTorch)

This directory contains a small web app that retrains a configurable MLP every time you click **Train** and shows **train/test accuracy**.

## Run

1. Activate a venv that has the dependencies (`torch`, `scikit-learn`, `flask`).
   - If you’re using the venv already present in `crypto-predictor`:
     ```bash
     cd /home/dev/Desktop/training/crypto-predictor
     source venv/bin/activate
     ```

2. Start the server:
   ```bash
   cd /home/dev/Desktop/training/ANN
   python app.py
   ```

3. Open:
   - `http://localhost:5000/`

## What you can change

- Hidden layers / units
- Activation
- Weight initialization
- Batch normalization toggle
- Dropout
- Optimizer (SGD/Adam/AdamW/RMSprop)
- Learning rate
- L2 regularization (weight decay)
- Batch size / epochs

