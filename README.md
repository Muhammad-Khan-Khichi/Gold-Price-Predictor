# ✦ GoldSense · Gold Price Predictor

A beautifully designed Streamlit application that predicts **GLD (Gold ETF) prices** using a tuned Random Forest Regressor trained on market signals — SPX, USO, SLV, and EUR/USD.

---

## Features

- **Single Prediction** — Enter market values manually and get an instant GLD price forecast
- **Batch Forecast** — Upload a CSV to run predictions on multiple rows at once, with summary stats and a download button
- **Feature Importance Bars** — Visual breakdown of how much each market signal contributes to the prediction
- **Dark gold luxury UI** — Custom-styled Streamlit interface with Cormorant Garamond + DM Mono typography

---

## 🌐 Live Demo

👉 Try the app here:
https://gold-price-predictor-mk.streamlit.app/

## Project Structure

```
.
├── gold_price_app.py        # Streamlit frontend
├── gold_price_model.pkl     # Trained Random Forest model (joblib)
├── gld_price_data.csv       # Original training dataset
├── train_model.py           # Model training script (see below)
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

### 2. Train & save the model

If you haven't saved your model yet, run this script:

```python
# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

df = pd.read_csv('gld_price_data.csv')
df.drop('Date', axis=1, inplace=True)

X = df.drop('GLD', axis=1)
y = df['GLD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=33),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Best Params : {grid_search.best_params_}")
print(f"Test R²     : {metrics.r2_score(y_test, y_pred):.4f}")
print(f"MAE         : {metrics.mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE        : {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.4f}")

joblib.dump(best_model, 'gold_price_model.pkl')
print("Model saved → gold_price_model.pkl")
```

```bash
python train_model.py
```

### 3. Launch the app

```bash
streamlit run gold_price_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dataset

The model is trained on `gld_price_data.csv` with the following columns:

| Column    | Description                        |
|-----------|------------------------------------|
| `Date`    | Trading date (dropped before training) |
| `SPX`     | S&P 500 Index price                |
| `GLD`     | Gold ETF price **(target)**        |
| `USO`     | Crude Oil ETF price                |
| `SLV`     | Silver ETF price                   |
| `EUR/USD` | Euro to US Dollar exchange rate    |

---

## Model Performance

| Metric    | Value   |
|-----------|---------|
| R² Score  | 0.9912  |
| MAE       | ~2.14   |
| RMSE      | ~3.07   |

Tuned via `GridSearchCV` with 5-fold cross-validation across 144 hyperparameter combinations.

---

## Batch Forecast CSV Format

Upload a CSV with these exact column names (no `Date` or `GLD` needed):

```csv
SPX,USO,SLV,EUR/USD
1500.0,35.0,15.0,1.10
1520.5,34.2,15.8,1.09
```

The app will append a `Predicted GLD` column and let you download the results.

---

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [scikit-learn](https://scikit-learn.org/) — Random Forest + GridSearchCV
- [joblib](https://joblib.readthedocs.io/) — Model serialization
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — Data handling

---

> **Disclaimer:** For research and educational purposes only. Not financial advice.
