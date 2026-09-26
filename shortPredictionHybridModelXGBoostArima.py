import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from pmdarima import auto_arima

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 1. LOAD HISTORY (TRAINING DATA) ---
train_path = os.path.join(script_dir, "output", "aggPerDay.csv")
df_train = pd.read_csv(train_path, sep=";")
df_train["DATE"] = pd.to_datetime(df_train["DATE"], format="mixed", dayfirst=True)

# Feature Engineering: Weekend & Cyclic Months
df_train["is_weekend"] = np.where(df_train["dayOfWeekNbr"] >= 5, 1, 0)
df_train['month_sin'] = np.sin(2 * np.pi * df_train['monthNbr']/12)
df_train['month_cos'] = np.cos(2 * np.pi * df_train['monthNbr']/12)

# UPDATED FEATURES: Removed 'avgBlockHour' and 'monthNbr', replaced with Cyclic math
features = [
    "totalFlights", "acOnline", "dayOfWeekNbr", "is_weekend", 
    "month_sin", "month_cos", "quarter", "Year", 
    "peakSeasonValidation", "fuelCrisisValidation", "avgBlockHour"
]

X_train = df_train[features]
y_train = df_train["totalBlockHour"]

# --- 2. TRAIN HYBRID XGBOOST BASELINE ---
print("Running Expanded Tactical Grid Search...")
# Expanded the grid to let the sniper find a sharper fit
param_grid = {
    'n_estimators': [100, 200, 300], 
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6], 
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0], 
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid, scoring='neg_mean_absolute_error',
    cv=5, verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# --- 3. TRAIN ARIMA ON RESIDUALS ---
print("Initiating Hybrid Protocol (ARIMA)...")
residuals = y_train - model.predict(X_train)
# Removed hardcoded d=0, added max_d=1 so it can find trends naturally
arima_model = auto_arima(
    residuals, seasonal=True, m=7, max_d=1, max_p=2, max_q=2, 
    max_P=1, max_Q=1, max_D=1, max_order=4, trace=False, 
    suppress_warnings=True, stepwise=True
)

# --- 4. LOAD SHORT-TERM FUTURE SCHEDULE ---
test_path = os.path.join(script_dir, "output", "aggForBlockHourMLForecast.csv")
df_test = pd.read_csv(test_path, sep=";")
df_test["DATE"] = pd.to_datetime(df_test["DATE"], format="mixed", dayfirst=True)
df_test = df_test.sort_values("DATE").reset_index(drop=True)

if len(df_test) > 60:
    print(f"⚠️ Truncating input to 60 days to protect ARIMA!")
    df_test = df_test.head(60)

if "totalAcOnline" in df_test.columns:
    df_test["acOnline"] = df_test["totalAcOnline"]

# Feature Engineering for Future Data
df_test["dayOfWeekNbr"] = df_test["DATE"].dt.dayofweek
df_test["is_weekend"] = np.where(df_test["dayOfWeekNbr"] >= 5, 1, 0)
df_test["monthNbr"] = df_test["DATE"].dt.month
df_test['month_sin'] = np.sin(2 * np.pi * df_test['monthNbr']/12)
df_test['month_cos'] = np.cos(2 * np.pi * df_test['monthNbr']/12)
df_test["quarter"] = df_test["DATE"].dt.quarter
df_test["Year"] = df_test["DATE"].dt.year

X_test = df_test[features]

# --- 5. GENERATE TACTICAL FORECAST ---
print("Predicting Tactical Horizon...")
xgb_preds = model.predict(X_test)
arima_corr = arima_model.predict(n_periods=len(df_test))
df_test["Forecast"] = (xgb_preds + arima_corr.values).round(2)
df_test["Forecast_Type"] = "Tactical Hybrid"

# --- 6. OVERWRITE POWERBI MASTER FILE ---
print("Overwriting Master PowerBI File with Tactical updates...")
pbi_path = os.path.join(script_dir, "output", "blockHourPredictionPBI.csv")

df_pbi = pd.read_csv(pbi_path, sep=";")
df_pbi["date"] = pd.to_datetime(df_pbi["date"], format="mixed", dayfirst=True)

df_tactical = df_test[["DATE", "totalFlights", "acOnline", "Forecast", "Forecast_Type"]].copy()

df_pbi.set_index("date", inplace=True)
df_tactical.set_index("DATE", inplace=True)

df_pbi.update(df_tactical)

df_pbi.reset_index(inplace=True)
df_pbi["date"] = df_pbi["date"].dt.strftime('%Y-%m-%d')
df_pbi.to_csv(pbi_path, sep=";", index=False)

print(f"✅ Master PBI File successfully updated with Hybrid data!\n")