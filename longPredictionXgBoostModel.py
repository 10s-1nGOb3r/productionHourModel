import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 1. LOAD HISTORY (TRAINING DATA) ---
train_path = os.path.join(script_dir, "output", "aggPerDay.csv")
df_train = pd.read_csv(train_path, sep=";")
df_train["DATE"] = pd.to_datetime(df_train["DATE"], format="mixed", dayfirst=True)

df_train["is_weekend"] = np.where(df_train["dayOfWeekNbr"] >= 5, 1, 0)
df_train["month_sin"] = np.sin(2 * np.pi * df_train["monthNbr"] / 12.0)
df_train["month_cos"] = np.cos(2 * np.pi * df_train["monthNbr"] / 12.0)

features = [
    "totalFlights", "acOnline", "dayOfWeekNbr", "is_weekend", 
    "month_sin", "month_cos", "Year", "disruptedProduction", 
    "peakSeasonValidation", "fuelCrisisValidation"
]

X_train = df_train[features]
y_train = df_train["totalBlockHour"]

# --- 2. TRAIN THE STRATEGIC XGBOOST SNIPER ---
print("Running Strategic Grid Search for Long-Term Forecasting...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
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

# --- 3. LOAD SYNTHETIC SCHEDULE ---
print("\nLoading future schedule from syntheticSchedule.csv...")
test_path = os.path.join(script_dir, "output", "syntheticSchedule.csv")
df_test = pd.read_csv(test_path, sep=";")
df_test["DATE"] = pd.to_datetime(df_test["DATE"], format="mixed", dayfirst=True)

if "acOnline" in df_test.columns:
    df_test["acOnline"] = df_test["acOnline"]

df_test["dayOfWeekNbr"] = df_test["DATE"].dt.dayofweek
df_test["is_weekend"] = np.where(df_test["dayOfWeekNbr"] >= 5, 1, 0)
df_test["monthNbr"] = df_test["DATE"].dt.month
df_test["month_sin"] = np.sin(2 * np.pi * df_test["monthNbr"] / 12.0)
df_test["month_cos"] = np.cos(2 * np.pi * df_test["monthNbr"] / 12.0)
df_test["Year"] = df_test["DATE"].dt.year

X_test = df_test[features]

# --- 4. STRATEGIC PREDICTION & POWERBI EXPORT ---
print("\nGenerating Long-Range Strategy Forecast...")
df_test["Forecast"] = model.predict(X_test).round(2)
df_test["Forecast_Type"] = "Strategic" # Tags it for PowerBI

# Save the Master PBI File
csv_path = os.path.join(script_dir, "output", "blockHourPredictionPBI.csv")
# Formatting DATE cleanly for PowerBI recognition
df_test["DATE"] = df_test["DATE"].dt.strftime('%Y-%m-%d')
df_test[["DATE", "totalFlights", "acOnline", "Forecast", "Forecast_Type"]].to_csv(csv_path, sep=";", index=False)
print(f"✅ PowerBI Master File created at: {csv_path}\n")