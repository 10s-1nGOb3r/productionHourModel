# Flight Block Hour Forecasting Pipeline

A machine learning pipeline for forecasting **daily airline Block Hours** by combining **long-term strategic forecasting** with **short-term tactical adjustments**.

The pipeline processes historical flight operations, generates a synthetic future schedule for long-horizon forecasting, applies a tactical hybrid **XGBoost + ARIMA** correction to near-term schedules, and reconciles the forecast against realized actual Block Hours.

The final dataset is structured for **Power BI visualization**, enabling comparison between:

* Strategic forecast
* Tactical Hybrid forecast
* Realized actual Block Hours

---

## 1. Pipeline Overview

The pipeline follows a sequential forecasting architecture:

```text
Historical Flight Data
        │
        ▼
┌──────────────────────────────┐
│ dataTrainingCleanAgg_2.py   │
│ Historical Data Preparation  │
└──────────────┬───────────────┘
               │
               ▼
       aggPerDay.csv
               │
               ▼
┌──────────────────────────────┐
│ yearlySyntheticSchedule_2.py│
│ Future Schedule Simulation   │
└──────────────┬───────────────┘
               │
               ▼
      syntheticSchedule.csv
               │
               ▼
┌──────────────────────────────┐
│ longPredictionXgBoostModel_2 │
│ Strategic Forecasting        │
└──────────────┬───────────────┘
               │
               ▼
   blockHourPredictionPBI.csv
               ▲
               │
┌──────────────┴───────────────┐
│ dataTestCleanAgg_2.py        │
│ Tactical Schedule Preparation│
└──────────────┬───────────────┘
               │
               ▼
 aggForBlockHourMLForecast.csv
               │
               ▼
┌──────────────────────────────┐
│ shortPredictionHybrid...    │
│ XGBoost + ARIMA Correction  │
└──────────────┬───────────────┘
               │
               ▼
   Tactical Hybrid Forecast
               │
               ▼
┌──────────────────────────────┐
│ actualBlockHourForComparisson│
│ Actuals Reconciliation       │
└──────────────┬───────────────┘
               │
               ▼
      blockHourPredictionPBI.csv
               │
               ▼
             Power BI
```

---

# 2. Forecasting Strategy

The pipeline separates forecasting into two operational horizons.

### Strategic Forecast

The strategic forecast provides a **longer-horizon projection** based on historical operational patterns and a synthetic future schedule.

It uses:

* Historical Block Hours
* Flight volume
* Active aircraft
* Calendar effects
* Month seasonality
* Weekend indicators
* Lag features
* Operational utilization

The strategic model is implemented using **XGBoost regression**.

### Tactical Hybrid Forecast

The tactical forecast is designed for the **near-term operational horizon**, where an actual published flight schedule is available.

The tactical layer combines:

```text
XGBoost Forecast
       +
ARIMA Residual Forecast
       =
Hybrid Tactical Forecast
```

XGBoost captures the relationship between operational features and Block Hours, while ARIMA is applied to the XGBoost residuals to capture short-term temporal patterns that the machine-learning model does not fully explain.

The tactical forecasting horizon is capped at **60 days** to limit degradation of the ARIMA residual model over longer prediction periods.

---

# 3. Pipeline Components

## 3.1 `dataTrainingCleanAgg_2.py`

### Historical ETL

Prepares historical operational data for model training.

### Inputs

```text
input/
├── dfsBlockML.csv
├── peakSeasonDB.csv
└── fuelCrisisDb.csv
```

### Processing

The script:

1. Reads historical flight records.
2. Cleans Block Time data.
3. Aggregates operational data by calendar date.
4. Calculates:

   * Total flights
   * Total Block Hours
   * Active aircraft count
   * Utilization
5. Creates lag features:

   * 1-day lag
   * 7-day lag
6. Incorporates external business/calendar indicators.

### Main Output

```text
output/aggPerDay.csv
```

This dataset becomes the primary historical training dataset for subsequent forecasting stages.

---

# 4. `yearlySyntheticSchedule_2.py`

## Long-Horizon Schedule Simulation

Creates a synthetic future operational schedule from historical data.

The process takes the previous year's aggregated schedule and shifts calendar dates forward by exactly one year.

A **3% year-over-year growth factor** is then applied to:

* Scheduled flight volume
* Active aircraft count

External calendar indicators are also re-applied to the generated schedule.

### Main Output

```text
output/syntheticSchedule.csv
```

This dataset represents the expected operational structure used by the strategic forecasting model.

---

# 5. `longPredictionXgBoostModel_2.py`

## Strategic Forecasting Model

Trains an **XGBRegressor** using the historical aggregated dataset.

### Model Optimization

Hyperparameters are optimized using:

```text
GridSearchCV
5-fold Cross Validation
```

The search includes parameters such as:

* Learning rate
* Maximum tree depth
* Number of estimators

### Feature Engineering

The model uses operational and calendar-based features, including:

* Flight volume
* Active aircraft
* Utilization
* Lag variables
* Month
* Cyclical month encoding
* Weekend indicator
* External event indicators

Cyclical month encoding is implemented using:

```text
sin(month)
cos(month)
```

This allows the model to represent the cyclical relationship between December and January rather than treating them as distant numerical values.

### Output

The model generates the strategic Block Hour forecast and creates the initial Power BI dataset:

```text
output/blockHourPredictionPBI.csv
```

Forecast records are tagged:

```text
Strategic
```

---

# 6. `dataTestCleanAgg_2.py`

## Tactical Schedule ETL

Processes the immediate published flight schedule used for short-term forecasting.

### Input

```text
input/dfsPlanML.csv
```

### Processing

The script aggregates planned operations by date, including:

* Planned Block Hours
* Flight count
* Active aircraft
* Other operational forecasting features

### Output

```text
output/aggForBlockHourMLForecast.csv
```

This dataset provides the tactical model with the latest available operational schedule.

---

# 7. `shortPredictionHybridModelXGBoostArima_2.py`

## Tactical Hybrid Forecasting

This stage produces the short-term tactical forecast.

The model architecture is:

```text
Historical Data
       │
       ▼
    XGBoost
       │
       ▼
XGBoost Prediction
       │
       ├───────────────┐
       │               │
       ▼               ▼
   Actual Value     Prediction
       │               │
       └───────┬───────┘
               ▼
           Residual
               │
               ▼
            ARIMA
               │
               ▼
      Residual Forecast
               │
               ▼
    XGBoost + ARIMA
               │
               ▼
    Hybrid Block Hour
       Forecast
```

### XGBoost Layer

An expanded XGBoost model is trained using the historical dataset.

The model produces the baseline Block Hour forecast.

### Residual Layer

The residual is calculated as:

```text
Residual = Actual Block Hours - XGBoost Prediction
```

An `auto_arima` model is then fitted to the residual series.

This allows the pipeline to capture short-term temporal behavior that remains after the XGBoost prediction.

### Hybrid Forecast

The final tactical forecast is generated using:

```text
Hybrid Forecast =
XGBoost Forecast + ARIMA Residual Forecast
```

The tactical horizon is limited to **60 days**.

### Master Dataset Update

The tactical predictions overwrite the corresponding dates in:

```text
output/blockHourPredictionPBI.csv
```

These records are tagged:

```text
Tactical Hybrid
```

---

# 8. `actualBlockHourForComparisson_2.py`

## Actuals Reconciliation

The final stage overlays realized operational Block Hours onto the master forecasting dataset.

Actual Block Hours are retrieved from the historical aggregation and filtered for the current calendar year.

The script then:

1. Loads the master forecast dataset.
2. Retrieves realized Block Hours.
3. Matches records by calendar date.
4. Adds the actual Block Hour value.
5. Standardizes date formatting.
6. Handles missing values.
7. Preserves date/index alignment.

The final dataset is:

```text
output/blockHourPredictionPBI.csv
```

This becomes the primary dataset consumed by Power BI.

---

# 9. Directory Structure

```text
.
├── input/
│   ├── dfsBlockML.csv
│   ├── dfsPlanML.csv
│   ├── peakSeasonDB.csv
│   └── fuelCrisisDb.csv
│
├── output/
│   ├── aggPerDay.csv
│   ├── detailCleanedData.csv
│   ├── syntheticSchedule.csv
│   ├── detailCleanedDataForForecastAggregation.csv
│   ├── aggForBlockHourMLForecast.csv
│   └── blockHourPredictionPBI.csv
│
├── actualBlockHourForComparisson_2.py
├── dataTestCleanAgg_2.py
├── dataTrainingCleanAgg_2.py
├── longPredictionXgBoostModel_2.py
├── shortPredictionHybridModelXGBoostArima_2.py
├── yearlySyntheticSchedule_2.py
└── README.md
```

---

# 10. Requirements

The pipeline requires Python 3.x and the following packages:

```text
pandas
numpy
scikit-learn
xgboost
pmdarima
```

Install the dependencies using:

```bash
pip install pandas numpy scikit-learn xgboost pmdarima
```

---

# 11. Execution

The scripts must be executed **sequentially** because each stage generates data required by subsequent stages.

Run:

```bash
python dataTrainingCleanAgg_2.py
```

Then:

```bash
python yearlySyntheticSchedule_2.py
```

Then:

```bash
python longPredictionXgBoostModel_2.py
```

Then:

```bash
python dataTestCleanAgg_2.py
```

Then:

```bash
python shortPredictionHybridModelXGBoostArima_2.py
```

Finally:

```bash
python actualBlockHourForComparisson_2.py
```

### Complete Execution

```bash
python dataTrainingCleanAgg_2.py
python yearlySyntheticSchedule_2.py
python longPredictionXgBoostModel_2.py
python dataTestCleanAgg_2.py
python shortPredictionHybridModelXGBoostArima_2.py
python actualBlockHourForComparisson_2.py
```

---

# 12. Data Dependency Flow

The primary data dependencies are:

```text
dfsBlockML.csv
      │
      ▼
dataTrainingCleanAgg_2.py
      │
      ▼
aggPerDay.csv
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
yearlySyntheticSchedule   XGBoost Training
      │                      │
      ▼                      │
syntheticSchedule.csv        │
      │                      │
      └──────────┬───────────┘
                 ▼
       Strategic Forecast
                 │
                 ▼
    blockHourPredictionPBI.csv
                 ▲
                 │
dfsPlanML.csv    │
      │          │
      ▼          │
dataTestCleanAgg │
      │          │
      ▼          │
aggForBlockHourML│
      │          │
      ▼          │
XGBoost + ARIMA  │
      │          │
      └──────────┘
                 │
                 ▼
       Actual Reconciliation
                 │
                 ▼
    blockHourPredictionPBI.csv
                 │
                 ▼
              Power BI
```

---

# 13. Forecast Output

The final `blockHourPredictionPBI.csv` is designed as the reporting layer for Power BI.

Conceptually, the dataset contains:

| Date       | Forecast Type   | Forecast Block Hour | Actual Block Hour |
| ---------- | --------------- | ------------------: | ----------------: |
| YYYY-MM-DD | Strategic       |                 ... |               ... |
| YYYY-MM-DD | Tactical Hybrid |                 ... |               ... |
| YYYY-MM-DD | Actual          |                 ... |               ... |

The exact columns depend on the current implementation of the individual scripts.

The `Forecast Type` field allows Power BI to distinguish between the long-horizon strategic forecast and the short-horizon tactical correction.

---

# 14. Forecasting Concept

The overall forecasting approach can be summarized as:

```text
                    HISTORICAL OPERATIONS
                             │
                             ▼
                    Historical Aggregation
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
       Long-Term Projection       Historical Model Training
                │                         │
                ▼                         ▼
       Synthetic Schedule             XGBoost
                │                         │
                └────────────┬────────────┘
                             ▼
                     Strategic Forecast
                             │
                             ▼
                 Published Flight Schedule
                             │
                             ▼
                    Tactical XGBoost
                             │
                             ▼
                     Model Residuals
                             │
                             ▼
                           ARIMA
                             │
                             ▼
                    Tactical Correction
                             │
                             ▼
                     Final Forecast
                             │
                             ▼
                  Actuals Reconciliation
                             │
                             ▼
                         Power BI
```

---

# 15. Key Design Principles

### Strategic + Tactical Forecasting

Instead of relying on a single forecasting model across the entire planning horizon, the pipeline separates:

* **Strategic forecasting** for longer-term planning
* **Tactical forecasting** for near-term operational adjustments

This allows the forecast to incorporate increasingly reliable schedule information as the operating date approaches.

### Machine Learning + Time Series

The pipeline combines two modeling approaches:

**XGBoost**

Used to model nonlinear relationships between operational features and Block Hours.

**ARIMA**

Used to model temporal behavior remaining in the XGBoost residuals.

The resulting hybrid approach is:

```text
Operational Features
        ↓
     XGBoost
        ↓
Baseline Forecast
        +
Temporal Residual Pattern
        ↓
      ARIMA
        ↓
Tactical Correction
```

### Power BI-Oriented Output

Rather than producing isolated model predictions, the pipeline maintains a consolidated master dataset:

```text
blockHourPredictionPBI.csv
```

This allows the forecasting process to feed directly into a business intelligence layer for operational monitoring and forecast-vs-actual analysis.

---

# 16. Important Execution Notes

The scripts currently operate as a **sequential pipeline**.

Therefore:

* Run scripts in the defined order.
* Do not delete intermediate output files before downstream processes finish.
* Ensure the required input CSV files are available in `input/`.
* Ensure the output directory exists before execution.
* Confirm that date fields are correctly formatted.
* The tactical model is intentionally restricted to a maximum 60-day horizon.
* The synthetic schedule applies a fixed 3% year-over-year growth assumption.

The 3% growth factor is a business assumption and should be treated as a configurable forecasting parameter if the pipeline is adapted for production use.

---

# 17. Potential Production Improvements

For production deployment, the pipeline can be further developed by introducing:

* Centralized configuration using `config.yaml` or `.env`
* Parameterized forecast horizons
* Configurable growth assumptions
* Structured logging
* Error handling and validation
* Data-quality checks
* Model performance monitoring
* Automated model retraining
* Model versioning
* Unit and integration tests
* Database/API input instead of CSV-only ingestion
* Automated Power BI dataset refresh
* Scheduled execution through an orchestration platform

A future production architecture could therefore evolve from:

```text
Python Scripts
     ↓
CSV Files
     ↓
Power BI
```

into:

```text
Operational Data Sources
          ↓
      ETL Pipeline
          ↓
   Data Validation
          ↓
 Forecasting Engine
 ┌────────┴─────────┐
 │                  │
XGBoost           ARIMA
 │                  │
 └────────┬─────────┘
          ↓
 Forecast Repository
          ↓
      Power BI
          ↓
Operational Planning
```

---

# 18. Project Objective

The objective of this pipeline is not simply to predict Block Hours.

It is designed to support **operational planning by progressively improving the forecast as more reliable schedule information becomes available**.

The forecasting logic follows the principle:

```text
Long-Term Planning
       ↓
Strategic Forecast
       ↓
Schedule Becomes Available
       ↓
Tactical Adjustment
       ↓
Actual Operations
       ↓
Forecast vs Actual Analysis
```

This creates a continuous forecasting framework connecting **strategic planning, tactical scheduling, and operational performance monitoring**.
