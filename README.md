Flight Block Hour Forecasting PipelineAn end-to-end flight operations forecasting engine designed to predict daily block hours (totalBlockHour). The pipeline combines long-term strategic forecasting using tuned XGBoost Regressors with short-term tactical error-correction via an XGBoost + ARIMA Hybrid Model, outputting reconciled figures ready for PowerBI ingestion.   Architecture & Data FlowPlaintext[ dfsBlockML.csv ] (Historical Flights)
       │
       ▼
 1. dataTrainingCleanAgg.py ──────► output/aggPerDay.csv
                                           │
          ┌────────────────────────────────┴──────────────────────┐
          ▼                                                       ▼
 2. yearlySyntheticSchedule.py                             6. actualBlockHourForComparisson.py
          │                                                       ▲
          ▼ (output/syntheticSchedule.csv)                        │
 3. longPredictionXgBoostModel.py                                 │
          │                                                       │
          ▼ (Strategic Baseline)                                  │
    [ output/blockHourPredictionPBI.csv ]                         │
          ▲                                                       │
          │ (Overwrites Tactical Horizon ≤ 60d)                   │
 5. shortPredictionHybridModelXGBoostArima.py                     │
          ▲                                                       │
          │ (output/aggForBlockHourMLForecast.csv)                │
 4. dataTestCleanAgg.py                                           │
          ▲                                                       │
          │                                                       │
[ dfsPlanML.csv ] (Published Flight Plan)                         │
                                                                  │
                                   Reconciles Actuals vs Forecast ┘
Pipeline SequencedataTrainingCleanAgg.py (Historical Data ETL)   Ingests raw historical flight records (dfsBlockML.csv).   Aggregates metrics by calendar date: total flights, total block hours, active tail count (acOnline), and average block hours.   Enriches records with external business indicators: calendar features, production disruption markers, peak season flags (peakSeasonDB.csv), and fuel crisis events (fuelCrisisDb.csv).   Outputs: output/aggPerDay.csv.   yearlySyntheticSchedule.py (Long-Horizon Simulator)   Filters the aggregated history for baseline reference ($Year_{current} - 1$) and shifts calendar dates forward by one full year.   Simulates operational expansion by applying a 3% uplift to daily flight counts and online aircraft.   Re-applies external calendar, peak season, and fuel disruption tags.   Outputs: output/syntheticSchedule.csv.   longPredictionXgBoostModel.py (Strategic Forecasting Engine)   Trains an XGBRegressor baseline using GridSearchCV (5-fold CV) on historical features (fleet size, flight volume, cyclical sine/cosine month encodings, weekend indicators, and disruption flags).   Generates daily strategic forecasts across the full synthetic schedule.   Initializes the master PowerBI feed labeled as Strategic.   Outputs: output/blockHourPredictionPBI.csv.   dataTestCleanAgg.py (Short-Horizon Tactical ETL)   Parses the immediate published flight schedule (dfsPlanML.csv).   Computes daily aggregated planned flight counts, block hours, fleet online, and utilization ratios.   Joins peak season and fuel crisis indicators.   Outputs: output/aggForBlockHourMLForecast.csv.   shortPredictionHybridModelXGBoostArima.py (Tactical Hybrid Correction)   Tunes an XGBoost baseline on historical daily figures.   Fits an auto_arima model on the XGBoost prediction residuals ($y_{train} - \hat{y}_{XGB}$) with weekly seasonality ($m=7$) to capture high-frequency operational autocorrelation.   Evaluates short-term planned flights (capped at 60 days to prevent ARIMA variance explosion).   Overwrites matching dates in output/blockHourPredictionPBI.csv with combined predictions ($\hat{y}_{XGB} + \hat{y}_{ARIMA}$), tagging rows as Tactical Hybrid.   actualBlockHourForComparisson.py (PowerBI Actuals Ingestion)   Pulls realized block hours for the current calendar year from aggPerDay.csv.   Overlays realized actuals (totalblockhour) alongside forecasted numbers in blockHourPredictionPBI.csv.   Formats final integer numbers and dates (YYYY-MM-DD) for reporting.   Directory StructurePlaintext├── input/
│   ├── dfsBlockML.csv          # Historical raw operational flight records
│   ├── dfsPlanML.csv           # Tactical published flight schedules
│   ├── peakSeasonDB.csv        # Calendar mapping of historical/future peak periods
│   └── fuelCrisisDb.csv        # Calendar flags for fuel disruption intervals
├── output/
│   ├── aggPerDay.csv           # Cleaned daily aggregated historical data
│   ├── syntheticSchedule.csv   # Future 1-year simulated operational plan
│   ├── aggForBlockHourMLForecast.csv
│   └── blockHourPredictionPBI.csv  # Final PowerBI export (Forecast vs. Actuals)
├── actualBlockHourForComparisson.py
├── dataTestCleanAgg.py
├── dataTrainingCleanAgg.py
├── longPredictionXgBoostModel.py
├── shortPredictionHybridModelXGBoostArima.py
└── yearlySyntheticSchedule.py
Prerequisites & InstallationEnsure Python 3.10+ is installed:Bashpip install pandas numpy scikit-learn xgboost pmdarima
Execution GuideRun the pipeline sequentially from the root project directory:Bash# 1. Clean historical flight records
python dataTrainingCleanAgg.py

# 2. Build synthetic 1-year future schedule
python yearlySyntheticSchedule.py

# 3. Generate strategic long-range forecast
python longPredictionXgBoostModel.py

# 4. Clean published operational schedule
python dataTestCleanAgg.py

# 5. Overlay tactical hybrid short-range adjustments
python shortPredictionHybridModelXGBoostArima.py

# 6. Reconcile predictions against realized actuals
python actualBlockHourForComparisson.py
Output Schema (output/blockHourPredictionPBI.csv)FieldTypeDescriptiondateString (YYYY-MM-DD)   Target schedule date   totalflightsInteger   Total scheduled movements   aconlineInteger   Active operating tails for the date   forecastInteger   Predicted block hours (Strategic or Tactical Hybrid)   forecast_typeString   Source model tag (Strategic vs Tactical Hybrid)   totalblockhourInteger   Realized actual block hours (populated for elapsed dates)   
