import pandas as pd
import numpy as np
import os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir,"output","aggPerDay.csv")
file_path2 = os.path.join(script_dir,"output","blockHourPredictionPBI.csv")
save_at = os.path.join(script_dir,"output","blockHourPredictionPBI.csv")

# --- 1. LOAD & CLEAN HISTORY ---
df = pd.read_csv(file_path,sep=";")
df.columns = df.columns.str.lower()

timeNow = datetime.now()
currentYear = timeNow.year

df = df[df["year"] == currentYear]

collection = ["totalflights", "avgblockhour",
              "aconline", "dayofweeknbr",
              "monthname", "monthnbr",
              "quarter", "year", "dayofweek",
              "utilization", "lag1",
              "lag7", "disruptedproduction",
              "peakseasonvalidation", "fuelcrisisvalidation"]

for field in collection:
    df = df.drop(columns=field,errors="ignore")

# --- 2. LOAD & CLEAN PREDICTIONS ---
df3 = pd.read_csv(file_path2,sep=";")
df3.columns = df3.columns.str.lower()

oldMergeGarbage = ["totalblockhour_y","month", "year"]
df3 = df3.drop(columns=oldMergeGarbage, errors="ignore")

# Force columns to be real datetimes so index alignment is mathematically identical
df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
df3["date"] = pd.to_datetime(df3["date"], format="mixed", dayfirst=True)

# 💡 THE FIX: Initialize totalblockhour in df3 so .update() has a target column to fill!
df3["totalblockhour"] = np.nan

df.set_index("date", inplace=True)
df3.set_index("date", inplace=True)

# Now it overlays beautifully because the column exists in both dfs!
df3.update(df)
df3.reset_index(inplace=True)

# --- 3. SAFE FILL, ROUND, & CAST LOOP ---
collection2 = ["forecast","totalblockhour"]

for field2 in collection2:
    df3[field2] = df3[field2].fillna(0).round(0).astype(int)

# Format back to text strings right before writing file
df3["date"] = df3["date"].dt.strftime('%Y-%m-%d')

df3.to_csv(save_at,sep=";",index=False)
