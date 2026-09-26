import pandas as pd
import numpy as np
import os
from datetime import datetime

#Creating the file path for reading files and producing ouputs
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir,"output","aggPerDay.csv")
file_path2 = os.path.join(script_dir,"input","peakSeasonDB.csv")
file_path3 = os.path.join(script_dir,"input","fuelCrisisDb.csv")
save_at = os.path.join(script_dir, "output", "syntheticSchedule.csv")

#Reading the file from the training files
df = pd.read_csv(file_path,sep=";")

#Do a selection for the data that will be synthetized
#Into a data test for One Year Prediction
yearNow = datetime.now().year
yearMinusOne = yearNow - 1

conditions = [(df["Year"] > yearMinusOne),
              (df["Year"] < yearMinusOne)]

choices = [1,1]

df["rowsToBeDropped"] = np.select(conditions,choices,default=0)

#Cleaning out the dataset
mask = (df["rowsToBeDropped"] == 1)
df = df.drop(df[mask].index)

df = df.drop(columns=["rowsToBeDropped"])

#Composing the dataset 
#For data testing of 1 Year Model Prediction
df["Year"] = df["Year"] + 1

df["DATE"] = pd.to_datetime(df["DATE"], format="%d/%m/%Y", errors="coerce")
df["DATE"] = df["DATE"] + pd.DateOffset(years=1)

collection = ["totalFlights","acOnline"]
for field in collection:
    df[field] = df[field] + (df[field] * 0.03)
    df[field] = df[field].round(0)
    df[field] = df[field].astype(int)

df["dayOfWeek"] = df["DATE"].dt.day_name()
df["dayOfWeekNbr"] = df["DATE"].dt.dayofweek

df["totalBlockHour"] = np.nan

df = df.drop(columns=["Lag1", "Lag7","Utilization","disruptedProduction","peakSeasonValidation","fuelCrisisValidation"])

df["disruptedProduction"] = np.where((df["DATE"].dt.month > 1) & (df["DATE"].dt.year == 2025),1,0)
df["disruptedProduction"] = df["disruptedProduction"].astype(int)

#Reading and Merging tables for Peak Season datasets
df2 = pd.read_csv(file_path2,sep=";")
df2["DATE"] = pd.to_datetime(df2["DATE"],format="%d/%m/%Y")
df3 = pd.merge(df,df2,how="left",on=["DATE"])
df3["peakSeasonValidation"] = np.where(df3["peakSeasonValidation"].astype(str).str.lower() == "peakseason", 1, 0)
df3["peakSeasonValidation"] = df3["peakSeasonValidation"].fillna(0).astype(int)

#Reading and Merging tables for Peak Season datasets
df4 = pd.read_csv(file_path3,sep=";")
df4["DATE"] = pd.to_datetime(df4["DATE"],format="%d/%m/%Y")
df5 = pd.merge(df3,df4,how="left",on=["DATE"])
df5["fuelCrisisValidation"] = df5["fuelCrisisValidation"].fillna(0).astype(int)

#Export them into csv files
df5.to_csv(save_at,sep=";",index=False)

