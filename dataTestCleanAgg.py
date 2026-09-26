import pandas as pd
import numpy as np
import os

#Creating the file path for reading files and producing ouputs
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir,"input","dfsPlanML.csv")
file_path2 = os.path.join(script_dir,"input","peakSeasonDB.csv")
file_path3 = os.path.join(script_dir,"input","fuelCrisisDb.csv")
save_at = os.path.join(script_dir, "output", "detailCleanedDataForForecastAggregation.csv")
save_at2 = os.path.join(script_dir, "output", "aggForBlockHourMLForecast.csv")

#Reading the data files for extracting fields such as
#DATE, BLOCK, REG, FLT
df = pd.read_csv(file_path)

df["DATE"] = pd.to_datetime(df["DATE"],format="%d/%m/%y")

df["BLOCK"] = pd.to_timedelta(df["BLOCK"].astype(str) + ":00",errors="coerce")
df["BLOCK"] = df["BLOCK"].ffill()
df["BLOCK_DEC"] = df["BLOCK"].dt.total_seconds() / 3600
df["BLOCK_DEC"] = df["BLOCK_DEC"].round(2)
conditions = [df["REG"].isnull(),df["REG"] != df["REG"].shift(1)]
choices = [0,1]
df["AC_COUNT"] = np.select(conditions,choices,default=0)
df["AC_COUNT"] = df["AC_COUNT"].astype(int)
df["FLT_COUNT"] = np.where((df["FLT"] != df["FLT"].shift(1)) | (df["DEP"] != df["DEP"].shift(1)),1,0)
df["FLT_COUNT"] = df["FLT_COUNT"].astype(int)
df["DATE_COUNT"] = np.where(df["DATE"] != df["DATE"].shift(1),1,0)
df["DATE_COUNT"] = df["DATE_COUNT"].astype(int)

#Aggregation from raw data which has been done its calculation 
df2 = df.groupby(["DATE"]).agg(
    totalFlights = ("FLT_COUNT","sum"),
    totalBlockHour = ("BLOCK_DEC","sum"),
    acOnline = ("AC_COUNT","sum"),
    totalDay = ("DATE_COUNT","sum")
).reset_index()

df2["utilDec"] = (df2["totalBlockHour"] / df2["acOnline"]) / df2["totalDay"]
df2["utilDec"] = df2["utilDec"].round(2)
df2["avgBlockHour"] = df2["totalBlockHour"] / df2["totalFlights"]
df2["avgBlockHour"] = df2["avgBlockHour"].round(2)
df2["monthNumber"] = df2["DATE"].dt.month
df2["yearNumber"] = df2["DATE"].dt.year

#Reading the peakseason datasource 
#And extract them into a dataframe
df3 = pd.read_csv(file_path2,sep=";")
df3["DATE"] = pd.to_datetime(df3["DATE"],format="%d/%m/%Y")
df3["peakSeasonValidation"] = np.where(df3["peakSeasonValidation"].astype(str).str.lower() == "peakseason", 1, 0)
df3["peakSeasonValidation"] = df3["peakSeasonValidation"].astype(int)

#Merge the aggregated tables and the peakseason database
df4 = pd.merge(df2,df3,how="left",on=["DATE"])
df4["peakSeasonValidation"] = df4["peakSeasonValidation"].fillna(0).astype(int)

df5 = pd.read_csv(file_path3,sep=";")
df5["DATE"] = pd.to_datetime(df5["DATE"],format="%d/%m/%Y")

#Reading the datasource for Fuel Crisis Information
#Merge them into the aggregated table
df6 = pd.merge(df4,df5,how="left",on=["DATE"])
df6["fuelCrisisValidation"] = df6["fuelCrisisValidation"].fillna(0).astype(int)

conditions = [df6["yearNumber"] > 2025,
              (df6["monthNumber"] > 1 ) & (df6["yearNumber"] == 2025)]

choices = [1,1]

df6["disruptedProduction"] = np.select(conditions,choices,default=0)

#df5.info()

#Load them into csv files
df.to_csv(save_at,sep=";",index=False)
df6.to_csv(save_at2,sep=";",index=False)