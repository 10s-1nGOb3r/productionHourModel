import pandas as pd
import numpy as np
import os

#Creating the file path for reading files and producing ouputs
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir,"input","dfsBlockML.csv")
file_path2 = os.path.join(script_dir,"input","peakSeasonDB.csv")
file_path3 = os.path.join(script_dir,"input","fuelCrisisDb.csv")
save_at = os.path.join(script_dir, "output", "aggPerDay.csv")
save_at2 = os.path.join(script_dir, "output", "detailCleanedData.csv")

#Reading files from AIMS. Fields being read such as
#DATE,FLT,TYPE,REG,AC,DEP,ARR,STD,STA,ATD,ATA,BLOCK
df = pd.read_csv(file_path)

#Reformating and data cleansing
#Including putting some logic toward some calculated fields
#Such as AC_COUNT, FLT_COUNT
df["DATE"] = pd.to_datetime(df["DATE"],format="%d/%m/%y")

collection2 = ["REG","FLT","DEP"]
for loop1 in collection2:
    df[loop1] = df[loop1].astype(str)

df["BLOCK"] = pd.to_timedelta(df["BLOCK"].astype(str) + ":00",errors="coerce")
df["BLOCK"] = df["BLOCK"].ffill()
df["BLOCK_DEC"] = df["BLOCK"].dt.total_seconds() / 3600
df["BLOCK_DEC"] = df["BLOCK_DEC"].round(3)
df["AC_COUNT"] = np.where(df["REG"] != df["REG"].shift(1),1,0)
df["AC_COUNT"] = df["AC_COUNT"].astype(int)
df["FLT_COUNT"] = np.where((df["FLT"] != df["FLT"].shift(1)) | (df["DEP"] != df["DEP"].shift(1)),1,0)
df["FLT_COUNT"] = df["FLT_COUNT"].astype(int)

#Aggregation process by DATE
#Some calculated fields also being made
#Such as Lag1, Lag7, disruptedProduction, Utilization
df2 = df.groupby(["DATE"]).agg(
    totalFlights = ("FLT_COUNT","sum"),
    avgBlockHour = ("BLOCK_DEC","mean"),
    totalBlockHour = ("BLOCK_DEC","sum"),
    acOnline = ("AC_COUNT","sum")
).reset_index()
df2["avgBlockHour"] = df2["avgBlockHour"].astype(float)
df2["avgBlockHour"] = df2["avgBlockHour"].round(2)
df2["dayOfWeek"] = df2["DATE"].dt.day_name()
df2["dayOfWeekNbr"] = df2["DATE"].dt.dayofweek
df2["monthName"] = df2["DATE"].dt.month_name()
df2["monthNbr"] = df2["DATE"].dt.month
df2["quarter"] = df2["DATE"].dt.quarter
df2["Year"] = df2["DATE"].dt.year
df2["Utilization"] = df2["totalBlockHour"] / df2["acOnline"]
df2["Lag1"] = df2["totalBlockHour"].shift(1).fillna(0)
df2["Lag7"] = df2["totalBlockHour"].shift(7).fillna(0)
df2["disruptedProduction"] = np.where((df2["DATE"].dt.month > 1) & (df2["DATE"].dt.year == 2025),1,0)
df2["disruptedProduction"] = df2["disruptedProduction"].astype(int)

collection = ["avgBlockHour","totalBlockHour","Utilization","Lag1","Lag7"]
for loop2 in collection:
    df2[loop2] = df2[loop2].astype(float)
    df2[loop2] = df2[loop2].round(2)

#Accessing the peakseason database throughout 2021-2025
df3 = pd.read_csv(file_path2,sep=";")
df3["DATE"] = pd.to_datetime(df3["DATE"],format="%d/%m/%Y")
df3["peakSeasonValidation"] = np.where(df3["peakSeasonValidation"].astype(str).str.lower() == "peakseason", 1, 0)
df3["peakSeasonValidation"] = df3["peakSeasonValidation"].astype(int)

#Merge the aggregated tables and the peakseason database
df4 = pd.merge(df2,df3,how="left",on=["DATE"])
df4["peakSeasonValidation"] = df4["peakSeasonValidation"].fillna(0).astype(int)

df5 = pd.read_csv(file_path3,sep=";")
df5["DATE"] = pd.to_datetime(df5["DATE"],format="%d/%m/%Y")

df5 = pd.merge(df4,df5,how="left",on=["DATE"])
df5["fuelCrisisValidation"] = df5["fuelCrisisValidation"].fillna(0).astype(int)

df.to_csv(save_at2,sep=";",index=False)
df5.to_csv(save_at,sep=";",index=False)
