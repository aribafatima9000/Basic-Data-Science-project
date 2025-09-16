import pandas as pd


data = {
    "Date": ["2020-03-01", "2020-03-01", "2020-03-01", "2020-03-02", "2020-03-02", "2020-03-02"],
    "Country": ["Pakistan", "India", "USA", "Pakistan", "India", "USA"],
    "Confirmed": [2, 5, 20, 4, 12, 35],
    "Recovered": [0, 1, 5, 1, 3, 10],
    "Deaths": [0, 0, 1, 0, 1, 2]
}


df = pd.DataFrame(data)


df["Date"] = pd.to_datetime(df["Date"])

print(" COVID-19 Dataset")
print(df, "\n")


print("🔹 Total Cases per Country:")
print(df.groupby("Country")[["Confirmed", "Recovered", "Deaths"]].sum(), "\n")


highest = df.groupby("Country")["Confirmed"].sum().idxmax()
print(" Country with Highest Confirmed Cases:", highest, "\n")


print(" Daily New Cases:")
print(df.groupby("Date")["Confirmed"].sum(), "\n")


df_country = df.groupby("Country")[["Confirmed", "Recovered"]].sum()
df_country["RecoveryRate (%)"] = (df_country["Recovered"] / df_country["Confirmed"]) * 100
print(" Recovery Rate per Country:")
print(df_country[["RecoveryRate (%)"]])
