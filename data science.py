

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_erro
data = {
    "HouseID": range(1, 11),
    "Location": ["CityA", "CityB", "CityA", "CityC", "CityB", 
                 "CityA", "CityC", "CityB", "CityA", "CityC"],
    "Size": [1200, 1500, 1100, 1800, 1600, 1300, 2000, 1700, 1250, 2100],
    "Bedrooms": [2, 3, 2, 4, 3, 3, 5, 4, 2, 5],
    "Bathrooms": [1, 2, 1, 3, 2, 2, 3, 2, 1, 4],
    "YearBuilt": [2005, 2010, 2000, 2015, 2012, 2008, 2018, 2011, 2007, 2020],
    "Garage": ["Yes", "No", "No", "Yes", "Yes", "No", "Yes", "Yes", "No", "Yes"],
    "Price": [150000, 200000, 140000, 250000, 220000, 
              170000, 300000, 230000, 160000, 350000]
}

df = pd.DataFrame(data)

print(" Dataset:")
print(df)


df["Garage"] = df["Garage"].map({"Yes": 1, "No": 0})   # Convert Yes/No to 1/0
df["HouseAge"] = 2025 - df["YearBuilt"]               # Feature Engineering

df = pd.get_dummies(df, columns=["Location"], drop_first=True)

print("\n🔹 Cleaned Data:")
print(df.head())


plt.figure(figsize=(6,4))
sns.histplot(df["Price"], bins=5, kde=True)
plt.title("House Price Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=df["Size"], y=df["Price"], hue=df["Bedrooms"], palette="viridis")
plt.title("Size vs Price")
plt.show()


X = df.drop(columns=["HouseID", "Price"])  # Features
y = df["Price"]                           # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


print("\n🔹 Linear Regression R² Score:", r2_score(y_test, y_pred_lr))
print("🔹 Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

print("\n🔹 Random Forest R² Score:", r2_score(y_test, y_pred_rf))
print("🔹 Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

new_house = pd.DataFrame({
    "Size": [1550],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "YearBuilt": [2013],
    "Garage": [1],
    "HouseAge": [2025-2013],
    "Location_B": [1],  # CityB
    "Location_C": [0]   # Not CityC
})

predicted_price = rf.predict(new_house)
print("\n Predicted House Price:", predicted_price[0])
