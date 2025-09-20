import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("sales_data.csv")

# 2. Data Cleaning
df.dropna(inplace=True)   # remove missing values
df = df[df['Order Date'].str[0:2] != 'Or']  # remove invalid rows
df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'])
df['Price Each'] = pd.to_numeric(df['Price Each'])

# 3. Add new column: Sales
df['Sales'] = df['Quantity Ordered'] * df['Price Each']

# 4. Convert Order Date into datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.month
df['Hour'] = df['Order Date'].dt.hour

# 5. Best month for sales
monthly_sales = df.groupby('Month')['Sales'].sum()
print("Monthly Sales:\n", monthly_sales)

monthly_sales.plot(kind='bar', title="Monthly Sales")
plt.show()

# 6. Sales by City
df['City'] = df['Purchase Address'].apply(lambda x: x.split(",")[1].strip())
city_sales = df.groupby('City')['Sales'].sum()
city_sales.plot(kind='bar', title="Sales by City")
plt.show()

# 7. Most sold products
product_group = df.groupby('Product').sum()
product_group['Quantity Ordered'].plot(kind='bar', title="Most Sold Products")
plt.show()

# 8. Time-based sales trend
df.groupby('Hour')['Order ID'].count().plot(kind='line', title="Sales by Hour")
plt.xticks(range(0,24))
plt.xlabel("Hour of Day")
plt.ylabel("Number of Orders")
plt.show()
