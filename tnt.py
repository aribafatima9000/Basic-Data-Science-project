# ------------------------------
# E-COMMERCE CUSTOMER ANALYTICS
# ------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# ------------------------------
# STEP 1: Generate Sample Data
# ------------------------------

# Customers
customers = pd.DataFrame({
    'CustomerID': range(1, 21),
    'Name': [f'Customer_{i}' for i in range(1, 21)],
    'Age': np.random.randint(18, 60, 20),
    'Location': np.random.choice(['CityA', 'CityB', 'CityC'], 20)
})

# Transactions
products = ['Laptop', 'Phone', 'Headphones', 'Keyboard', 'Mouse', 'Monitor']
categories = ['Electronics', 'Accessories']

transactions = pd.DataFrame({
    'OrderID': range(1, 101),
    'CustomerID': np.random.choice(customers['CustomerID'], 100),
    'Product': np.random.choice(products, 100),
    'Category': np.random.choice(categories, 100),
    'Quantity': np.random.randint(1, 5, 100),
    'Price': np.random.randint(50, 1000, 100),
    'Date': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 180, 100), unit='d')
})

# ------------------------------
# STEP 2: Data Cleaning
# ------------------------------

transactions.drop_duplicates(inplace=True)
transactions['Total'] = transactions['Quantity'] * transactions['Price']

# ------------------------------
# STEP 3: Exploratory Data Analysis (EDA)
# ------------------------------

# Total revenue per customer
customer_revenue = transactions.groupby('CustomerID')['Total'].sum().sort_values(ascending=False)
print("Customer Revenue:\n", customer_revenue)

# Top selling products
top_products = transactions.groupby('Product')['Total'].sum().sort_values(ascending=False)
print("\nTop Products:\n", top_products)

# Revenue trend over time
revenue_trend = transactions.groupby('Date')['Total'].sum()
print("\nRevenue Trend:\n", revenue_trend.head())

# ------------------------------
# STEP 4: Customer Segmentation (RFM)
# ------------------------------

latest_date = transactions['Date'].max()
recency = transactions.groupby('CustomerID')['Date'].max().apply(lambda x: (latest_date - x).days)
frequency = transactions.groupby('CustomerID')['OrderID'].count()
monetary = transactions.groupby('CustomerID')['Total'].sum()

rfm = pd.DataFrame({
    'Recency': recency,
    'Frequency': frequency,
    'Monetary': monetary
}).reset_index()

print("\nRFM Table:\n", rfm.head())

# ------------------------------
# STEP 5: Visualizations
# ------------------------------

plt.figure(figsize=(8,5))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top Products by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.show()

plt.figure(figsize=(10,5))
revenue_trend.plot()
plt.title("Revenue Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()

plt.figure(figsize=(6,6))
rfm_labels = ['High', 'Medium', 'Low']
rfm['Segment'] = pd.qcut(rfm['Monetary'], q=3, labels=rfm_labels)
rfm['Segment'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Customer Segments by Monetary Value")
plt.ylabel("")
plt.show()

# ------------------------------
# STEP 6: Insights
# ------------------------------

print("\nTop Customers by Revenue:")
top_customers = rfm.sort_values('Monetary', ascending=False).head(5)
print(top_customers[['CustomerID', 'Monetary', 'Segment']])
