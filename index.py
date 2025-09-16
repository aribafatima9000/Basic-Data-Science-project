import pandas as pd


data = {
    "OrderID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Customer": ["Ali", "Sara", "Bilal", "Ayesha", "Ali", "Zain", "Sara", "Omar"],
    "Product": ["Laptop", "Mobile", "Laptop", "Tablet", "Headphones", "Mobile", "Laptop", "Tablet"],
    "Quantity": [1, 2, 1, 1, 3, 2, 1, 2],
    "Price": [800, 300, 800, 200, 50, 300, 800, 200],
    "Date": [
        "2022-01-10", "2022-01-11", "2022-01-11", "2022-01-12",
        "2022-01-13", "2022-01-13", "2022-01-14", "2022-01-15"
    ]
}


df = pd.DataFrame(data)


df["Date"] = pd.to_datetime(df["Date"])


df["Total"] = df["Quantity"] * df["Price"]

print(" E-commerce Dataset")
print(df, "\n")


print(" Total Sales per Product:")
print(df.groupby("Product")["Total"].sum(), "\n")


print(" Top 3 Customers by Spending:")
print(df.groupby("Customer")["Total"].sum().nlargest(3), "\n")


best_seller = df.groupby("Product")["Quantity"].sum().idxmax()
print(" Best-Selling Product:", best_seller, "\n")


print(" Daily Sales Trend:")
print(df.groupby("Date")["Total"].sum(), "\n")


avg_order = df.groupby("Customer")["Total"].mean()
print(" Average Order Value per Customer:")
print(avg_order)
