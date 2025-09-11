import numpy as np
import pandas as pd

np.random.seed(42)  
marks = np.random.randint(0, 101, (100, 5)).astype(float)


marks[0, 2] = np.nan
marks[4, 1] = np.nan
marks[10, 3] = np.nan


df = pd.DataFrame(marks, columns=["Math", "English", "Science", "History", "Computer"])
print("Original Data (with NaN values):\n", df.head())

df = df.fillna(df.mean(numeric_only=True))

df["Total"] = df.sum(axis=1)
df["Average"] = df["Total"] / 5

topper = df.loc[df["Average"].idxmax()]
print("\nTopper Student:\n", topper)

failures = df[df["Average"] < 40].shape[0]
print("\nNumber of Failures:", failures)

# Step 8: Subject-wise statistics
print("\nSubject-wise Mean:\n", df[["Math","English","Science","History","Computer"]].mean())
print("\nSubject-wise Median:\n", df[["Math","English","Science","History","Computer"]].median())
print("\nSubject-wise Mode:\n", df[["Math","English","Science","History","Computer"]].mode().iloc[0])
