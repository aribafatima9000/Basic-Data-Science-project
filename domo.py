

import pandas as pd
import numpy as np


def generate_synthetic_data(n=50, seed=42):
    np.random.seed(seed)
    names = [f"Student_{i+1}" for i in range(n)]
    math = np.random.normal(loc=65, scale=12, size=n).clip(0, 100).round().astype(int)
    physics = np.random.normal(loc=60, scale=15, size=n).clip(0, 100).round().astype(int)
    chemistry = np.random.normal(loc=63, scale=10, size=n).clip(0, 100).round().astype(int)
    english = np.random.normal(loc=70, scale=8, size=n).clip(0, 100).round().astype(int)
    
    mask = np.random.choice([0, 1], size=n, p=[0.9, 0.1]).astype(bool)
    physics[mask] = np.nan
    df = pd.DataFrame({
        "Name": names,
        "Math": math,
        "Physics": physics,
        "Chemistry": chemistry,
        "English": english
    })
    return df
df = generate_synthetic_data(n=60)
print("Initial data (first 6 rows):")
print(df.head(6))


def fill_missing_with_mean(df, cols):
    df = df.copy()
    for c in cols:
        if df[c].isna().any():
            mean_val = int(round(df[c].mean(skipna=True)))
            df[c] = df[c].fillna(mean_val)
            print(f"Filled missing in {c} with mean = {mean_val}")
    return df

numeric_cols = ["Math", "Physics", "Chemistry", "English"]
df = fill_missing_with_mean(df, numeric_cols)


scores_array = df[numeric_cols].to_numpy(dtype=float)  
df["Total"] = scores_array.sum(axis=1).astype(int)
df["Percentage"] = (scores_array.mean(axis=1)).round(2)  

print("\nAfter filling missing and adding Total/Percentage:")
print(df.head(6))


def z_score_normalize(arr):
    
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)

    std_adj = np.where(std == 0, 1, std)
    z = (arr - mean) / std_adj
    return z, mean, std

arr = df[numeric_cols].to_numpy(dtype=float)
z, mean_vals, std_vals = z_score_normalize(arr)
z_df = pd.DataFrame(z, columns=[f"{c}_z" for c in numeric_cols])
df = pd.concat([df.reset_index(drop=True), z_df.reset_index(drop=True)], axis=1)

print("\nMean values for subjects:", dict(zip(numeric_cols, np.round(mean_vals,2))))
print("Std dev for subjects:", dict(zip(numeric_cols, np.round(std_vals,2))))


def assign_grade(percent_array):
    
    grades = np.full(percent_array.shape, "F", dtype=object)
    grades[np.where(percent_array >= 90)] = "A+"
    grades[np.where((percent_array >= 80) & (percent_array < 90))] = "A"
    grades[np.where((percent_array >= 70) & (percent_array < 80))] = "B"
    grades[np.where((percent_array >= 60) & (percent_array < 70))] = "C"
    grades[np.where((percent_array >= 50) & (percent_array < 60))] = "D"
    return grades

df["Grade"] = assign_grade(df["Percentage"].to_numpy())


top_5 = df.sort_values(by="Total", ascending=False).head(5)
print("\nTop 5 students by Total:")
print(top_5[["Name", "Math", "Physics", "Chemistry", "English", "Total", "Percentage", "Grade"]])


subject_stats = df[numeric_cols].agg([np.mean, np.median, np.std, np.min, np.max]).T.rename(columns={
    "mean": "Mean",
    "median": "Median",
    "std": "StdDev",
    "amin": "Min",
    "amax": "Max"
})

subject_stats = subject_stats.rename(columns={0: "Mean", 1: "Median", 2: "StdDev", 3: "Min", 4: "Max"}) \
    if subject_stats.shape[1] != 5 else subject_stats

print("\nSubject-wise stats:")
print(subject_stats)


corr_matrix = np.corrcoef(arr.T)  # 
corr_df = pd.DataFrame(corr_matrix, index=numeric_cols, columns=numeric_cols)
print("\nCorrelation matrix between subjects:")
print(corr_df.round(3))


grade_summary = df.groupby("Grade").agg(
    Count=("Name", "count"),
    Avg_Percentage=("Percentage", "mean"),
    Avg_Total=("Total", "mean")
).sort_values(by="Avg_Percentage", ascending=False)
print("\nGrade summary:")
print(grade_summary.round(2))


df.to_csv("students_processed.csv", index=False)
subject_stats.to_csv("subject_stats.csv")
corr_df.to_csv("subject_correlation.csv")
grade_summary.to_csv("grade_summary.csv")
print("\nSaved: students_processed.csv, subject_stats.csv, subject_correlation.csv, grade_summary.csv")