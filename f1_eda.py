# member1_eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load Algerian Forest Fire Dataset
df = pd.read_csv("algerian_forest_fires.csv", skiprows=1)
df.columns = df.columns.str.strip()
df = df[df["Classes"].notna()].copy()
df["Classes"] = df["Classes"].str.strip()
print("Shape:", df.shape)
print("\nClass distribution:\n", df["Classes"].value_counts())
print("\nMissing values:\n", df.isnull().sum())
print("\nBasic stats:\n", df.describe())
features = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]
df[features] = df[features].apply(pd.to_numeric, errors="coerce")
# Feature distribution plots
fig, axes = plt.subplots(2, 5, figsize=(18, 6))
for ax, feat in zip(axes.flat, features):
    df[feat].hist(ax=ax, bins=20, color="tomato", edgecolor="white")
    ax.set_title(feat)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("eda_distributions.png")
plt.show()
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_correlation.png")
plt.show()
