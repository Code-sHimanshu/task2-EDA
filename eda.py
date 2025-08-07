
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("dataset/cleaned_titanic.csv")

# Create folder for screenshots
os.makedirs("screenshots", exist_ok=True)

# Summary statistics
print("Summary Statistics:")
print(df.describe(include='all'))

# Histograms for numeric features
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Histogram of {col}")
    plt.savefig(f"screenshots/histogram_{col}.png")
    plt.close()

# Boxplots for numeric features
for col in numeric_cols:
    plt.figure()
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"screenshots/boxplot_{col}.png")
    plt.close()

# Correlation matrix and heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.savefig("screenshots/correlation_heatmap.png")
plt.close()

# Pairplot for selected features
selected_features = ['Age', 'Fare', 'Pclass', 'Survived']
sns.pairplot(df[selected_features].dropna(), hue='Survived')
plt.savefig("screenshots/pairplot_selected_features.png")
plt.close()

print("EDA completed and visualizations saved in 'screenshots/' folder.")
