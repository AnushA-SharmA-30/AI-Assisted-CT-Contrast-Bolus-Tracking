import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for numeric selection
import os

# Set a nice style for the plots
sns.set_style("whitegrid")

# --- Step 1: Load the Data ---
filename = 'final_augmented_dataset.csv'


if not os.path.exists(filename):
    print(f"ERROR: Could not find '{filename}'. Please ensure it's in the same directory as the script.")
    exit()

try:
    df = pd.read_csv(filename)
    print("--- Successfully loaded final_augmented_dataset.csv ---")
    print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# --- Step 2: Initial Overview ---
print("\n[1] DATASET INFORMATION")
print(df.info())

print("\n[2] DESCRIPTIVE STATISTICS FOR NUMERICAL COLUMNS")
print(df.describe())

# --- Step 3: Univariate Analysis (Analyzing single variables) ---
print("\n[3] GENERATING UNIVARIATE PLOTS...")

# Histograms for numerical columns to see their distributions
df.hist(figsize=(12, 10), bins=20)
plt.tight_layout() # Adjusts plot to prevent labels from overlapping
plt.savefig('univariate_histograms.png')
plt.close() # Close the plot to free up memory
print("- Saved histograms to 'univariate_histograms.png'")

# Bar charts for categorical columns to see their balance
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='Sex', data=df, order=df['Sex'].value_counts().index)
plt.title('Distribution of Sex')

plt.subplot(1, 2, 2)
sns.countplot(x='contrast_type_category', data=df, order=df['contrast_type_category'].value_counts().index)
plt.title('Distribution of Contrast Type')

plt.tight_layout()
plt.savefig('categorical_distributions.png')
plt.close()
print("- Saved categorical distributions to 'categorical_distributions.png'")

# --- Step 4: Bivariate & Multivariate Analysis (Analyzing relationships) ---
print("\n[4] GENERATING BIVARIATE AND MULTIVARIATE PLOTS...")

# Scatter plots of key features against our target variable
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x='weight_kg', y='bolus_tracking_time_sec', data=df)
plt.title('Weight vs. Bolus Tracking Time')

plt.subplot(1, 3, 2)
sns.scatterplot(x='Age', y='bolus_tracking_time_sec', data=df)
plt.title('Age vs. Bolus Tracking Time')

plt.subplot(1, 3, 3)
sns.scatterplot(x='Flow rate', y='bolus_tracking_time_sec', data=df)
plt.title('Flow Rate vs. Bolus Tracking Time')

plt.tight_layout()
plt.savefig('scatter_plots_vs_target.png')
plt.close()
print("- Saved scatter plots to 'scatter_plots_vs_target.png'")

# Box plot to compare the target variable across categories
plt.figure(figsize=(6, 6))
sns.boxplot(x='Sex', y='bolus_tracking_time_sec', data=df)
plt.title('Bolus Tracking Time by Sex')
plt.savefig('boxplot_sex_vs_target.png')
plt.close()
print("- Saved box plot to 'boxplot_sex_vs_target.png'")

# Correlation Matrix Heatmap for a quick overview of linear relationships
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_heatmap.png')
plt.close()
print("- Saved correlation heatmap to 'correlation_heatmap.png'")

print("\n--- EDA Script Finished Successfully! ---")
print("Check your folder for the generated PNG image files.")