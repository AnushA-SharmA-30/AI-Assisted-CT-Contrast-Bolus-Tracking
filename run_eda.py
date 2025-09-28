import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set a professional style for the plots
sns.set_style("whitegrid")

# --- Step 1: Load and Verify the Preprocessed Data ---
filename = 'preprocessed_bolus_dataset.csv'
try:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"ERROR: Could not find '{filename}'. Please ensure it's in the same directory.")
    df = pd.read_csv(filename)
    print(f"--- Successfully loaded '{filename}' ---")
    print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# --- Verification Step: Check if the file is accurate ---
print("\n[1] VERIFYING DATASET ACCURACY (INFO)")
print(df.info())

print("\n[2] VERIFYING DATASET ACCURACY (STATISTICS)")
print(df.describe())

# --- Step 3: Univariate Analysis ---
print("\n[3] GENERATING UNIVARIATE PLOTS...")
df.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Features', size=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('univariate_histograms.png')
plt.close()
print("- Saved histograms to 'univariate_histograms.png'")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='Sex', data=df, order=df['Sex'].value_counts().index)
plt.title('Distribution of Sex')
plt.subplot(1, 2, 2)
# Create a new categorical type from the raw text for better plotting
df['contrast_type_category'] = df['contrast_type'].apply(
    lambda x: 'Iopromide' if 'iopromide' in str(x).lower()
    else 'Iohexol' if 'iohexol' in str(x).lower()
    else 'Other'
)
sns.countplot(x='contrast_type_category', data=df, order=df['contrast_type_category'].value_counts().index)
plt.title('Distribution of Contrast Type')
plt.tight_layout()
plt.savefig('categorical_distributions.png')
plt.close()
print("- Saved categorical distributions to 'categorical_distributions.png'")

# --- Step 4: Bivariate & Multivariate Analysis ---
print("\n[4] GENERATING BIVARIATE AND MULTIVARIATE PLOTS...")

plt.figure(figsize=(18, 5))
features_to_plot = ['weight_kg', 'Age', 'flow_rate']
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=feature, y='bolus_tracking_time_sec', data=df, alpha=0.6)
    plt.title(f'{feature.replace("_", " ").title()} vs. Bolus Tracking Time')
plt.tight_layout()
plt.savefig('scatter_plots_vs_target.png')
plt.close()
print("- Saved scatter plots to 'scatter_plots_vs_target.png'")

plt.figure(figsize=(7, 7))
sns.boxplot(x='Sex', y='bolus_tracking_time_sec', data=df)
plt.title('Bolus Tracking Time by Sex')
plt.savefig('boxplot_sex_vs_target.png')
plt.close()
print("- Saved box plot to 'boxplot_sex_vs_target.png'")

plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_heatmap.png')
plt.close()
print("- Saved correlation heatmap to 'correlation_heatmap.png'")

print("\n--- EDA Script Finished Successfully! ---")
print("Check your folder for the generated PNG image files.")