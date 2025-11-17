import pandas as pd
import numpy as np
import re

print("--- Focused Preprocessing Script Started ---")

# --- Step 1: Load the Dataset ---
filename = 'BOLUS - Sheet1.csv'
try:
    df = pd.read_csv(filename, encoding='utf-8')
    print(f"Successfully loaded '{filename}' with {len(df)} initial rows.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# --- Step 2: Standardize and Clean Columns ---
df.columns = df.columns.str.strip()
df.rename(columns={
    'Bolus tracking time(seconds)': 'bolus_tracking_time_sec',
    'Height (cm)': 'height_cm',
    'Weight (kg)': 'weight_kg',
    'Total volume of contrast given to patient': 'contrast_volume_ml_raw',
    'Contrast type': 'contrast_type',
    'Flow rate': 'flow_rate'
}, inplace=True)

# Apply specific cleaning functions
# Clean Age by removing 'Y'/'y'
if 'Age' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace(r'[Yy]', '', regex=True), errors='coerce')

# Clean Weight by removing 's'
if 'weight_kg' in df.columns:
    df['weight_kg'] = pd.to_numeric(df['weight_kg'].astype(str).str.replace('s', '', regex=False), errors='coerce')

# Ensure Height is numeric
if 'height_cm' in df.columns:
    df['height_cm'] = pd.to_numeric(df['height_cm'], errors='coerce')
    
# Clean Flow Rate (fix typos, cap outliers)
if 'flow_rate' in df.columns:
    df['flow_rate'] = pd.to_numeric(df['flow_rate'], errors='coerce')
    df.loc[df['flow_rate'] == 35, 'flow_rate'] = 3.5
    df.loc[df['flow_rate'] > 6.0, 'flow_rate'] = 6.0

# Parse Contrast Volume
if 'contrast_volume_ml_raw' in df.columns:
    def extract_contrast_volume(text):
        if isinstance(text, str):
            match = re.search(r'(\d+\.?\d*)', text)
            if match: return float(match.group(1))
        elif isinstance(text, (int, float)): return float(text)
        return np.nan
    df['contrast_volume_ml'] = df['contrast_volume_ml_raw'].apply(extract_contrast_volume)

# Ensure Target Variable is numeric
if 'bolus_tracking_time_sec' in df.columns:
    df['bolus_tracking_time_sec'] = pd.to_numeric(df['bolus_tracking_time_sec'], errors='coerce')

print("Initial data cleaning and type conversion complete.")

# --- Step 3: Handle Missing Values Incrementally ---

# First, define columns where missing values are unacceptable
essential_cols = ['Age', 'Sex', 'height_cm', 'weight_kg', 'bolus_tracking_time_sec']
# Ensure all essential columns exist before trying to drop NaNs in them
existing_essential = [col for col in essential_cols if col in df.columns]

initial_rows = len(df)
df.dropna(subset=existing_essential, inplace=True)
print(f"Dropped {initial_rows - len(df)} rows with missing essential data (Age, Sex, Height, Weight, or Target).")

# Now, for the remaining rows, impute other columns
# Impute flow_rate with its median
if 'flow_rate' in df.columns and df['flow_rate'].isnull().sum() > 0:
    median_flow = df['flow_rate'].median()
    df['flow_rate'].fillna(median_flow, inplace=True)
    print(f"Imputed {df['flow_rate'].isnull().sum()} missing 'flow_rate' values with the median ({median_flow}).")

# Impute contrast_volume_ml with its median
if 'contrast_volume_ml' in df.columns and df['contrast_volume_ml'].isnull().sum() > 0:
    median_volume = df['contrast_volume_ml'].median()
    df['contrast_volume_ml'].fillna(median_volume, inplace=True)
    print(f"Imputed {df['contrast_volume_ml'].isnull().sum()} missing 'contrast_volume_ml' values with the median ({median_volume}).")

# --- Step 4: Finalize and Save ---

# Select the final columns for the model
final_model_columns = [
    'Age', 'Sex', 'height_cm', 'weight_kg',
    'contrast_type', 'contrast_volume_ml', 'flow_rate',
    'bolus_tracking_time_sec'
]
# Ensure all final columns exist
existing_final = [col for col in final_model_columns if col in df.columns]
df_final = df[existing_final]

# Final check to ensure data types are correct (e.g., Age to integer)
df_final['Age'] = df_final['Age'].astype(int)

# Save the preprocessed dataset to a new CSV file
output_filename = 'preprocessed_bolus_dataset.csv'
df_final.to_csv(output_filename, index=False)

print(f"\n--- ✅ Preprocessing Complete! ---")
print(f"Final dataset has {len(df_final)} clean, ready-to-use rows.")
print(f"Saved to '{output_filename}'")