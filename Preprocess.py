import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load datasets
male_df = pd.read_csv("Cleaned_Male_CT_Data.csv")
female_df = pd.read_csv("Cleaned_Female_CT_Data.csv")

# Columns to use
features = [
    "Age", "Gender", "Height (cm)", "Weight (kg)", "HU", "Pitch", 
    "Rotation time", "Total volume of contrast given to patient", "Flow rate"
]
target = "Bolus tracking time(seconds)"

def preprocess(df):
    # Extract numerical pitch (e.g., "0.984:1" → 0.984)
    df["Pitch"] = df["Pitch"].str.extract(r"([\d.]+)").astype(float)
    
    # Encode Gender
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"].str.lower())
    
    # Select features + target
    df = df[features + [target]].dropna()
    
    # Scale features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df

# Preprocess separately
male_processed = preprocess(male_df)
female_processed = preprocess(female_df)
# ✅ Save processed versions
male_processed.to_csv("Processed_Male_CT_Data.csv", index=False)
female_processed.to_csv("Processed_Female_CT_Data.csv", index=False)
# Preview result
male_processed.head(), female_processed.head()
