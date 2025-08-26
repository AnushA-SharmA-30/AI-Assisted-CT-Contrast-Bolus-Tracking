import pandas as pd

# Load the filtered Excel file
file_path = "filtered_CT_study.xlsx"
df = pd.read_excel(file_path)

# Create Chest dataset with exact match
chest_df = df[df['Study type'].str.strip().str.lower() == "chest"]

# Save to a new Excel file
chest_df.to_excel("chest_only_CT_study.xlsx", index=False)
