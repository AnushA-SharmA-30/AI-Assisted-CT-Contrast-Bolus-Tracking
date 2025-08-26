import pandas as pd

# Load the filtered Excel file
file_path = "filtered_CT_study.xlsx"
df = pd.read_excel(file_path)

# Create Chest dataset
chest_df = df[df['Study type'].str.contains("Chest", case=False, na=False)]

# Create Abdomen dataset (includes "Abdomen" and "Abdomen & Pelvis")
abdomen_df = df[df['Study type'].str.contains("Abdomen", case=False, na=False)]

# Save to separate Excel files
chest_df.to_excel("chest_CT_study.xlsx", index=False)
abdomen_df.to_excel("abdomen_CT_study.xlsx", index=False)
