import pandas as pd
import os

print("--- Data Augmentation Script Started ---")

try:
    # Load both the clean and incomplete datasets
    df_clean = pd.read_csv('final_cleaned_data.csv')
    df_incomplete = pd.read_csv('incomplete_data_for_review.csv')

    # --- Step 1: Calculate the Median from the Clean Data ---
    # We will use the median of 'Flow rate' as it is robust to outliers.
    median_flow_rate = df_clean['Flow rate'].median()
    print(f"Calculated Median Flow Rate from clean data: {median_flow_rate:.2f}")

    # --- Step 2: Identify Rows to Rescue ---
    # We will only rescue rows from the incomplete set that are missing EXACTLY 1 value.
    rows_missing_one_value = df_incomplete[df_incomplete.isnull().sum(axis=1) == 1]
    
    # Of those, we only want the ones where the missing value is indeed the 'Flow rate'.
    rows_to_rescue = rows_missing_one_value[rows_missing_one_value['Flow rate'].isnull()]
    print(f"Identified {len(rows_to_rescue)} rows that can be safely rescued.")

    # --- Step 3: Perform the Imputation ---
    # Create a copy to avoid warnings
    df_rescued = rows_to_rescue.copy()
    df_rescued['Flow rate'].fillna(median_flow_rate, inplace=True)
    print("Filled missing 'Flow rate' values with the median.")

    # --- Step 4: Combine and Save the Final Dataset ---
    # Concatenate the original clean dataframe with the newly rescued one.
    df_final_augmented = pd.concat([df_clean, df_rescued], ignore_index=True)

    output_filename = 'final_augmented_dataset.csv'
    df_final_augmented.to_csv(output_filename, index=False)

    print("\n--- ✅ SCRIPT FINISHED SUCCESSFULLY! ---")
    print(f"Original clean rows: {len(df_clean)}")
    print(f"Rescued rows: {len(df_rescued)}")
    print(f"New total usable rows: {len(df_final_augmented)}")
    print(f"Final augmented data has been saved to '{output_filename}'")
    
except FileNotFoundError as e:
    print(f"ERROR: Make sure both 'final_cleaned_data.csv' and 'incomplete_data_for_review.csv' are in the directory. Details: {e}")
except Exception as e:
    print(f"An error occurred: {e}")