import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

def get_user_input():
    """
    Prompts the user to enter patient data and returns it as a dictionary.
    Includes error handling for numeric inputs.
    """
    print("--- Please Enter New Patient Data ---")
    
    patient_data = {}
    
    # --- Robustly get numeric inputs ---
    numeric_features = {
        "Age": int,
        "height_cm": float,
        "weight_kg": float,
        "contrast_volume_ml": float,
        "flow_rate": float
    }
    
    for feature, dtype in numeric_features.items():
        while True:
            try:
                value = input(f"Enter {feature.replace('_', ' ')}: ")
                patient_data[feature] = dtype(value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    # --- Get categorical inputs ---
    patient_data['Sex'] = input("Enter Sex (e.g., Male, Female): ")
    patient_data['contrast_type'] = input("Enter Contrast Type (e.g., Iopromide 370 mgI): ")
    
    return patient_data

# --- Main Script ---
print("--- Interactive Prediction Script ---")

# --- Load the Saved Model and Components ---
try:
    model = joblib.load('final_xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
    print("Model and supporting files loaded successfully.")
except FileNotFoundError:
    print("\nERROR: Could not find model files (.joblib).")
    print("Please make sure you have run the 'save_final_model.py' script first.")
    exit()

# --- Get data from the user ---
new_patient_data = get_user_input()
df_new = pd.DataFrame([new_patient_data])

print(f"\nPredicting for new patient:\n{df_new.iloc[0].to_dict()}")

# --- The Prediction Pipeline (same as before) ---
# 1. Apply Feature Engineering
df_new['dose_per_kg'] = df_new['contrast_volume_ml'] / df_new['weight_kg']
df_new['bmi'] = df_new['weight_kg'] / ((df_new['height_cm'] / 100) ** 2)

# 2. Apply One-Hot Encoding
df_new = pd.get_dummies(df_new)

# 3. Align Columns to match the training data
df_new = df_new.reindex(columns=model_columns, fill_value=0)

# 4. Apply the loaded scaler
new_data_scaled = scaler.transform(df_new)

# 5. Make the prediction
prediction = model.predict(new_data_scaled)

# --- Display the Result ---
print("\n--- 🥁 PREDICTION 🥁 ---")
print(f"The predicted bolus tracking time is: {prediction[0]:.2f} seconds")