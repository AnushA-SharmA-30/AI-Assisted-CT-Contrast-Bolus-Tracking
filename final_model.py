import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import warnings

warnings.filterwarnings("ignore")
print("--- Final Model Training & Saving Script ---")

# --- Load and Prepare Full Dataset ---
df = pd.read_csv('preprocessed_bolus_dataset.csv')

# Apply the winning Feature Engineering
df['dose_per_kg'] = df['contrast_volume_ml'] / df['weight_kg']
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
df_processed = pd.get_dummies(df, columns=['Sex', 'contrast_type'], drop_first=True)

X = df_processed.drop('bolus_tracking_time_sec', axis=1)
y = df_processed['bolus_tracking_time_sec']
print("Data preparation complete.")

# --- Scale the Entire Dataset ---
# We fit the scaler on ALL data now because this is for the final model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data scaling complete.")

# --- Train the Champion Model on ALL Data ---
# We use the best hyperparameters we found from our experiments
print("Training the final XGBoost model on all 799 rows...")
final_xgb_model = XGBRegressor(
    n_estimators=200, 
    max_depth=7, 
    learning_rate=0.05, 
    gamma=0.2, 
    random_state=42,
    n_jobs=-1
)
final_xgb_model.fit(X_scaled, y)
print("Model training complete.")

# --- Save the Model and Supporting Components ---
# We use joblib as it's efficient for saving scikit-learn models
joblib.dump(final_xgb_model, 'final_xgb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X.columns, 'model_columns.joblib') # Save the column order
print("\n--- ✅ Model, Scaler, and Columns have been saved to files! ---")
print("- final_xgb_model.joblib")
print("- scaler.joblib")
print("- model_columns.joblib")