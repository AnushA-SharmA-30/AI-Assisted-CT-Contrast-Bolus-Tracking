#feature engineering RF,XGboost performed better ,ANN performed worse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
import warnings

# Suppress harmless warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore", category=FutureWarning)


print("--- Final Model Comparison with Feature Engineering ---")

# --- Part 1: Load Data ---
filename = 'preprocessed_bolus_dataset.csv'
try:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"ERROR: Could not find '{filename}'.")
    df = pd.read_csv(filename)
    print(f"Successfully loaded {len(df)} clean rows.")
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# --- Part 2: Feature Engineering ---
print("\n--- [Step 1/4] Engineering New Features... ---")
df['dose_per_kg'] = df['contrast_volume_ml'] / df['weight_kg']
height_in_meters = df['height_cm'] / 100
df['bmi'] = df['weight_kg'] / (height_in_meters ** 2)
print("Created 'dose_per_kg' and 'bmi' features.")


# --- Part 3: Model Preparation ---
df_processed = pd.get_dummies(df, columns=['Sex', 'contrast_type'], drop_first=True)
X = df_processed.drop('bolus_tracking_time_sec', axis=1)
y = df_processed['bolus_tracking_time_sec']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation complete.")


# --- Part 4: Model Training & Evaluation ---
print("\n--- [Step 2/4] Evaluating Models with New Features ---")

# Model 1: Linear Regression
lr_model = LinearRegression().fit(X_train_scaled, y_train)
lr_mae = mean_absolute_error(y_test, lr_model.predict(X_test_scaled))
print(f"  - Linear Regression MAE: {lr_mae:.3f} seconds")

# Model 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
rf_mae = mean_absolute_error(y_test, rf_model.predict(X_test_scaled))
print(f"  - Random Forest MAE: {rf_mae:.3f} seconds")

# Model 3: XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train_scaled, y_train)
xgb_mae = mean_absolute_error(y_test, xgb_model.predict(X_test_scaled))
print(f"  - XGBoost MAE: {xgb_mae:.3f} seconds")

# Model 4: Artificial Neural Network (ANN)
print(f"  - Training ANN (this may take a moment)...")
ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
ann_mae = mean_absolute_error(y_test, ann_model.predict(X_test_scaled))
print(f"  - ANN MAE: {ann_mae:.3f} seconds")


# --- Part 5: Feature Importance Analysis ---
print("\n--- [Step 3/4] Feature Importances (from Random Forest) ---")
final_feature_names = X.columns
rf_importances = rf_model.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': final_feature_names,
    'Importance': rf_importances
}).sort_values(by='Importance', ascending=False)
print(rf_importance_df.head(10))


# --- Part 6: Final Comparison ---
print("\n--- [Step 4/4] Final Performance Summary ---")
print(f"Previous Best Model (RF without new features): MAE = 2.41 seconds")
print(f"Random Forest (WITH new features):           MAE = {rf_mae:.3f} seconds")
improvement = 2.41 - rf_mae
print(f"Performance change: {improvement:+.3f} seconds")

print("\n--- ✅ Script Finished! ---")