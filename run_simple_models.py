#feature selection random forest,ann,xgboost all performed better
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
import warnings

# Suppress harmless warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore", category=FutureWarning)


print("--- Simplified Advanced Modeling Script Started ---")

# --- Part 1: Data Preparation ---
filename = 'preprocessed_bolus_dataset.csv'
try:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"ERROR: Could not find '{filename}'.")
    df = pd.read_csv(filename)
    print(f"Successfully loaded {len(df)} clean rows.")
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# One-Hot Encoding
df_processed = pd.get_dummies(df, columns=['Sex', 'contrast_type'], drop_first=True)
X = df_processed.drop('bolus_tracking_time_sec', axis=1)
y = df_processed['bolus_tracking_time_sec']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data preparation complete.")

# --- Part 2: Automated Feature Selection (RFE) ---
print("\n--- [Step 1/3] Performing Automated Feature Selection (RFE)... ---")
estimator = RandomForestRegressor(n_estimators=50, random_state=42)
selector = RFE(estimator, n_features_to_select=6, step=1)
selector = selector.fit(X_train, y_train)

selected_features = X_train.columns[selector.support_]
print(f"RFE selected the following top 6 features: {list(selected_features)}")

X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]

# --- Part 3: Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_rfe)
X_test_scaled = scaler.transform(X_test_rfe)
print("Feature scaling applied to the selected features.")

# --- Part 4: Final Model "Bake-Off" ---
print("\n--- [Step 2/3] Evaluating Final Models on Selected Features ---")

# Model 1: Random Forest
print("\n--- [Model 1/3] Random Forest ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_mae = mean_absolute_error(y_test, rf_predictions)
print(f"  - MAE: {rf_mae:.2f} seconds")

# Model 2: Artificial Neural Network (ANN)
print("\n--- [Model 2/3] Artificial Neural Network (ANN) ---")
ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
ann_model.fit(X_train_scaled, y_train)
ann_predictions = ann_model.predict(X_test_scaled)
ann_mae = mean_absolute_error(y_test, ann_predictions)
print(f"  - MAE: {ann_mae:.2f} seconds")

# Model 3: XGBoost
print("\n--- [Model 3/3] XGBoost ---")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
print(f"  - MAE: {xgb_mae:.2f} seconds")

# --- Part 5: Final Comparison ---
print("\n--- [Step 3/3] Final Results Summary ---")
print(f"Random Forest (on top 6 features): MAE = {rf_mae:.2f} seconds")
print(f"ANN (on top 6 features):           MAE = {ann_mae:.2f} seconds")
print(f"XGBoost (on top 6 features):       MAE = {xgb_mae:.2f} seconds")
print("\n--- ✅ Script Finished! ---")