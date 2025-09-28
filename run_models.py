import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
import os

# Suppress harmless warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

print("--- Final Model Training & Comparison Script ---")

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

# --- Part 2: Model Preparation ---

# One-Hot Encoding for categorical features
# We create a new category from the raw text first for robustness
df['contrast_type_category'] = df['contrast_type'].apply(
    lambda x: 'Iopromide' if 'iopromide' in str(x).lower()
    else 'Iohexol' if 'iohexol' in str(x).lower()
    else 'Other'
)
df_processed = pd.get_dummies(df, columns=['Sex', 'contrast_type_category'], drop_first=True)
# We drop the original 'contrast_type' as it's now encoded
df_processed.drop('contrast_type', axis=1, inplace=True)

# Separate Features (X) and Target (y)
X = df_processed.drop('bolus_tracking_time_sec', axis=1)
y = df_processed['bolus_tracking_time_sec']

# Train-Test Split (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation complete (Encoding, Splitting, Scaling).")

# --- Part 3: Model Training & Evaluation ---

# Model 1: Linear Regression (Baseline)
print("\n--- [Model 1/3] Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
print(f"  - MAE: {lr_mae:.2f} seconds")
print(f"  - RMSE: {lr_rmse:.2f} seconds")

# Model 2: Random Forest (Bagging)
print("\n--- [Model 2/3] Random Forest ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
print(f"  - MAE: {rf_mae:.2f} seconds")
print(f"  - RMSE: {rf_rmse:.2f} seconds")

# Model 3: XGBoost (Boosting)
print("\n--- [Model 3/3] XGBoost ---")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
print(f"  - MAE: {xgb_mae:.2f} seconds")
print(f"  - RMSE: {xgb_rmse:.2f} seconds")

# --- Part 4: Feature Importance Analysis ---
print("\n--- Feature Importances from Advanced Models ---")

# Get importances from Random Forest
rf_importances = rf_model.feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Random Forest Importance': rf_importances
}).sort_values(by='Random Forest Importance', ascending=False)

# Get importances from XGBoost
xgb_importances = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'XGBoost Importance': xgb_importances
}).sort_values(by='XGBoost Importance', ascending=False)

print("\n-- Random Forest --")
print(rf_importance_df)

print("\n-- XGBoost --")
print(xgb_importance_df)

print("\n--- ✅ Script Finished! ---")