import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import os
import warnings

# Suppress harmless warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')
warnings.filterwarnings("ignore", category=FutureWarning)


print("--- Final Model Comparison Script Started ---")

# --- Part 1: Data Preparation ---
filename = 'preprocessed_bolus_dataset.csv'
df = pd.read_csv(filename)
df_processed = pd.get_dummies(df, columns=['Sex', 'contrast_type'], drop_first=True)
X = df_processed.drop('bolus_tracking_time_sec', axis=1)
y = df_processed['bolus_tracking_time_sec']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data preparation complete.")

# --- Part 2: Feature Selection Path ---
print("\n--- [Experiment A] Models Trained on 6 Selected Features ---")
# Perform RFE to get the top 6 features
estimator = RandomForestRegressor(n_estimators=50, random_state=42)
selector = RFE(estimator, n_features_to_select=6, step=1).fit(X_train, y_train)
selected_features = X_train.columns[selector.support_]
print(f"Top 6 features selected: {list(selected_features)}")

# Create datasets with only the selected features
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]

# Scale the feature-selected data
scaler_rfe = StandardScaler()
X_train_scaled_rfe = scaler_rfe.fit_transform(X_train_rfe)
X_test_scaled_rfe = scaler_rfe.transform(X_test_rfe)

# Train RF and XGBoost on the 6 features
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled_rfe, y_train)
rf_mae = mean_absolute_error(y_test, rf_model.predict(X_test_scaled_rfe))
print(f"  - Random Forest (6 features) MAE: {rf_mae:.2f} seconds")

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train_scaled_rfe, y_train)
xgb_mae = mean_absolute_error(y_test, xgb_model.predict(X_test_scaled_rfe))
print(f"  - XGBoost (6 features) MAE: {xgb_mae:.2f} seconds")

# --- Part 3: All Features Path for ANN ---
print("\n--- [Experiment B] ANN Model Trained on ALL Features ---")
# Scale the original, full dataset
scaler_full = StandardScaler()
X_train_scaled_full = scaler_full.fit_transform(X_train)
X_test_scaled_full = scaler_full.transform(X_test)

# Train the ANN on all available features
ann_model_full = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42).fit(X_train_scaled_full, y_train)
ann_full_mae = mean_absolute_error(y_test, ann_model_full.predict(X_test_scaled_full))
print(f"  - ANN (All features) MAE: {ann_full_mae:.2f} seconds")


# --- Part 4: Final Summary ---
print("\n--- Final Results Summary ---")
print(f"Random Forest (on top 6 features): MAE = {rf_mae:.2f} seconds")
print(f"XGBoost (on top 6 features):       MAE = {xgb_mae:.2f} seconds")
print(f"ANN (on ALL features):             MAE = {ann_full_mae:.2f} seconds")
print("\n--- ✅ Script Finished! ---")