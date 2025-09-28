import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from xgboost import XGBRegressor
import os

print("--- Advanced Modeling Script Started ---")

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
print("\n--- [Step 1/4] Performing Automated Feature Selection (RFE)... ---")
# We will use a simple Random Forest to help RFE decide on the best features
estimator = RandomForestRegressor(n_estimators=50, random_state=42)
# We ask RFE to select the top 6 most powerful features
selector = RFE(estimator, n_features_to_select=6, step=1)
selector = selector.fit(X_train, y_train)

# Get the names of the selected features
selected_features = X_train.columns[selector.support_]
print(f"RFE selected the following top 6 features: {list(selected_features)}")

# Create new training and testing sets with only the selected features
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]

# --- Part 3: Feature Scaling ---
# We scale AFTER feature selection
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_rfe)
X_test_scaled = scaler.transform(X_test_rfe)
print("Feature scaling applied to the selected features.")

# --- Part 4: Hyperparameter Tuning (Grid Search) ---
print("\n--- [Step 2/4] Performing Hyperparameter Tuning for Random Forest... ---")
print("(This may take a minute or two...)")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, 
                           scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

best_rf_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# --- Part 5: Final Model "Bake-Off" ---
print("\n--- [Step 3/4] Evaluating Final Models ---")

# Model 1: Tuned Random Forest
print("\n--- [Model 1/2] Tuned Random Forest ---")
rf_tuned_predictions = best_rf_model.predict(X_test_scaled)
rf_tuned_mae = mean_absolute_error(y_test, rf_tuned_predictions)
rf_tuned_rmse = np.sqrt(mean_squared_error(y_test, rf_tuned_predictions))
print(f"  - MAE: {rf_tuned_mae:.2f} seconds")
print(f"  - RMSE: {rf_tuned_rmse:.2f} seconds")

# Model 2: Artificial Neural Network (ANN)
print("\n--- [Model 2/2] Artificial Neural Network (ANN) ---")
# Using a simple but effective architecture. max_iter increased for convergence.
ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
ann_model.fit(X_train_scaled, y_train)
ann_predictions = ann_model.predict(X_test_scaled)
ann_mae = mean_absolute_error(y_test, ann_predictions)
ann_rmse = np.sqrt(mean_squared_error(y_test, ann_predictions))
print(f"  - MAE: {ann_mae:.2f} seconds")
print(f"  - RMSE: {ann_rmse:.2f} seconds")

# Model 3: XGBoost (on selected features)
print("\n--- [Model 3/3] XGBoost (on selected features) ---")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
print(f"  - MAE: {xgb_mae:.2f} seconds")
print(f"  - RMSE: {xgb_rmse:.2f} seconds")



# --- Part 6: Final Comparison ---
print("\n--- [Step 4/4] Final Results Summary ---")
print(f"Previous Best Model (Untuned RF on all features): MAE = 2.41 seconds")
print(f"Tuned Random Forest (on top 6 features):       MAE = {rf_tuned_mae:.2f} seconds")
print(f"Artificial Neural Network (on top 6 features):  MAE = {ann_mae:.2f} seconds")
print(f"XGBoost (on top 6 features):                    MAE = {xgb_mae:.2f} seconds") # Added this line
print("\n--- ✅ Advanced Modeling Script Finished! ---")