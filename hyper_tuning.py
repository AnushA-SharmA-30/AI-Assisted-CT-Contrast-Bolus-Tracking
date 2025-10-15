#hypertuning for random forest,xgboost(performed better) along with feature engineering
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import os
import warnings

# Suppress harmless warnings
warnings.filterwarnings("ignore")

print("--- Advanced Hyperparameter Tuning Script ---")

# --- Part 1: Load and Prepare Data with Engineered Features ---
filename = 'preprocessed_bolus_dataset.csv'
df = pd.read_csv(filename)

df['dose_per_kg'] = df['contrast_volume_ml'] / df['weight_kg']
height_in_meters = df['height_cm'] / 100
df['bmi'] = df['weight_kg'] / (height_in_meters ** 2)
df_processed = pd.get_dummies(df, columns=['Sex', 'contrast_type'], drop_first=True)

X = df_processed.drop('bolus_tracking_time_sec', axis=1)
y = df_processed['bolus_tracking_time_sec']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Loaded and prepared {len(df)} rows with engineered features.")

# --- Part 2: Randomized Search for Random Forest ---
print("\n--- [Step 1/3] Tuning Random Forest ---")
# Define the search space for Random Forest hyperparameters
param_dist_rf = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestRegressor(random_state=42)
# n_iter sets how many random combinations to try. cv is for cross-validation.
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=20, cv=5, 
                                      scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
random_search_rf.fit(X_train_scaled, y_train)
best_rf = random_search_rf.best_estimator_
print(f"Best RF Parameters found: {random_search_rf.best_params_}")
rf_mae_tuned = mean_absolute_error(y_test, best_rf.predict(X_test_scaled))

# --- Part 3: Randomized Search for XGBoost ---
print("\n--- [Step 2/3] Tuning XGBoost ---")
# Define the search space for XGBoost hyperparameters
param_dist_xgb = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'gamma': [0, 0.1, 0.2]
}
xgb = XGBRegressor(random_state=42)
random_search_xgb = RandomizedSearchCV(xgb, param_distributions=param_dist_xgb, n_iter=20, cv=5, 
                                       scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1)
random_search_xgb.fit(X_train_scaled, y_train)
best_xgb = random_search_xgb.best_estimator_
print(f"Best XGBoost Parameters found: {random_search_xgb.best_params_}")
xgb_mae_tuned = mean_absolute_error(y_test, best_xgb.predict(X_test_scaled))


# --- Part 4: Final Results ---
print("\n--- [Step 3/3] Tuning Performance Summary ---")
print("\n-- Random Forest --")
print(f"  - Performance BEFORE tuning (MAE): 2.375 seconds")
print(f"  - Performance AFTER tuning (MAE):  {rf_mae_tuned:.3f} seconds")
rf_improvement = 2.375 - rf_mae_tuned
print(f"  - Improvement: {rf_improvement:+.3f} seconds")

print("\n-- XGBoost --")
print(f"  - Performance BEFORE tuning (MAE): 2.328 seconds")
print(f"  - Performance AFTER tuning (MAE):  {xgb_mae_tuned:.3f} seconds")
xgb_improvement = 2.328 - xgb_mae_tuned
print(f"  - Improvement: {xgb_improvement:+.3f} seconds")

print("\n--- ✅ Script Finished! ---")