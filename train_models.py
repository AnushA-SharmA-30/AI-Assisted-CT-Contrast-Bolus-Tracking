import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

print("--- Pathway 3: Model Training & Evaluation Script Started ---")

# --- Part 1: Data Preparation (Repeating the steps from our last script) ---
df = pd.read_csv('final_augmented_dataset.csv')

# Clean 'Age' and 'Flow rate'
df['Age'] = df['Age'].astype(str).str.replace(r'[Yy]', '', regex=True)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Flow rate'] = df['Flow rate'].apply(lambda x: 6.0 if x > 6.0 else x)
df.dropna(inplace=True)
df['Age'] = df['Age'].astype(int)

# One-Hot Encoding
df_processed = pd.get_dummies(df, columns=['Sex', 'contrast_type_category'], drop_first=True)

# Separate Features (X) and Target (y)
X = df_processed.drop('bolus_tracking_time_sec', axis=1)
y = df_processed['bolus_tracking_time_sec']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation complete.")

# --- Part 2: Model Training & Evaluation ---

# Model 1: Linear Regression (Our Baseline)
print("\n--- Training Model 1: Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)

# Evaluate Linear Regression
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
print(f"Linear Regression MAE: {lr_mae:.2f} seconds")
print(f"Linear Regression RMSE: {lr_rmse:.2f} seconds")


# Model 2: Random Forest Regressor (Our Advanced Model)
print("\n--- Training Model 2: Random Forest ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
print(f"Random Forest MAE: {rf_mae:.2f} seconds")
print(f"Random Forest RMSE: {rf_rmse:.2f} seconds")


# --- Part 3: Feature Importance Analysis (The "Why") ---
print("\n--- Analyzing Feature Importance from Random Forest ---")
# Get feature importances from the trained Random Forest model
importances = rf_model.feature_importances_
# Create a DataFrame to view them clearly
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)


print("\n--- ✅ Script Finished! ---")