import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Pathway 2: Model Preparation Script Started ---")

# --- Step 1: Load and Fix Data ---
df = pd.read_csv('final_augmented_dataset.csv')
print(f"Loaded {len(df)} rows from the dataset.")

# FIX 1: Clean the 'Age' column
df['Age'] = df['Age'].astype(str).str.replace(r'[Yy]', '', regex=True)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# FIX 2: Cap the 'Flow rate' outlier
df['Flow rate'] = df['Flow rate'].apply(lambda x: 6.0 if x > 6.0 else x)

# Drop any rows that became null after cleaning
df.dropna(inplace=True)
df['Age'] = df['Age'].astype(int)
print(f"Data cleaned. Proceeding with {len(df)} high-quality rows.")


# --- Step 2: Feature Engineering (One-Hot Encoding) ---
# Convert categorical columns into numeric format
df_processed = pd.get_dummies(df, columns=['Sex', 'contrast_type_category'], drop_first=True)
print("Applied one-hot encoding to categorical features.")
# Note: drop_first=True is used to avoid multicollinearity, which is good practice.


# --- Step 3: Separate Features (X) and Target (y) ---
# 'X' is our set of features (the inputs to the model)
X = df_processed.drop('bolus_tracking_time_sec', axis=1)

# 'y' is our target variable (what we want to predict)
y = df_processed['bolus_tracking_time_sec']


# --- Step 4: Train-Test Split ---
# Split the data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")


# --- Step 5: Feature Scaling ---
# Scale the numerical features so they have a similar range
scaler = StandardScaler()

# Fit the scaler ONLY on the training data to avoid data leakage
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the scaler fitted on the training data
X_test_scaled = scaler.transform(X_test)
print("Applied feature scaling (Standardization).")


print("\n--- ✅ Model Preparation Complete! ---")
print("Your data is now fully cleaned, encoded, split, and scaled.")
print("The variables X_train_scaled, X_test_scaled, y_train, and y_test are ready for model training.")