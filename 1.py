import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- LOAD & CLEAN ---
df = pd.read_excel("abdomen_female.xlsx")
df['Total volume of contrast'] = df['Total volume of contrast'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
df['Height'] = df['Height'].astype(str).str.extract('(\d+\.?\d*)').astype(float)
df['Weight'] = df['Weight'].astype(str).str.extract('(\d+\.?\d*)').astype(float)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'].astype(str))
df['Scan type'] = le.fit_transform(df['Scan type'].astype(str))

df.dropna(subset=['Scan type', 'Total volume of contrast', 'Age', 'Sex', 'Height', 'Weight', 'Total Exposure Time'], inplace=True)

# --- FEATURES & TARGET ---
X = df[['Scan type', 'Total volume of contrast', 'Age', 'Sex', 'Height', 'Weight']]
y = df['Total Exposure Time']

# --- SCALE FEATURES ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- BUILD NN ---
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer (for regression)

model.compile(optimizer='adam', loss='mse')

# --- TRAIN ---
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)

# --- PREDICT & EVALUATE ---
y_pred = model.predict(X_test).flatten()

print("R² Score (NN):", r2_score(y_test, y_pred))
print("RMSE (NN):", mean_squared_error(y_test, y_pred, squared=False))
