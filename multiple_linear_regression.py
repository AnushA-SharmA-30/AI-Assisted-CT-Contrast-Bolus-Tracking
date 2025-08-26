import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("abdomen_female.xlsx")

# --- PREPROCESSING ---

# Clean and convert Total Volume to numeric
df['Total volume of contrast given to patient'] = df['Total volume of contrast given to patient']\
    .astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
# Clean 'Age' column (remove characters like 'Y', spaces, etc.)
df['Age'] = df['Age'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)


# Encode categorical columns
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'].astype(str))
df['Scan type'] = le.fit_transform(df['Scan type'].astype(str))

# Clean height and weight columns (correct column names!)
df['Height (cm)'] = df['Height (cm)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['Weight (kg)'] = df['Weight (kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
# Clean 'Age' column
df['Age'] = df['Age'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)


# Clean Total Exposure time column
df['Total Exposure time (total time of all the scans put together)'] = df['Total Exposure time (total time of all the scans put together)']\
    .astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# Drop missing values in required columns
df.dropna(subset=[
    'Total volume of contrast given to patient',
    'Age',
    'Sex',
    'Height (cm)',
    'Weight (kg)',
    'Scan type',
    'Total Exposure time (total time of all the scans put together)'
], inplace=True)

# --- FEATURE SELECTION ---
X = df[[
    'Scan type',
    'Total volume of contrast given to patient',
    'Age',
    'Sex',
    'Height (cm)',
    'Weight (kg)'
]]
y = df['Total Exposure time (total time of all the scans put together)']

# --- TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL FITTING ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- PREDICTION ---
y_pred = model.predict(X_test)

# --- EVALUATION ---
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# --- PLOT ---
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Exposure Time")
plt.ylabel("Predicted Exposure Time")
plt.title("Actual vs Predicted Total Exposure Time")
plt.grid(True)
plt.show()
