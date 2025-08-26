import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_excel("abdomen_female.xlsx")  # Change to your actual filename

# --- PREPROCESSING ---
# Clean volume column
df['Total volume of contrast given to patient'] = df['Total volume of contrast given to patient'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
# Clean 'Age' column (remove characters like 'Y', spaces, etc.)
df['Age'] = df['Age'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
# Label encode 'Sex' and 'Scan type'
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'].astype(str))
df['Scan type'] = le.fit_transform(df['Scan type'].astype(str))

# Clean height & weight
df['Height (cm)'] = df['Height (cm)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['Weight (kg)'] = df['Weight (kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
# Clean 'Age' column
df['Age'] = df['Age'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
# Clean Total Exposure time column
df['Total Exposure time (total time of all the scans put together)'] = df['Total Exposure time (total time of all the scans put together)']\
    .astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# Drop missing values
df.dropna(subset=['Total volume of contrast given to patient', 'Age', 'Sex', 'Height (cm)', 'Weight (kg)', 'Scan type', 'Total Exposure time (total time of all the scans put together)'], inplace=True)

# --- FEATURES & TARGET ---
X = df[['Scan type', 'Total volume of contrast given to patient', 'Age', 'Sex', 'Height (cm)', 'Weight (kg)']]
y = df['Total Exposure time (total time of all the scans put together)']

# --- POLYNOMIAL FEATURES ---
poly = PolynomialFeatures(degree=2, include_bias=False)  # Degree 2 is good to start with
X_poly = poly.fit_transform(X)

# --- TRAIN TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# --- MODEL TRAINING ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- PREDICTION & EVALUATION ---
y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# --- OPTIONAL: SCATTER PLOT ---
plt.scatter(y_test, y_pred, color='purple')
plt.xlabel("Actual Exposure Time")
plt.ylabel("Predicted Exposure Time")
plt.title("Polynomial Regression - Actual vs Predicted")
plt.grid(True)
plt.show()
