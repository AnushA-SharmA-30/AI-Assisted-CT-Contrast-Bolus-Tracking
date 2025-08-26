import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# --- LOAD DATA ---
df = pd.read_excel("abdomen_female.xlsx")
df['Total volume of contrast given to patient'] = df['Total volume of contrast given to patient'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
# Clean 'Age' column (remove characters like 'Y', spaces, etc.)
df['Age'] = df['Age'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# Clean height and weight if needed
df['Height (cm)'] = df['Height (cm)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['Weight (kg)'] = df['Weight (kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'].astype(str))  # Female = 0, Male = 1 (usually)
df['Scan type'] = le.fit_transform(df['Scan type'].astype(str))
# Clean Total Exposure time column
df['Total Exposure time (total time of all the scans put together)'] = df['Total Exposure time (total time of all the scans put together)']\
    .astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
# Drop missing values from relevant columns
df.dropna(subset=['Scan type', 'Total volume of contrast given to patient', 'Age', 'Sex', 'Height (cm)', 'Weight (kg)', 'Total Exposure time (total time of all the scans put together)'], inplace=True)

X = df[['Scan type', 'Total volume of contrast given to patient', 'Age', 'Sex', 'Height (cm)', 'Weight (kg)']]
y = df['Total Exposure time (total time of all the scans put together)']


# --- FEATURES & TARGET ---


# --- POLYNOMIAL FEATURES ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# --- TRAIN TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# --- LASSO REGRESSION ---
lasso_model = Lasso(alpha=0.5)  # Start with 0.5; tune this!
lasso_model.fit(X_train, y_train)

# --- PREDICTION & EVALUATION ---
y_pred = lasso_model.predict(X_test)

print("R² Score (Lasso):", r2_score(y_test, y_pred))
print("RMSE (Lasso):", mean_squared_error(y_test, y_pred, squared=False))

# --- PLOT ---
plt.scatter(y_test, y_pred, color='orange')
plt.xlabel("Actual Exposure Time")
plt.ylabel("Predicted Exposure Time")
plt.title("Lasso Regression - Actual vs Predicted")
plt.grid(True)
plt.show()
