import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

# --- TRAIN TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- DECISION TREE REGRESSOR ---
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)  # try different depths!
tree_model.fit(X_train, y_train)

# --- PREDICT & EVALUATE ---
y_pred = tree_model.predict(X_test)

print("R² Score (Decision Tree):", r2_score(y_test, y_pred))
print("RMSE (Decision Tree):", mean_squared_error(y_test, y_pred, squared=False))

# --- PLOT ---
plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual Exposure Time")
plt.ylabel("Predicted Exposure Time")
plt.title("Decision Tree - Actual vs Predicted")
plt.grid(True)
plt.show()
