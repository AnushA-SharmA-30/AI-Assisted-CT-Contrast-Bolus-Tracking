import streamlit as st
import pandas as pd
import joblib

# --- Caching the model and components for faster re-runs ---
# This decorator ensures that the model, scaler, and columns are loaded only once.
@st.cache_resource
def load_model():
    try:
        model = joblib.load('final_xgb_model.joblib')
        scaler = joblib.load('scaler.joblib')
        model_columns = joblib.load('model_columns.joblib')
        return model, scaler, model_columns
    except FileNotFoundError:
        return None, None, None

# --- Load the saved model and components ---
model, scaler, model_columns = load_model()

# --- App Title and Description ---
st.title('🩺 CT Scan Bolus Tracking Time Predictor')
st.write("""
This app predicts the optimal bolus tracking time for a CT scan based on patient and protocol data. 
Enter the patient's details below to get a prediction.
""")

# --- Check if model files are loaded ---
if model is None:
    st.error("Model files not found. Please make sure 'final_xgb_model.joblib', 'scaler.joblib', and 'model_columns.joblib' are in the same folder as this script.")
else:
    # --- Create the User Input Form ---
    with st.form("prediction_form"):
        st.header("Patient and Protocol Information")

        # Create columns for a cleaner layout
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Age', min_value=1, max_value=100, value=55)
            height_cm = st.number_input('Height (cm)', min_value=50.0, max_value=250.0, value=175.0, step=0.5)
            weight_kg = st.number_input('Weight (kg)', min_value=20.0, max_value=200.0, value=80.0, step=0.5)
            sex = st.selectbox('Sex', ('Male', 'Female'))

        with col2:
            contrast_volume_ml = st.number_input('Contrast Volume (ml)', min_value=20.0, max_value=150.0, value=85.0)
            flow_rate = st.number_input('Flow Rate (ml/s)', min_value=1.0, max_value=6.0, value=3.5, step=0.1)
            # Use a selectbox for contrast type to avoid typos
            contrast_type = st.selectbox('Contrast Type', ('Iopromide 370 mgI', 'Iohexol 350 mgI', 'Other'))
        
        # Submit button for the form
        submit_button = st.form_submit_button(label='Predict Time')

    # --- Prediction Logic ---
    if submit_button:
        # Create a dictionary from the user inputs
        new_patient_data = {
            'Age': age,
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'contrast_type': contrast_type,
            'contrast_volume_ml': contrast_volume_ml,
            'flow_rate': flow_rate,
            'Sex': sex
        }
        df_new = pd.DataFrame([new_patient_data])
        
        st.write("---")
        st.subheader("Input Data Summary:")
        st.write(df_new)

        # --- The Prediction Pipeline ---
        # 1. Feature Engineering
        df_new['dose_per_kg'] = df_new['contrast_volume_ml'] / df_new['weight_kg']
        df_new['bmi'] = df_new['weight_kg'] / ((df_new['height_cm'] / 100) ** 2)
        # 2. One-Hot Encoding
        df_new = pd.get_dummies(df_new)
        # 3. Align Columns
        df_new = df_new.reindex(columns=model_columns, fill_value=0)
        # 4. Scaling
        new_data_scaled = scaler.transform(df_new)
        # 5. Prediction
        prediction = model.predict(new_data_scaled)

        # --- Display the Result ---
        st.subheader("Prediction Result")
        st.metric(label="Predicted Bolus Tracking Time", value=f"{prediction[0]:.2f} seconds")