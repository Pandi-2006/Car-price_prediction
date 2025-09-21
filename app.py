import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from PIL import Image

# Load the trained model and feature names
try:
    model = joblib.load("model.pkl")
    feature_names = joblib.load("feature_names.pkl")
except:
    st.error("Model files not found. Please make sure 'model.pkl' and 'feature_names.pkl' are in the same directory.")
    st.stop()

# Load the original dataset to get unique values for brand and model
@st.cache_data
def load_data():
    df = pd.read_csv("cardekho_dataset.csv")
    return df

try:
    df_original = load_data()
except:
    st.warning("Could not load the original dataset. Some features might not work correctly.")
    df_original = pd.DataFrame()

# Streamlit app
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# Title and description
st.title("🚗 Car Price Prediction App")
st.markdown("""
This app predicts the selling price of used cars based on their features.
Simply fill in the details below and click the **Predict Price** button.
""")

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    # Brand selection
    if not df_original.empty:
        brands = sorted(df_original['brand'].unique())
        selected_brand = st.selectbox("Brand", brands)
        
        # Model selection based on brand
        models = sorted(df_original[df_original['brand'] == selected_brand]['model'].unique())
        selected_model = st.selectbox("Model", models)
    else:
        selected_brand = st.text_input("Brand")
        selected_model = st.text_input("Model")
    
    # Vehicle age
    vehicle_age = st.slider("Vehicle Age (years)", min_value=1, max_value=20, value=5)
    
    # Kilometers driven
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)

with col2:
    # Seller type
    seller_type = st.radio("Seller Type", ["Individual", "Dealer"])
    
    # Fuel type
    fuel_type = st.radio("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
    
    # Transmission type
    transmission_type = st.radio("Transmission Type", ["Manual", "Automatic"])
    
    # Technical specifications
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, value=20.0, step=0.1)
    engine = st.number_input("Engine (cc)", min_value=600, max_value=5000, value=1200, step=100)
    max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=500.0, value=80.0, step=0.1)
    seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8, 9, 10])

# Create a button for prediction
if st.button("Predict Price", type="primary"):
    # Create a dataframe with the input values
    input_data = {
        'vehicle_age': [vehicle_age],
        'km_driven': [km_driven],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats],
        'power_per_engine': [max_power / engine if engine > 0 else 0],
        'mileage_engine_factor': [mileage * engine]
    }
    
    # Add one-hot encoded columns with default 0
    for col in feature_names:
        if col not in input_data:
            input_data[col] = [0]
    
    # Create DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Set the correct one-hot encoded columns to 1
    seller_type_col = f"seller_type_{seller_type}"
    fuel_type_col = f"fuel_type_{fuel_type}"
    transmission_type_col = f"transmission_type_{transmission_type}"
    brand_col = f"brand_{selected_brand}" if selected_brand else ""
    model_col = f"model_{selected_model}" if selected_model else ""
    
    if seller_type_col in input_df.columns:
        input_df[seller_type_col] = 1
        
    if fuel_type_col in input_df.columns:
        input_df[fuel_type_col] = 1
        
    if transmission_type_col in input_df.columns:
        input_df[transmission_type_col] = 1
        
    if brand_col and brand_col in input_df.columns:
        input_df[brand_col] = 1
        
    if model_col and model_col in input_df.columns:
        input_df[model_col] = 1
    
    # Ensure the columns are in the same order as during training
    input_df = input_df[feature_names]
    
    # Make prediction
    try:
        log_prediction = model.predict(input_df)[0]
        prediction = np.exp(log_prediction)
        
        # Display the prediction
        st.success(f"### Predicted Selling Price: ₹{prediction:,.2f}")
        
        # Show some additional information
        st.info(f"""
        - Price in lakhs: ₹{prediction/100000:.2f} lakhs
        - This prediction is based on the features you provided.
        - Actual selling price may vary based on condition, location, and market demand.
        """)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add some information about the model
with st.expander("About the Model"):
    st.markdown("""
    This prediction model uses XGBoost, which achieved:
    - R² Score: 0.9345
    - RMSE: 0.1674
    
    The model was trained on historical car sales data and considers features like:
    - Vehicle age and kilometers driven
    - Technical specifications (engine, power, mileage)
    - Fuel type, transmission, and seller type
    - Brand and model information
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Car Price Prediction App by sandy | Built with Streamlit")