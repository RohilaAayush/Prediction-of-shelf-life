import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="FMCG Shelf-Life Prediction Dashboard", layout="wide")

# Load assets
def load_assets():
    model = joblib.load('model.joblib')
    encoder = joblib.load('encoder.joblib')
    data = pd.read_csv('product_data.csv')
    return model, encoder, data

model, encoder, data = load_assets()

# UI Header
st.title("FMCG Shelf-Life Prediction Dashboard")
st.markdown("""
This tool predicts the remaining shelf life (in days) of FMCG products based on environmental and product-specific factors.\
You can view the training data, try predictions, and see which features influence shelf life the most.
""")

# Sidebar for user input
st.sidebar.header("Predict Shelf Life for a New Batch")
product_type = st.sidebar.selectbox("Product Type", ['Dairy', 'Bakery', 'Beverage', 'Snack'])
storage_temperature = st.sidebar.slider("Storage Temperature (°C)", 2.0, 35.0, 8.0, step=0.1)
storage_humidity = st.sidebar.slider("Storage Humidity (%)", 20.0, 90.0, 50.0, step=0.1)
packaging_type = st.sidebar.selectbox("Packaging Type", ['Plastic', 'Glass', 'Cardboard', 'Can'])
initial_quality_score = st.sidebar.slider("Initial Quality Score", 1, 10, 7, step=1)

user_input = pd.DataFrame({
    'product_type': [product_type],
    'packaging_type': [packaging_type],
    'storage_temperature_celsius': [storage_temperature],
    'storage_humidity_percent': [storage_humidity],
    'initial_quality_score': [initial_quality_score]
})

if st.sidebar.button("Predict Shelf Life"):
    # One-hot encode categorical features
    X_cat = user_input[['product_type', 'packaging_type']]
    X_num = user_input[['storage_temperature_celsius', 'storage_humidity_percent', 'initial_quality_score']]
    X_cat_encoded = encoder.transform(X_cat)
    X_all = np.concatenate([X_num.values, X_cat_encoded], axis=1)
    prediction = model.predict(X_all)[0]
    st.success(f"Predicted Shelf Life: {int(round(prediction))} days")

    # Feature importances
    # Combine feature names
    num_features = ['storage_temperature_celsius', 'storage_humidity_percent', 'initial_quality_score']
    cat_features = encoder.get_feature_names_out(['product_type', 'packaging_type'])
    all_features = list(num_features) + list(cat_features)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    st.subheader("Feature Importances")
    st.bar_chart(importance_df.set_index('Feature'))

# Show data
st.subheader("Training Data Used for Model")
st.dataframe(data, use_container_width=True)