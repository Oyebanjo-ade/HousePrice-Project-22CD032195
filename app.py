import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .prediction-box {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-price {
        font-size: 3rem;
        font-weight: 700;
        color: #667eea;
        margin: 1rem 0;
    }
    .feature-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('model/house_price_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        encoder = joblib.load('model/neighborhood_encoder.pkl')
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure all model files are in the 'model/' directory")
        return None, None, None

# Load the model
model, scaler, encoder = load_model_and_preprocessors()

# Header
st.title("🏠 House Price Prediction System")
st.markdown("### Predict house prices using machine learning")
st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.header("📊 About This App")
    st.markdown("""
    This application predicts house sale prices using a **Random Forest Regressor** trained on the Kaggle House Prices dataset.
    
    **Features Used:**
    - Overall Quality
    - Living Area
    - Basement Area
    - Garage Size
    - Year Built
    - Neighborhood
    
    **Model Performance:**
    - R² Score: ~0.85
    - RMSE: ~$30,000
    - MAE: ~$20,000
    """)
    
    st.markdown("---")
    st.markdown("**Algorithm:** Random Forest")
    st.markdown("**Persistence:** Joblib")
    
    # Model status
    if model is not None:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model not loaded")

# Main content
if model is not None and scaler is not None and encoder is not None:
    
    # Get neighborhood options
    neighborhoods = sorted(encoder.classes_.tolist())
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏗️ Property Features")
        
        overall_qual = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=7,
            help="Overall material and finish quality (1=Poor, 10=Excellent)"
        )
        
        gr_liv_area = st.number_input(
            "Living Area (sq ft)",
            min_value=0,
            max_value=10000,
            value=1500,
            step=50,
            help="Above grade living area in square feet"
        )
        
        total_bsmt_sf = st.number_input(
            "Basement Area (sq ft)",
            min_value=0,
            max_value=5000,
            value=1000,
            step=50,
            help="Total basement area in square feet"
        )
    
    with col2:
        st.subheader("🏡 Additional Details")
        
        garage_cars = st.selectbox(
            "Garage Size (cars)",
            options=[0, 1, 2, 3, 4, 5],
            index=2,
            help="Number of cars that can fit in the garage"
        )
        
        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2024,
            value=2000,
            step=1,
            help="Original construction year"
        )
        
        neighborhood = st.selectbox(
            "Neighborhood",
            options=neighborhoods,
            help="Physical location within Ames city limits"
        )
    
    # Display input summary
    st.markdown("---")
    st.subheader("📋 Input Summary")
    
    input_data = {
        "Feature": ["Overall Quality", "Living Area", "Basement Area", "Garage Size", "Year Built", "Neighborhood"],
        "Value": [overall_qual, f"{gr_liv_area:,} sq ft", f"{total_bsmt_sf:,} sq ft", 
                  f"{garage_cars} cars", year_built, neighborhood]
    }
    input_df = pd.DataFrame(input_data)
    st.table(input_df)
    
    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict House Price", use_container_width=True):
        
        with st.spinner("Calculating prediction..."):
            try:
                # Encode neighborhood
                neighborhood_encoded = encoder.transform([neighborhood])[0]
                
                # Create feature array
                # Order: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, YearBuilt, Neighborhood_Encoded
                features = np.array([[
                    overall_qual,
                    gr_liv_area,
                    total_bsmt_sf,
                    garage_cars,
                    year_built,
                    neighborhood_encoded
                ]])
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                
                # Display prediction
                st.markdown("---")
                st.markdown("""
                    <div class='prediction-box'>
                        <h3>Predicted Sale Price</h3>
                        <div class='prediction-price'>${:,.2f}</div>
                        <p style='color: #666;'>Based on the features you provided</p>
                    </div>
                """.format(prediction), unsafe_allow_html=True)
                
                # Additional insights
                st.success("✅ Prediction completed successfully!")
                
                # Show confidence information
                with st.expander("📊 View Prediction Details"):
                    st.markdown(f"""
                    **Prediction Breakdown:**
                    - Base prediction: ${prediction:,.2f}
                    - Confidence interval: ±${prediction * 0.1:,.2f} (approximate)
                    
                    **Key Factors:**
                    - Quality rating has high impact on price
                    - Living area is a strong predictor
                    - Neighborhood affects pricing significantly
                    """)
                    
                    # Feature values used
                    st.markdown("**Feature Values Used:**")
                    feature_df = pd.DataFrame({
                        'Feature': ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'Neighborhood'],
                        'Value': [overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, year_built, neighborhood_encoded]
                    })
                    st.dataframe(feature_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Please check your input values and try again.")
    
    # Example predictions section
    st.markdown("---")
    with st.expander("💡 Try These Example Houses"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Starter Home**
            - Quality: 5
            - Living: 1000 sq ft
            - Basement: 500 sq ft
            - Garage: 1 car
            - Year: 1980
            - Area: NAmes
            
            *Expected: ~$120,000*
            """)
        
        with col2:
            st.markdown("""
            **Family Home**
            - Quality: 7
            - Living: 1500 sq ft
            - Basement: 1000 sq ft
            - Garage: 2 cars
            - Year: 2000
            - Area: Gilbert
            
            *Expected: ~$200,000*
            """)
        
        with col3:
            st.markdown("""
            **Luxury Home**
            - Quality: 9
            - Living: 2500 sq ft
            - Basement: 1500 sq ft
            - Garage: 3 cars
            - Year: 2010
            - Area: NoRidge
            
            *Expected: ~$350,000*
            """)

else:
    st.error("⚠️ Model files not found!")
    st.markdown("""
    ### Setup Instructions:
    
    1. Train the model using `model_building.ipynb`
    2. Ensure these files exist in the `model/` directory:
       - `house_price_model.pkl`
       - `scaler.pkl`
       - `neighborhood_encoder.pkl`
    3. Restart the Streamlit app
    
    **Need help?** Check the README.md for detailed instructions.
    """)