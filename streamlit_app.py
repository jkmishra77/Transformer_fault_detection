import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os

# App title
st.title("🔧 Transformer Fault Prediction")

# Initialize session state
if 'sample_df' not in st.session_state:
    st.session_state.sample_df = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Load model with error handling
def load_model():
    try:
        model_path = "models/best_model.joblib"
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.info("Please make sure your model is in the 'models' folder")
            return None
        
        model = joblib.load(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load sample data with error handling
def load_sample_data():
    try:
        data_path = "data/processed/validate.csv"
        if not os.path.exists(data_path):
            st.warning(f"⚠️ Data file not found at: {data_path}")
            st.info("Running in demo mode without sample data")
            return None
        
        sample_data = pd.read_csv(data_path)
        return sample_data
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Load model and data
if st.session_state.model is None:
    st.session_state.model = load_model()

sample_data = load_sample_data()

# Column descriptions
COLUMN_DESCRIPTIONS = {
    "vl1": "Phase Line 1 Voltage",
    "vl2": "Phase Line 2 Voltage", 
    "vl3": "Phase Line 3 Voltage",
    "il1": "Current Line 1",
    "il2": "Current Line 2",
    "il3": "Current Line 3",
    "vl12": "Voltage line 1-2",
    "vl23": "Voltage line 2-3",
    "vl31": "Voltage line 3-1",
    "inut": "Neutral Current",
    "oti": "Oil Temperature Indicator",
    "wti": "Winding Temperature Indicator", 
    "ati": "Ambient Temperature Indicator",
    "oli": "Oil Level Indicator",
    "oti_a": "Oil Temperature Indicator Alarm",
    "oti_t": "Oil Temperature Indicator Trip",
    "mog_a": "REAL TARGET FOR PREDICTION (0: OK, 1: FAULTY)"
}

# Main application logic
if sample_data is not None:
    # Initialize or refresh sample data
    if st.session_state.sample_df is None:
        st.session_state.sample_df = sample_data.sample(5, random_state=42)
    
    # Display samples
    st.subheader("📊 Samples from Dataset")
    st.dataframe(st.session_state.sample_df, use_container_width=True)
    
    # Let user select a sample
    selected_index = st.selectbox(
        "Choose a sample to test:", 
        range(len(st.session_state.sample_df)), 
        format_func=lambda x: f"Sample {x+1}"
    )
    
    # Get the selected sample
    selected_sample = st.session_state.sample_df.iloc[selected_index]
    
    # Display selected sample with descriptions
    st.subheader("📋 Selected Sample Details")
    for col, value in selected_sample.items():
        col_lower = col.lower()
        description = COLUMN_DESCRIPTIONS.get(col_lower, "No description available")
        
        # Special formatting for mog_a
        if col_lower == "mog_a":
            status = "✅ OK" if value == 0 else "❌ FAULTY"
            st.write(f"**{col}** ({description}): {value} {status}")
        else:
            st.write(f"**{col}** ({description}): {value:.4f}")
    
    # Prepare data for prediction (remove timestamp and target)
    prediction_data = selected_sample.drop(["devicetimestamp", "mog_a"], errors="ignore")
    
    # Predict button (only show if model is loaded)
    if st.session_state.model is not None:
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                try:
                    prediction = st.session_state.model.predict([prediction_data.values])
                    
                    # Colored prediction result
                    st.subheader("🎯 Prediction Result")
                    if prediction[0] == 1:
                        st.error("🚨 **FAULT DETECTED!**")
                    else:
                        st.success("✅ **NO FAULT DETECTED**")
                    
                    # Show confidence if available
                    if hasattr(st.session_state.model, 'predict_proba'):
                        proba = st.session_state.model.predict_proba([prediction_data.values])
                        confidence = max(proba[0])
                        st.metric("Prediction Confidence", f"{confidence:.1%}")
                    
                    # Show actual value for comparison
                    actual_mog = selected_sample.get('mog_a', None)
                    if actual_mog is not None:
                        st.info(f"Actual Status: {'FAULTY' if actual_mog == 1 else 'OK'}")
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
    
    # Refresh button
    if st.button("🔄 Refresh Samples", use_container_width=True):
        st.session_state.sample_df = sample_data.sample(5, random_state=42)
        st.rerun()

else:
    st.warning("⚠️ Sample data not available. Please check your data files.")
    st.info("""
    **Expected file structure:**
    ```
    your-app/
    ├── app.py
    ├── requirements.txt
    ├── models/
    │   └── best_model.joblib
    └── data/
        └── processed/
            └── validate.csv
    ```
    """)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.info("""
**About this app:**
- Predicts transformer faults using machine learning
- Uses validation data samples for testing
- Shows detailed feature descriptions
- Provides confidence scores for predictions
""")

# Add footer
st.markdown("---")
st.caption("Transformer Fault Prediction System | Built with Streamlit")