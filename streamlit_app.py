import streamlit as st
import pandas as pd
import joblib

# App title
st.title("Transformer Fault Prediction")

# Load model
model = joblib.load("models/best_model.joblib")

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
    "mog_a": "REAL TERGET FOR PREDICTION (0: OK, 1: FAULTY)"
}

# Load and store sample data in session state
if 'sample_df' not in st.session_state:
    sample_data = pd.read_csv("data/processed/validate.csv")
    st.session_state.sample_df = sample_data.sample(5)

# Display samples
st.subheader("Samples from Dataset")
st.dataframe(st.session_state.sample_df, use_container_width=True)

# Let user select a sample
selected_index = st.selectbox("Choose a sample to test:", range(5), format_func=lambda x: f"Sample {x+1}")

# Get the selected sample
selected_sample = st.session_state.sample_df.iloc[selected_index]

# Display selected sample with descriptions
st.write("**Selected Sample Values:**")
for col, value in selected_sample.items():
    col_lower = col.lower()
    description = COLUMN_DESCRIPTIONS.get(col_lower, "No description available")
    
    # Special formatting for mog_a
    if col_lower == "mog_a":
        status = "✅ OK" if value == 0 else "❌ FAULTY"
        st.write(f"**{col}** ({description}): {value} {status}")
    else:
        st.write(f"**{col}** ({description}): {value}")

# Prepare data for prediction (remove timestamp and target)
prediction_data = selected_sample.drop(["devicetimestamp", "mog_a"], errors="ignore")

# Predict
if st.button("Predict"):
    prediction = model.predict([prediction_data.values])
    
    # Colored prediction result
    if prediction[0] == 1:
        st.error("🚨 **FAULT DETECTED!**")
    else:
        st.success("✅ **NO FAULT DETECTED**")

# Refresh button to get new samples
if st.button("Refresh Samples"):
    sample_data = pd.read_csv("data/processed/validate.csv")
    st.session_state.sample_df = sample_data.sample(5)
    st.rerun()