import pickle
import streamlit as st


# Load the scaler
with open('StandardScaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict_fire_risk(input_data):
    # Scale the input data
    scaled_data = scaler.transform([input_data])
    
    # Make prediction
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)
    
    return prediction[0], prediction_proba[0][1]

st.title("Forest Fire Risk Prediction")
st.write("Enter the following parameters to predict the risk of forest fire:")

# Input fields
temperature = st.number_input("Temperature (°C)", min_value=-30.0, max_value=50.0, value=20.0)
ffmc = st.number_input("FFMC", min_value=0.0, max_value=100.0, value=85.0)
dmc = st.number_input("DMC", min_value=0.0, max_value=300.0, value=50.0)
dc = st.number_input("DC", min_value=0.0, max_value=500.0, value=20.0)
isi = st.number_input("ISI", min_value=0.0, max_value=50.0, value=10.0)
bui = st.number_input("BUI", min_value=0.0, max_value=200.0, value=15.0)
fwi = st.number_input("FWI", min_value=0.0, max_value=50.0, value=5.0)

if st.button("Predict Fire Risk"):
    input_data = [temperature, ffmc, dmc, dc, isi, bui, fwi]
    risk, prob = predict_fire_risk(input_data)
    st.write(f"Predicted Fire Risk: {'High' if risk == 1 else 'Low'}")
    st.write(f"Probability of High Fire Risk: {prob:.2f}")
