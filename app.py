import streamlit as st
import pandas as pd
import pickle

# Load artifacts
@st.cache_data
def load_artifacts():
    model    = pickle.load(open('models/rf_model.pkl','rb'))
    scaler   = pickle.load(open('models/scaler.pkl','rb'))
    features = pickle.load(open('models/feature_names.pkl','rb'))
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

st.title('Used Car Price Predictor')

# 1) User inputs
year = st.number_input('Year of Purchase', 1980, 2025, 2015)
km   = st.number_input('Kilometers Driven', 0, 500000, 50000)
mil  = st.number_input('Mileage (kmpl)', 0.0, 50.0, 18.0)
eng  = st.number_input('Engine (CC)', 500.0, 5000.0, 1200.0)
powr = st.number_input('Max Power (bhp)', 10.0, 500.0, 75.0)
seats_opt = [2,4,5,6,7,8]
seat  = st.selectbox('Seats', seats_opt)
fuel_opts = ['Petrol','Diesel','LPG']
fuel = st.selectbox('Fuel Type', fuel_opts)
trans_opts = ['Manual','Automatic']
trans = st.selectbox('Transmission', trans_opts)
owner_opts = ['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car']
owner = st.selectbox('Owner Type', owner_opts)

# Compute derived feature
CURRENT_YEAR = 2025
car_age = CURRENT_YEAR - year

# 2) Build a zero-filled DataFrame with correct features
X_new = pd.DataFrame([{c: 0 for c in feature_names}])

# 3) Assign numeric values
for col, val in zip(
    ['km_driven','mileage','engine','max_power','seats','car_age'],
    [km, mil, eng, powr, seat, car_age]
):
    X_new[col] = val

# 4) One-hot encode categorical selections matching feature_names exactly
# Fuel
X_new[f'fuel_{fuel}'] = 1
# Transmission
if trans == 'Manual':
    X_new['transmission_Manual'] = 1
# Owner Type
owner_col = 'owner_' + owner
if owner_col in X_new.columns:
    X_new[owner_col] = 1

# 5) Scale numeric features
num_cols = ['km_driven','mileage','engine','max_power','seats','car_age']
X_new[num_cols] = scaler.transform(X_new[num_cols])

# 6) Predict
if st.button('Predict Price'):
    price = model.predict(X_new)[0]
    st.success(f'Estimated Selling Price: ₹{price:,.0f}')