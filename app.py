import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(
    page_title="Used Car Price Estimator | AutoAI",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Remove Streamlit default style */
    #MainMenu, footer, header {visibility: hidden;}

    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        margin-top: -20px;
        color: #1f77b4;
    }

    .sub-title {
        font-size: 20px;
        text-align: center;
        color: #444;
        margin-bottom: 40px;
    }

    .result {
        font-size: 28px;
        color: green;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
    }

    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: #888;
        padding: 10px 0;
    }

    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_data(show_spinner="🔍 Loading model and features...")
def load_artifacts():
    model = pickle.load(open('models/rf_model.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    features = pickle.load(open('models/feature_names.pkl', 'rb'))
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

# Title and subtitle
st.markdown("<div class='main-title'>Used Car Price Estimator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Get an instant price estimate for your pre-owned vehicle</div>", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("📥 Car Details")
    year = st.number_input('Year of Purchase', 1980, 2025, 2015)
    km = st.number_input('Kilometers Driven', 0, 500000, 50000)
    mil = st.number_input('Mileage (km/l)', 0.0, 50.0, 18.0)
    eng = st.number_input('Engine (CC)', 500.0, 5000.0, 1200.0)
    powr = st.number_input('Max Power (bhp)', 10.0, 500.0, 75.0)
    seat = st.selectbox('Seats', [2, 4, 5, 6, 7, 8])
    fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'LPG'])
    trans = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.selectbox('Owner Type', [
        'First Owner', 'Second Owner', 'Third Owner',
        'Fourth & Above Owner', 'Test Drive Car'
    ])

# Compute derived values
car_age = 2025 - year

# Initialize prediction row
X_new = pd.DataFrame([{c: 0 for c in feature_names}])
numeric_data = {
    'km_driven': km,
    'mileage': mil,
    'engine': eng,
    'max_power': powr,
    'seats': seat,
    'car_age': car_age
}
for col, val in numeric_data.items():
    X_new[col] = val

# One-hot encode
X_new[f'fuel_{fuel}'] = 1
if trans == 'Manual':
    X_new['transmission_Manual'] = 1
owner_col = f'owner_{owner}'
if owner_col in X_new.columns:
    X_new[owner_col] = 1

# Scale numeric features
X_new[list(numeric_data.keys())] = scaler.transform(X_new[list(numeric_data.keys())])

# Predict
st.markdown("### 🔮 Estimated Price")
if st.button("Predict Price"):
    price = model.predict(X_new)[0]
    st.markdown(f"<div class='result'>💰 ₹{price:,.0f}</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2025 AutoAI • Built with Streamlit</div>", unsafe_allow_html=True)
