
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Vehicle Valuation System", layout="wide")

st.title("Vehicle Valuation and Advisory System")
st.write("An advanced machine learning pipeline for local market price estimation and vehicle health tracking.")

# load the trained random forest model
with open('app/vehicle_model.pkl', 'rb') as f:
    model = pickle.load(f)

# main layout columns to separate inputs from the results view
col_input, col_display = st.columns([1, 1])

with col_input:
    st.subheader("Vehicle Parameters")
    
    year = st.number_input("Model Year", min_value=1990, max_value=2026, value=2018)
    milage_km = st.number_input("Total Mileage driven (in Kilometers)", min_value=0, value=65000)
    
    brand_select = st.selectbox("Manufacturer / Brand", ["Toyota", "Suzuki", "Honda", "Nissan", "Ford", "Hyundai", "BMW", "Audi", "Other"])
    fuel_select = st.selectbox("Engine Fuel Configuration", ["Gasoline", "Diesel", "Hybrid", "Electric", "Plug-In Hybrid", "Unknown"])
    trans_select = st.selectbox("Gearbox / Transmission Type", ["Automatic", "Manual", "Tiptronic"])
    
    accident_select = st.selectbox("Reported Accident History", ["None reported", "At least 1 accident or damage reported", "Unknown"])
    title_select = st.selectbox("Registration Book Status (Clean Title)", ["Yes", "Unknown"])
    
    engine_input = st.text_input("Engine Displacement (cc or Liters)", "1500cc")
    ext_color = st.text_input("Body Color Exterior", "White")
    int_color = st.text_input("Cabin Color Interior", "Black")
    car_model_input = st.text_input("Specific Trim / Variant Name", "Prius")

with col_display:
    st.subheader("System Output and Analysis")
    
    st.info("System Ready. Click the action button below to run the prediction pipeline.")
    
    # dictionary mapping tokens for the underlying random forest model
    brand_dict = {"Ford": 0, "Hyundai": 1, "Audi": 4, "BMW": 5, "Toyota": 6, "Honda": 7, "Suzuki": 8, "Nissan": 8, "Other": 8}
    fuel_dict = {"Gasoline": 1, "Hybrid": 2, "Unknown": 3, "Diesel": 4, "Electric": 5, "Plug-In Hybrid": 5}
    trans_dict = {"Automatic": 2, "Tiptronic": 1, "Manual": 4}
    accident_dict = {"At least 1 accident or damage reported": 0, "None reported": 1, "Unknown": 2}
    title_dict = {"Yes": 0, "Unknown": 1}

    if st.button("Evaluate Market Valuation", use_container_width=True):
        
        # internal unit conversion step from metric km to imperial miles
        miles_converted = milage_km * 0.621371
        
        # map user selections to structural array values
        brand_code = brand_dict.get(brand_select, 8)
        fuel_code = fuel_dict.get(fuel_select, 3)
        trans_code = trans_dict.get(trans_select, 2)
        accident_code = accident_dict.get(accident_select, 2)
        title_code = title_dict.get(title_select, 1)
        
        # string fallback tokenization parsing
        model_code = len(car_model_input) % 50
        engine_code = len(engine_input) % 50
        ext_code = len(ext_color) % 30
        int_code = len(int_color) % 30
        
        # build input feature matrix matching training frame
        input_matrix = np.array([[brand_code, model_code, year, miles_converted, fuel_code, engine_code, trans_code, ext_code, int_code, accident_code, title_code]])
        
        # base value generation
        base_usd_evaluation = model.predict(input_matrix)[0]
        
        # macro economic adjustment factors for local import structures
        exchange_rate_lkr = 300.0
        scarcity_premium_multiplier = 2.4
        
        expected_lkr = base_usd_evaluation * exchange_rate_lkr * scarcity_premium_multiplier
        
        # generate realistic range brackets for used market variations
        min_lkr_range = expected_lkr * 0.92
        max_lkr_range = expected_lkr * 1.08
        
        st.write("---")
        st.write("### Estimated Local Market Valuation")
        st.metric(label="Expected Market Price (LKR)", value=f"Rs. {expected_lkr:,.2f}")
        
        st.write("#### Valuation Range Chart")
        
        # put range values into a quick dataframe for plotting
        range_df = pd.DataFrame({
            'Market Bracket': ['Minimum Value', 'Expected Value', 'Maximum Value'],
            'Price (LKR)': [min_lkr_range, expected_lkr, max_lkr_range]
        })
        range_df = range_df.set_index('Market Bracket')
        
        # display native streamlit bar chart for visualization
        st.bar_chart(range_df)
        
        st.write(f"**Trading Bracket Details:** Rs. {min_lkr_range:,.2f} to Rs. {max_lkr_range:,.2f}")
        
        # section for dynamic maintenance alerts based on user metrics
        st.write("---")
        st.write("#### Technical Maintenance Risks")
        if milage_km > 100000:
            st.warning("High Mileage Risk: Inspect engine timing belt, suspension bushings, and transmission fluid degradation immediately.")
        elif milage_km > 60000:
            st.warning("Mid Mileage Checkup: Evaluate brake rotor wear profile and spark plug lifespan values.")
        else:
            st.success("Low Mileage Framework: Standard preventive oil changes and cabin filter refreshes recommended.")
            
        if fuel_select == "Hybrid" and year < 2016:
            st.warning("Hybrid Battery Alert: Check the state of health (SOH) of the inverter cooling system and cell balance parameters.")

        # section for administrative/legal requirements
        st.write("---")
        st.write("#### Required Legal Next Steps")
        st.write("To transfer this vehicle safely under new ownership:")
        st.checkbox("Obtain the MTA 6 and MTA 8 official transfer documents signed by the previous owner.")
        st.checkbox("Verify the chassis number plate coordinates directly against the registration book entries.")
        st.checkbox("Secure a clean vehicle clearance report from the local police department data logs.")