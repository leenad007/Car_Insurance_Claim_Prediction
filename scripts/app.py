# import streamlit as st
# import requests
# import json

# # FastAPI URL (change to your local server or deployed API URL)
# FASTAPI_URL = "http://127.0.0.1:8000/predict/"

# # Title of the Streamlit app
# st.title("Prediction App")

# # Input form for user data
# with st.form(key='prediction_form'):
#     # Example of form fields for input
#     policy_tenure = st.number_input("Policy Tenure", min_value=0, max_value=100, step=1)
#     age_of_car = st.number_input("Age of Car", min_value=0, max_value=100, step=1)
#     age_of_policyholder = st.number_input("Age of Policyholder", min_value=18, max_value=100, step=1)
#     area_cluster = st.number_input("Area Cluster", min_value=0, max_value=10, step=1)  # New feature added
#     population_density = st.number_input("Population Density", min_value=0, max_value=10000, step=1)
#     make = st.text_input("Make of the Car")
#     segment = st.text_input("Segment")
#     model = st.text_input("Model")
#     fuel_type = st.text_input("Fuel Type")
#     max_torque = st.number_input("Max Torque", min_value=0, max_value=1000, step=1)
#     max_power = st.number_input("Max Power", min_value=0, max_value=1000, step=1)
#     engine_type = st.text_input("Engine Type")
#     airbags = st.number_input("Airbags", min_value=0, max_value=10, step=1)
#     is_esc = st.checkbox("Has ESC (Electronic Stability Control)")  # Boolean field
#     is_parking_sensors = st.checkbox("Has Parking Sensors")  # Boolean field
#     rear_brakes_type = st.text_input("Rear Brakes Type")
#     displacement = st.number_input("Displacement", min_value=0, max_value=5000, step=1)
#     cylinder = st.number_input("Cylinder", min_value=1, max_value=12, step=1)
#     transmission_type = st.text_input("Transmission Type")
#     gear_box = st.text_input("Gear Box Type")
#     steering_type = st.text_input("Steering Type")
#     turning_radius = st.number_input("Turning Radius", min_value=0.0, max_value=100.0, step=0.1)
#     length = st.number_input("Length", min_value=0.0, max_value=10.0, step=0.1)
#     width = st.number_input("Width", min_value=0.0, max_value=10.0, step=0.1)
#     height = st.number_input("Height", min_value=0.0, max_value=10.0, step=0.1)
#     gross_weight = st.number_input("Gross Weight", min_value=0, max_value=5000, step=1)
#     is_brake_assist = st.checkbox("Has Brake Assist")  # Boolean field
#     is_central_locking = st.checkbox("Has Central Locking")  # Boolean field
#     is_power_steering = st.checkbox("Has Power Steering")  # Boolean field
#     is_day_night_rear_view_mirror = st.checkbox("Has Day/Night Rear View Mirror")  # Boolean field
#     is_speed_alert = st.checkbox("Has Speed Alert")  # Boolean field
#     ncap_rating = st.number_input("NCAP Rating", min_value=0, max_value=5, step=1)

#     # Submit button
#     submit_button = st.form_submit_button(label='Make Prediction')

# # When the user submits the form
# if submit_button:
#     # Create the payload for the request
#     data = {
#         "policy_tenure": policy_tenure,
#         "age_of_car": age_of_car,
#         "age_of_policyholder": age_of_policyholder,
#         "area_cluster": area_cluster,
#         "population_density": population_density,
#         "make": make,
#         "segment": segment,
#         "model": model,
#         "fuel_type": fuel_type,
#         "max_torque": max_torque,
#         "max_power": max_power,
#         "engine_type": engine_type,
#         "airbags": airbags,
#         "is_esc": is_esc,
#         "is_parking_sensors": is_parking_sensors,
#         "rear_brakes_type": rear_brakes_type,
#         "displacement": displacement,
#         "cylinder": cylinder,
#         "transmission_type": transmission_type,
#         "gear_box": gear_box,
#         "steering_type": steering_type,
#         "turning_radius": turning_radius,
#         "length": length,
#         "width": width,
#         "height": height,
#         "gross_weight": gross_weight,
#         "is_brake_assist": is_brake_assist,
#         "is_central_locking": is_central_locking,
#         "is_power_steering": is_power_steering,
#         "is_day_night_rear_view_mirror": is_day_night_rear_view_mirror,
#         "is_speed_alert": is_speed_alert,
#         "ncap_rating": ncap_rating
#     }

#     # Send the data to the FastAPI model for prediction
#     try:
#         response = requests.post(FASTAPI_URL, json=data)
#         prediction = response.json().get("prediction")
        
#         if prediction is not None:
#             st.success(f"The predicted value is: {prediction}")
#         else:
#             st.error(f"Error: {response.json().get('error')}")
#     except Exception as e:
#         st.error(f"Error connecting to the FastAPI server: {str(e)}")




import streamlit as st
import requests
import numpy as np
import json

# Streamlit UI
st.title("Car Insurance Claim Prediction")

st.markdown("### Enter Vehicle and Policy Details")

# Input fields with meaningful names and proper types
policy_tenure = st.slider("Policy Tenure (Years)", 0, 10, 1)
age_of_car = st.slider("Car Age (Years)", 0, 20, 5)
age_of_policyholder = st.slider("Policyholder Age (Years)", 18, 100, 30)
population_density_log = st.number_input("Population Density of Area", min_value=0, max_value=1000000, value=50000, step=1000)
airbags = st.selectbox("Number of Airbags", [0, 1, 2, 4, 6, 8])
height = st.slider("Car Height (cm)", 100, 300, 150)
cylinder = st.selectbox("Engine Cylinders", [2, 3, 4, 6, 8])

# Categorical Features with meaningful labels
is_brake_assist = st.radio("Brake Assist", ["No", "Yes"]) 
is_front_fog_lights = st.radio("Front Fog Lights", ["No", "Yes"]) 
is_esc = st.radio("Electronic Stability Control (ESC)", ["No", "Yes"]) 
is_rear_window_defogger = st.radio("Rear Window Defogger", ["No", "Yes"]) 
is_parking_camera = st.radio("Parking Camera", ["No", "Yes"]) 
is_adjustable_steering = st.radio("Adjustable Steering", ["No", "Yes"]) 
is_speed_alert = st.radio("Speed Alert System", ["No", "Yes"]) 
is_parking_sensors = st.radio("Parking Sensors", ["No", "Yes"]) 
is_rear_window_wiper = st.radio("Rear Window Wiper", ["No", "Yes"]) 
is_rear_window_washer = st.radio("Rear Window Washer", ["No", "Yes"]) 
is_driver_seat_height_adjustable = st.radio("Driver Seat Height Adjustable", ["No", "Yes"]) 
is_day_night_rear_view_mirror = st.radio("Day-Night Rear View Mirror", ["No", "Yes"]) 
is_power_steering = st.radio("Power Steering", ["No", "Yes"]) 
steering_type_Power = st.radio("Power Steering Type", ["No", "Yes"]) 
engine_type_K10C = st.radio("Engine Type: K10C", ["No", "Yes"]) 
print("2222")
# Convert categorical values to binary
binary_mapping = {"No": 0, "Yes": 1}
binary_mapping_1 = {"No": False, "Yes": True}

data = {
    "policy_tenure": policy_tenure,
    "age_of_car": age_of_car,
    "age_of_policyholder": age_of_policyholder,
    "population_density_log": population_density_log,
    "airbags": airbags,
    "height": height,
    "cylinder": cylinder,
    "is_brake_assist": binary_mapping[is_brake_assist],
    "is_front_fog_lights": binary_mapping[is_front_fog_lights],
    "is_esc": binary_mapping[is_esc],
    "is_rear_window_defogger": binary_mapping[is_rear_window_defogger],
    "is_parking_camera": binary_mapping[is_parking_camera],
    "is_adjustable_steering": binary_mapping[is_adjustable_steering],
    "is_speed_alert": binary_mapping[is_speed_alert],
    "is_parking_sensors": binary_mapping[is_parking_sensors],
    "is_rear_window_wiper": binary_mapping[is_rear_window_wiper],
    "is_rear_window_washer": binary_mapping[is_rear_window_washer],
    "is_driver_seat_height_adjustable": binary_mapping[is_driver_seat_height_adjustable],
    "is_day_night_rear_view_mirror": binary_mapping[is_day_night_rear_view_mirror],
    "is_power_steering": binary_mapping[is_power_steering],
    "steering_type_Power": binary_mapping_1[steering_type_Power],
    "engine_type_K10C": binary_mapping_1[engine_type_K10C]
}

print("11111")
print(json.dumps(data, indent=2))  # Add this line before requests.post()

if st.button("Predict Claim Probability"):
    response = requests.post("http://127.0.0.1:8000/predict", data=json.dumps(data))
    result = response.json()
    st.success(f"Predicted Claim Probability: {float(result['prediction']):.2f}")
