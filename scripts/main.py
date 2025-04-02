# from fastapi import FastAPI
# import pickle
# import pandas as pd
# import os
# from sklearn.preprocessing import StandardScaler

# app = FastAPI()

# # Model and scaler paths
# MODEL_PATH = r"D:\Naira Documents\Northeastern University\Assignments\ALY6020\Final Proj\finalproject\scripts\logistic_regression_model.pkl"
# SCALER_PATH = r"D:\Naira Documents\Northeastern University\Assignments\ALY6020\Final Proj\finalproject\scripts\scaler.pkl"

# # Check if the model file exists before loading
# if os.path.exists(MODEL_PATH):
#     with open(MODEL_PATH, "rb") as f:
#         logistic_model = pickle.load(f)
# else:
#     logistic_model = None  # Handle missing model scenario

# # Check if the scaler file exists before loading
# if os.path.exists(SCALER_PATH):
#     with open(SCALER_PATH, "rb") as f:
#         scaler = pickle.load(f)
# else:
#     scaler = None  # Handle missing scaler scenario

# @app.get("/")
# def home():
#     return {"message": "FastAPI is running!"}

# @app.post("/predict/")
# def predict(data: dict):
#     if logistic_model is None or scaler is None:
#         return {"error": "Model or scaler file not found!"}

#     try:
#         # Convert input data to DataFrame
#         df = pd.DataFrame([data])

#         # Define all the features that were used during training
#         required_features = [
#             'policy_tenure', 'age_of_car', 'age_of_policyholder', 'area_cluster', 'population_density', 'make', 'segment',
#             'model', 'fuel_type', 'max_torque', 'max_power', 'engine_type', 'airbags', 'is_esc', 'is_parking_sensors', 
#             'rear_brakes_type', 'displacement', 'cylinder', 'transmission_type', 'gear_box', 'steering_type', 
#             'turning_radius', 'length', 'width', 'height', 'gross_weight', 'is_brake_assist', 'is_central_locking', 
#             'is_power_steering', 'is_day_night_rear_view_mirror', 'is_speed_alert', 'ncap_rating'
#         ]

        
#         # Ensure that the incoming data contains the required features, adding missing features with default values
#         for feature in required_features:
#             if feature not in df.columns:
#                 if 'is_' in feature:  # For boolean-like features (e.g., 'is_brake_assist')
#                     df[feature] = False
#                 else:
#                     df[feature] = 0  # Set numeric features to a default value (e.g., 0)
        
#         # Reorder columns to match the required features
#         df = df[required_features]

#         # Apply scaling to numeric features
#         numeric_features = ['policy_tenure', 'age_of_car', 'age_of_policyholder', 'population_density', 'displacement', 
#                             'cylinder', 'gear_box', 'turning_radius', 'length', 'width', 'height', 'gross_weight', 'ncap_rating']
        
#         # Ensure all numeric features are scaled
#         df[numeric_features] = scaler.transform(df[numeric_features])

#         # Make prediction
#         prediction = logistic_model.predict(df)[0]

#         return {"prediction": int(prediction)}
#     except Exception as e:
#         return {"error": str(e)}




















from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
import numpy as np
pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

# # Load the trained Logistic Regression model and scaler
# with open("model/logistic_regression_model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# with open("model/scaler.pkl", "rb") as scaler_file:
#     scaler = pickle.load(scaler_file)

model = pd.read_pickle("model/logistic_regression_model.pkl")
scaler = pd.read_pickle("model/scaler.pkl")

# Define feature lists
selected_features_l1 = [
    "is_brake_assist", "is_front_fog_lights", "is_esc", "is_rear_window_defogger",
    "policy_tenure", "is_parking_camera", "age_of_car", "is_adjustable_steering",
    "is_speed_alert",  "is_parking_sensors", "population_density_log",
    "airbags", "height", "engine_type_K10C", "is_rear_window_wiper", "is_rear_window_washer",
    "age_of_policyholder", "is_driver_seat_height_adjustable", "cylinder",
    "steering_type_Power", "is_day_night_rear_view_mirror", "is_power_steering"
]

numeric_features = [
    "policy_tenure", "age_of_car",  "population_density_log",
    "airbags", "height", "age_of_policyholder", "cylinder"
]

# Define input data model
class InputData(BaseModel):
    is_brake_assist: int
    is_front_fog_lights: int
    is_esc: int
    is_rear_window_defogger: int
    policy_tenure: float
    is_parking_camera: int
    age_of_car: float
    is_adjustable_steering: int
    is_speed_alert: int
    is_parking_sensors: int
    population_density_log: float
    airbags: int
    height: float
    engine_type_K10C: int
    is_rear_window_wiper: int
    is_rear_window_washer: int
    age_of_policyholder: float
    is_driver_seat_height_adjustable: int
    cylinder: int
    steering_type_Power: int
    is_day_night_rear_view_mirror: int
    is_power_steering: int

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[selected_features_l1]
    # Scale numeric features
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    print(input_df)
    # Predict using the model
    prediction = model.predict(input_df)[0]
    
    return {"prediction": int(prediction)}
