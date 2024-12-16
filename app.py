import streamlit as st
import numpy as np
import pandas as pd
from preprocessing.cleaning_data import calculate_distances
from predict.prediction import RealEstateDeployment

df = pd.read_csv("data/preprocess_properties_data.csv")

belgium_coordinates = {
    "Antwerpen": {"latitude": 51.2194485, "longitude": 4.4024644},
    "Leuven": {"latitude": 50.8798194, "longitude": 4.7004614},
    "Brugge": {"latitude": 51.2093453, "longitude": 3.2247013},
    "Gent": {"latitude": 51.0543384, "longitude": 3.7174184},
    "Hasselt": {"latitude": 50.9306783, "longitude": 5.3373844},
    "Wavre": {"latitude": 50.7175865, "longitude": 4.6119332},
    "Mons": {"latitude": 50.4542209, "longitude": 3.9567027},
    "Liège": {"latitude": 50.6292239, "longitude": 5.5796765},
    "Arlon": {"latitude": 49.6833798, "longitude": 5.8166743},
    "Namur": {"latitude": 50.4669009, "longitude": 4.8674819},
    "Brussels": {"latitude": 50.8503438, "longitude": 4.3517103},
    "Charleroi": {"latitude": 50.4113064, "longitude": 4.4447003},
}
valid_postal_codes = df["Postal_Code"].unique()

def get_region_and_province(postal_code):
    postal_data = df[df["Postal_Code"] == postal_code]
    if postal_data.empty:
        raise ValueError(f"Postal_Code {postal_code} not found in the dataset.")
    region_numeric = postal_data["Region_Numeric"].unique()
    province_numeric = postal_data["Province_numeric"].unique()
    if len(region_numeric) > 1 or len(province_numeric) > 1:
        raise ValueError(
            f"Postal_Code {postal_code} is associated with multiple regions or provinces."
        )
    return region_numeric[0], province_numeric[0]


deployment = RealEstateDeployment(
    model_path="model/real_estate_model.joblib",
    x_scaler_path="model/x_scaler.joblib",
    y_scaler_path="model/y_scaler.joblib",
)

st.title("Belgium Real Estate Market Prediction")

st.write(
    """
Provide the details of the property to predict its estimated value.
"""
)

st.header("Property Details")

rooms = st.number_input("Number of Bedrooms", min_value=1, max_value=50, step=1)
living_area = st.number_input(
    "Livable Space (sq meters)", min_value=10, max_value=5000, step=1
)
room_space_combined = float(rooms * living_area)

garden_area = st.number_input(
    "Garden Area (sq meters)", min_value=0, max_value=1000, step=1
)
terrace_area = st.number_input(
    "Terrace Area (sq meters)", min_value=0, max_value=25000, step=1
)
surface_land = st.number_input(
    "Surface of the land (sq meters)", min_value=0, max_value=30000, step=1
)
outside_area = float(terrace_area + garden_area + surface_land)

postal_code = st.number_input("Postal Code", min_value=1000, max_value=9990, step=1)

if postal_code in valid_postal_codes:
    st.success(f"Valid Postal Code")
else:
    st.error(
        f"{postal_code} is not a valid postal code. "
        f"Here are some valid options: {', '.join(map(str, sorted(valid_postal_codes)[:10]))}..."
    )

try:
    region_numeric, province_numeric = get_region_and_province(postal_code)

    distances = calculate_distances(postal_code, df, belgium_coordinates)

except ValueError as e:
    st.error(e)
    distances = None

kitchen = st.radio(
    "Does the property have a fully equipped kitchen?",
    ["Yes", "No", "Skip this question"],
)
if kitchen == "Yes":
    kitchen_encoded = 2
elif kitchen == "No":
    kitchen_encoded = 0
else:
    kitchen_encoded = 1

swimming_pool = st.radio("Does the property have a swimming pool?", ["Yes", "No"])
swimming_pool_encoded = 1 if swimming_pool == "Yes" else 0

building_state = st.radio(
    "Building State",
    [
        "As New",
        "Good",
        "Just Renovated",
        "To renovate",
        "To be done up",
        "To restore",
        "Skip this question",
    ],
)
if building_state in ["As New", "Good", "Just Renovated"]:
    building_state_encoded = 2
elif building_state in ["To renovate", "To be done up", "To restore"]:
    building_state_encoded = 0
else:
    building_state_encoded = 1

number_of_frontages = st.slider("Number of Frontages", 1, 4, value=2)

subtype_category = st.selectbox(
    "Subtype Category",
    [
        "HOUSE",
        "APARTMENT",
        "VILLA",
        "APARTMENT_BLOCK",
        "GROUND_FLOOR",
        "MIXED_USE_BUILDING",
        "DUPLEX",
        "PENTHOUSE",
        "FLAT_STUDIO",
        "EXCEPTIONAL_PROPERTY",
        "MANSION",
        "TOWN_HOUSE",
        "BUNGALOW",
        "SERVICE_FLAT",
        "COUNTRY_COTTAGE",
        "LOFT",
        "TRIPLEX",
        "FARMHOUSE",
    ],
)
subtype_mapping = {
    "HOUSE": 1,
    "APARTMENT": 2,
    "VILLA": 3,
    "APARTMENT_BLOCK": 4,
    "GROUND_FLOOR": 5,
    "MIXED_USE_BUILDING": 6,
    "DUPLEX": 7,
    "PENTHOUSE": 8,
    "FLAT_STUDIO": 9,
    "EXCEPTIONAL_PROPERTY": 10,
    "MANSION": 11,
    "TOWN_HOUSE": 12,
    "BUNGALOW": 13,
    "SERVICE_FLAT": 14,
    "COUNTRY_COTTAGE": 15,
    "LOFT": 16,
    "TRIPLEX": 17,
    "FARMHOUSE": 18,
}
subtype_encoded = subtype_mapping[subtype_category]
if distances:
    input_data = [
        room_space_combined,
        outside_area,
        float(postal_code),
        float(region_numeric),
        kitchen_encoded,
        swimming_pool_encoded,
        building_state_encoded,
        number_of_frontages,
        float(province_numeric),
        subtype_encoded,
    ] + list(map(float, distances.values()))
input_data = np.array([input_data])
if st.button("Predict"):
    prediction = deployment.predict(input_data)
    st.success(f"The estimated property value is: €{prediction:,.2f}")
