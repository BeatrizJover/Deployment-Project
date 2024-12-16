# Belgium Real Estate Market Prediction

This project provides a machine learning model deployed in a Streamlit app for predicting real estate prices in Belgium. The app allows users to input property details, such as the number of rooms, area, and location, to predict the estimated price of a property. The model leverages a set of features including geographical data and property attributes to provide an accurate estimation.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [Timeline](#timeline)

## Description

The Streamlit app allows the user to input various details about a property, such as:

- Number of bedrooms
- Living area (in square meters)
- Garden and terrace area
- Postal code (for geographical distance calculations)
- Whether the property has a kitchen or a swimming pool
- The condition of the building
- The number of frontages
- The property subtype (e.g., house, apartment, villa, etc.)

After entering the property details, the app uses a pre-trained machine learning model (`real_estate_model.joblib`) to predict the price of the property based on the input data. The model also takes into account the geographical location of the property using the postal code and calculates distances from several major Belgian cities.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

   Make sure you have the following packages:
   - `streamlit`
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `joblib`  

3. **Download the model and scaler files**:
   Ensure that the following files are available in the `model/` directory:
   - `real_estate_model.joblib`
   - `x_scaler.joblib`
   - `y_scaler.joblib`

4. **Run the Streamlit app**:

    ```bash
    streamlit run app.py
    ```

## Usage

1. Once the app is running, navigate to the Streamlit interface in your browser.
2. **Input Property Details**:
   - Enter the number of bedrooms, living area, garden area, and other details.
   - Choose whether the property has a fully equipped kitchen, a swimming pool, and other features.
   - Select the property’s condition and subtype.
   - Input the postal code to calculate geographical distances.

3. **Click "Predict"**:
   - Once all details are filled in, click the **Predict** button.
   - The app will process the input and display the predicted property value in Euros (€).

## Contributors

[BeatrizJover](https://github.com/BeatrizJover)

## Timeline

This stage of the project lasted 5 days.Deadline: 18/12/2024 17:00.
