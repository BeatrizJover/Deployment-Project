import numpy as np
from joblib import load
from typing import Dict


class RealEstateDeployment:
    def __init__(self, model_path, x_scaler_path, y_scaler_path):
        self.model = load(model_path)
        self.x_scaler = load(x_scaler_path)
        self.y_scaler = load(y_scaler_path)

    def preprocess_input(self, input_data: Dict[str, float]) -> np.ndarray:
        """
        Preprocesses the input data to match the training pipeline.
        """

        input_array = np.array(input_data).reshape(1, -1)

        # Scale the input using the saved scaler
        input_scaled = self.x_scaler.transform(input_array)
        return input_scaled

    def predict(self, input_data: Dict[str, float]) -> float:
        """
        Predicts the price for a given input.
        """
        input_scaled = self.preprocess_input(input_data)
        prediction_scaled = self.model.predict(input_scaled)
       
        prediction = self.y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[
            0, 0
        ]
        prediction = np.expm1(prediction)
        return prediction
