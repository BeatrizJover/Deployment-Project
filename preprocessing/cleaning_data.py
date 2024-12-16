from math import radians, sin, cos, sqrt, atan2
from typing import Dict
import pandas as pd


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two geographical points.

    Args:
    - lat1, lon1: Latitude and longitude of the first point.
    - lat2, lon2: Latitude and longitude of the second point.

    Returns:
    - Distance in kilometers as a float.
    """
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return round(earth_radius_km * c, 2)


def get_coordinates(postal_code, df):
    """
    Returns the latitude and longitude corresponding to a given postal code.

    Args:
        postal_code (str or int): The postal code to search for.
        df (pd.DataFrame): DataFrame containing the columns 'Postal_Code', 'Latitude', and 'Longitude'.

    Returns:
        tuple: A tuple with latitude and longitude (latitude, longitude).
        None: If the postal code is not found.
    """
    try:
        # Filter the DataFrame to find the postal code
        result = df[df["Postal_Code"] == postal_code]

        if not result.empty:
            latitude = result["Latitude"].values[0]
            longitude = result["Longitude"].values[0]
            return latitude, longitude
        else:
            print(f"Postal code {postal_code} not found.")
            return None
    except KeyError as e:
        print(f"Error: column not found. {e}")
        return None


def calculate_distances(
    postal_code: float,
    postal_df: pd.DataFrame,
    city_coords: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Calculate distances from a postal code to a set of cities.

    Args:
    - postal_code: The postal code to find the coordinates for.
    - postal_df: A DataFrame containing postal code, latitude, and longitude information.
    - city_coords: A dictionary of city names and their coordinates.

    Returns:
    - A dictionary with distances to each city.
    """
    coordinates = get_coordinates(postal_code, postal_df)

    if not coordinates:
        raise ValueError(f"Postal code {postal_code} not found in the DataFrame.")

    lat, lon = coordinates
    distances = {}

    for city, coords in city_coords.items():
        distances[f"Dist_{city}"] = haversine_distance(
            lat, lon, coords["latitude"], coords["longitude"]
        )

    return distances
