import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


TARGET_COL = "Delivery Time_taken(min)"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Calculate the great-circle distance between two points on Earth.
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip whitespace from column names and key string columns.
    df.columns = [col.strip() for col in df.columns]

    for col in ["Type_of_order", "Type_of_vehicle"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Convert numeric columns safely.
    numeric_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    if TARGET_COL in df.columns:
        numeric_cols.append(TARGET_COL)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def remove_invalid_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conditions = (
        df["Restaurant_latitude"].between(-90, 90)
        & df["Restaurant_longitude"].between(-180, 180)
        & df["Delivery_location_latitude"].between(-90, 90)
        & df["Delivery_location_longitude"].between(-180, 180)
    )
    return df[conditions]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0] if not df[col].mode().empty else "unknown")

    return df


def remove_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[col] >= lower) & (df[col] <= upper)]


def apply_baseline_calibration_value(value: float, calibration: Optional[Dict[str, float]]) -> float:
    if not calibration:
        return float(value)

    scale = calibration.get("scale", 1.0)
    offset = calibration.get("offset", 0.0)
    return float(value * scale + offset)


def apply_baseline_calibration(series: pd.Series, calibration: Optional[Dict[str, float]]) -> pd.Series:
    if not calibration:
        return series.astype(float)

    scale = calibration.get("scale", 1.0)
    offset = calibration.get("offset", 0.0)
    return series.astype(float) * scale + offset


def add_features(df: pd.DataFrame, seed: int = 42, calibration: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    df = df.copy()

    df["distance_km"] = df.apply(
        lambda row: haversine_km(
            row["Restaurant_latitude"],
            row["Restaurant_longitude"],
            row["Delivery_location_latitude"],
            row["Delivery_location_longitude"],
        ),
        axis=1,
    )

    df["rating_age_interaction"] = df["Delivery_person_Ratings"] * df["Delivery_person_Age"]

    # Simulate peak hour and traffic intensity for modeling consistency.
    rng = np.random.default_rng(seed)
    if "peak_hour" not in df.columns:
        df["peak_hour"] = rng.integers(0, 2, size=len(df))
    if "traffic_intensity" not in df.columns:
        df["traffic_intensity"] = rng.choice(["low", "medium", "high"], size=len(df), p=[0.3, 0.5, 0.2])

    df["eta_baseline_min"] = df.apply(
        lambda row: compute_eta_baseline(
            row["distance_km"],
            row["Type_of_vehicle"],
            row["traffic_intensity"],
            row["peak_hour"],
        ),
        axis=1,
    )
    df["eta_baseline_min"] = apply_baseline_calibration(df["eta_baseline_min"], calibration)

    return df


def compute_eta_baseline(
    distance_km: float,
    vehicle_type: str,
    traffic_intensity: str,
    peak_hour: int,
) -> float:
    # Baseline ETA derived from distance, vehicle speed, traffic, and peak-hour adjustments.
    speed_map = {
        "cycle": 12.0,
        "bike": 18.0,
        "scooter": 22.0,
        "motorcycle": 25.0,
        "car": 28.0,
    }
    traffic_map = {"low": 1.0, "medium": 1.25, "high": 1.6}
    traffic_delay_map = {"low": 0.0, "medium": 4.0, "high": 8.0}

    speed_kmh = speed_map.get(str(vehicle_type).lower(), 20.0)
    traffic_multiplier = traffic_map.get(str(traffic_intensity).lower(), 1.25)
    traffic_delay = traffic_delay_map.get(str(traffic_intensity).lower(), 4.0)
    peak_multiplier = 1.1 if int(peak_hour) == 1 else 1.0

    speed_km_per_min = max(speed_kmh / 60.0, 0.1)
    base_minutes = (distance_km / speed_km_per_min) * traffic_multiplier * peak_multiplier
    return float(base_minutes + traffic_delay)


def drop_raw_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    existing = [col for col in cols if col in df.columns]
    return df.drop(columns=existing)


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    feature_names: List[str] = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = transformer
            cat_names = list(ohe.get_feature_names_out(cols))
            feature_names.extend(cat_names)
        elif name == "num":
            feature_names.extend(list(cols))

    return feature_names


def extract_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    importances: Dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        importances = dict(zip(feature_names, np.abs(coef)))

    return importances


def prepare_inference_row(inputs: Dict[str, float], calibration: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    data = {
        "Delivery_person_Age": [inputs["age"]],
        "Delivery_person_Ratings": [inputs["rating"]],
        "Restaurant_latitude": [inputs["rest_lat"]],
        "Restaurant_longitude": [inputs["rest_lon"]],
        "Delivery_location_latitude": [inputs["del_lat"]],
        "Delivery_location_longitude": [inputs["del_lon"]],
        "Type_of_order": [inputs["order_type"]],
        "Type_of_vehicle": [inputs["vehicle_type"]],
        "peak_hour": [inputs.get("peak_hour", 0)],
        "traffic_intensity": [inputs.get("traffic_intensity", "medium")],
    }
    df = pd.DataFrame(data)
    df = clean_dataframe(df)
    df = handle_missing_values(df)
    df = add_features(df, seed=42, calibration=calibration)
    df = drop_raw_coordinates(df)
    return df
