import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    TARGET_COL,
    add_features,
    clean_dataframe,
    compute_eta_baseline,
    handle_missing_values,
    apply_baseline_calibration_value,
    prepare_inference_row,
)

DATA_PATH = os.path.join("data", "Dataset.csv")
MODEL_PATH = os.path.join("models", "best_model.pkl")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df)
    df = df.dropna(subset=[TARGET_COL])
    df = handle_missing_values(df)
    df = add_features(df, seed=42)
    return df


st.set_page_config(
    page_title="Smart Food Delivery Time Estimator",
    page_icon="\U0001F69A",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        background: linear-gradient(120deg, #0f766e 0%, #f4a261 100%);
        color: white;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        margin-bottom: 1.2rem;
    }
    .metric-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #fff7ed;
        border: 1px solid #fcd34d;
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        text-align: center;
        color: #111827;
    }
    .footer {
        margin-top: 2rem;
        padding: 0.8rem;
        color: #475569;
        text-align: center;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Smart Food Delivery Time Estimator</h1>
        <p>AI-powered delivery time prediction for optimized logistics.</p>
        <p>Plan staffing, set customer expectations, and optimize routes with reliable forecasts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

artifact = load_artifact()
model = artifact["model"]
metrics = artifact["metrics"]
feature_importances = artifact.get("feature_importances", {})
calibration = artifact.get("calibration")

try:
    df_choices = load_dataset()
    order_choices = sorted(df_choices["Type_of_order"].dropna().unique().tolist())
    vehicle_choices = sorted(df_choices["Type_of_vehicle"].dropna().unique().tolist())
except Exception:
    order_choices = ["snack", "meal", "beverage", "grocery", "drinks", "buffet"]
    vehicle_choices = ["bike", "scooter", "cycle", "car", "motorcycle"]

with st.sidebar:
    st.header("Order Inputs")
    age = st.slider("Delivery Partner Age", 18, 60, 30)
    rating = st.slider("Delivery Partner Rating", 1.0, 5.0, 4.5, 0.1)

    st.subheader("Restaurant Location")
    rest_lat = st.number_input("Restaurant Latitude", value=12.9716, format="%.6f")
    rest_lon = st.number_input("Restaurant Longitude", value=77.5946, format="%.6f")

    st.subheader("Delivery Location")
    del_lat = st.number_input("Delivery Latitude", value=12.9352, format="%.6f")
    del_lon = st.number_input("Delivery Longitude", value=77.6245, format="%.6f")

    order_type = st.selectbox("Type of Order", order_choices)
    vehicle_type = st.selectbox("Type of Vehicle", vehicle_choices)

    traffic_intensity = st.selectbox("Traffic Intensity", ["low", "medium", "high"], index=1)
    peak_hour = st.toggle("Peak Hour", value=False)

    predict_btn = st.button("Predict Delivery Time")

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Prediction")

    if predict_btn:
        with st.spinner("Crunching delivery signals..."):
            inputs = {
                "age": age,
                "rating": rating,
                "rest_lat": rest_lat,
                "rest_lon": rest_lon,
                "del_lat": del_lat,
                "del_lon": del_lon,
                "order_type": order_type,
                "vehicle_type": vehicle_type,
                "traffic_intensity": traffic_intensity,
                "peak_hour": int(peak_hour),
            }
            row = prepare_inference_row(inputs, calibration=calibration)
            model_prediction = float(model.predict(row)[0])
            distance_km = float(row["distance_km"].iloc[0])

            baseline_current = compute_eta_baseline(
                distance_km,
                vehicle_type,
                traffic_intensity,
                int(peak_hour),
            )
            baseline_medium = compute_eta_baseline(
                distance_km,
                vehicle_type,
                "medium",
                int(peak_hour),
            )

            baseline_current = apply_baseline_calibration_value(baseline_current, calibration)
            baseline_medium = apply_baseline_calibration_value(baseline_medium, calibration)

            prediction = max(1.0, model_prediction + (baseline_current - baseline_medium))

        st.markdown(
            f"<div class='metric-card'><h2>{prediction:.2f} Minutes</h2><p>Estimated Delivery Time (traffic-adjusted)</p></div>",
            unsafe_allow_html=True,
        )

        if prediction < 25:
            st.success("Quick delivery expected. Great for customer delight!")
        elif prediction < 45:
            st.warning("Moderate delivery time. Consider routing optimization.")
        else:
            st.error("Longer delivery expected. Proactive customer updates recommended.")
    else:
        st.info("Enter order details and click Predict Delivery Time.")

with col_right:
    st.subheader("Model Performance")
    st.metric("Model", artifact.get("model_name", "Best Model"))
    st.metric("MAE (avg error in minutes)", f"{metrics['mae']:.2f}")
    st.metric("RMSE (penalizes larger errors)", f"{metrics['rmse']:.2f}")
    st.metric("R2 (explained variance)", f"{metrics['r2']:.2f}")
    st.caption("Lower MAE/RMSE is better. R2 closer to 1.0 indicates stronger fit.")

    if feature_importances:
        fi = (
            pd.Series(feature_importances)
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fi.columns = ["feature", "importance"]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=fi, x="importance", y="feature", color="#2a9d8f", ax=ax)
        ax.set_title("Top Feature Importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        st.pyplot(fig)

st.subheader("Average Delivery Time by Distance Bucket")

try:
    df_viz = load_dataset()
    df_viz = df_viz.copy()
    df_viz["distance_bucket"] = pd.cut(
        df_viz["distance_km"],
        bins=[0, 2, 5, 8, 12, 20, 30, 50],
        right=False,
    )
    bucket_stats = (
        df_viz.groupby("distance_bucket")[TARGET_COL]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=bucket_stats, x="distance_bucket", y=TARGET_COL, color="#e76f51", ax=ax)
    ax.set_title("Average Delivery Time by Distance")
    ax.set_xlabel("Distance Range (km)")
    ax.set_ylabel("Avg Delivery Time (min)")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)
except Exception:
    st.info("Dataset not available for visualization.")

