import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from utils import (
    TARGET_COL,
    add_features,
    build_preprocessor,
    clean_dataframe,
    drop_raw_coordinates,
    extract_feature_importance,
    get_feature_names,
    handle_missing_values,
    remove_invalid_coordinates,
    remove_outliers_iqr,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "Dataset.csv")
MODEL_PATH = os.path.join("models", "best_model.pkl")
REPORTS_DIR = "reports"
FAST_TRAIN = os.getenv("FAST_TRAIN", "1") == "1"
CV_FOLDS = 3 if FAST_TRAIN else 5
N_ITER = 4 if FAST_TRAIN else 8


def save_eda_plots(df: pd.DataFrame) -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    plt.figure(figsize=(8, 4))
    sns.histplot(df[TARGET_COL], kde=True, bins=30, color="#2a9d8f")
    plt.title("Delivery Time Distribution")
    plt.xlabel("Delivery Time (min)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "delivery_time_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, cmap="YlGnBu", annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "correlation_heatmap.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="distance_km", y=TARGET_COL, data=df, alpha=0.4, color="#e76f51")
    plt.title("Distance vs Delivery Time")
    plt.xlabel("Distance (km)")
    plt.ylabel("Delivery Time (min)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "distance_vs_time.png"))
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(x="Type_of_vehicle", y=TARGET_COL, data=df, color="#f4a261")
    plt.title("Vehicle Type Impact on Delivery Time")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Delivery Time (min)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "vehicle_type_impact.png"))
    plt.close()


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df)

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    df = remove_invalid_coordinates(df)
    df = handle_missing_values(df)
    df = add_features(df, seed=RANDOM_STATE)
    raw_min = float(df["eta_baseline_min"].min())
    raw_max = float(df["eta_baseline_min"].max())
    target_min = float(df[TARGET_COL].min())
    target_max = float(df[TARGET_COL].max())
    scale = (target_max - target_min) / (raw_max - raw_min) if raw_max > raw_min else 1.0
    offset = target_min - (scale * raw_min)
    calibration = {
        "raw_min": raw_min,
        "raw_max": raw_max,
        "target_min": target_min,
        "target_max": target_max,
        "scale": scale,
        "offset": offset,
    }
    df["eta_baseline_min"] = df["eta_baseline_min"] * scale + offset
    df = drop_raw_coordinates(df)
    df = remove_outliers_iqr(df, TARGET_COL)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if "Delivery_person_ID" in df.columns:
        df = df.drop(columns=["Delivery_person_ID"])

    save_eda_plots(df)

    y = df[TARGET_COL]
    x = df.drop(columns=[TARGET_COL])

    numeric_features = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "distance_km",
        "rating_age_interaction",
        "eta_baseline_min",
        "peak_hour",
    ]
    categorical_features = [
        "Type_of_order",
        "Type_of_vehicle",
        "traffic_intensity",
    ]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
        )
    except Exception:
        print("XGBoost is not available. Skipping XGBoost model.")

    tuned_models = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        if name == "Random Forest":
            param_grid = {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [8, 12, 16, None],
                "model__min_samples_split": [2, 5, 10],
            }
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=N_ITER,
                cv=CV_FOLDS,
                scoring="neg_root_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            search.fit(x_train, y_train)
            tuned_models[name] = search.best_estimator_
        elif name == "Gradient Boosting":
            param_grid = {
                "model__n_estimators": [150, 250, 350],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4],
            }
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=N_ITER,
                cv=CV_FOLDS,
                scoring="neg_root_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            search.fit(x_train, y_train)
            tuned_models[name] = search.best_estimator_
        elif name == "XGBoost":
            param_grid = {
                "model__n_estimators": [300, 500],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [5, 7, 9],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            }
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=N_ITER,
                cv=CV_FOLDS,
                scoring="neg_root_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            search.fit(x_train, y_train)
            tuned_models[name] = search.best_estimator_
        else:
            pipeline.fit(x_train, y_train)
            tuned_models[name] = pipeline

    best_name = None
    best_model = None
    best_rmse = float("inf")
    best_metrics = None

    for name, model in tuned_models.items():
        metrics = evaluate_model(model, x_test, y_test)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_name = name
            best_model = model
            best_metrics = metrics

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    # Refit best model on full dataset for deployment.
    best_model.fit(x, y)

    feature_names = get_feature_names(best_model.named_steps["preprocessor"])
    model_component = best_model.named_steps["model"]
    feature_importances = extract_feature_importance(model_component, feature_names)

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "metrics": best_metrics,
        "feature_importances": feature_importances,
        "calibration": calibration,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Best model: {best_name}")
    print(f"Metrics: {best_metrics}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
