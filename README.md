# Smart Food Delivery Time Estimator

An end-to-end machine learning system that predicts food delivery time (in minutes) using historical delivery data. It includes data preprocessing, feature engineering, model comparison, evaluation, and a Streamlit web app for real-time predictions.

## Project Structure

```
IFDS/
├── data/
│   └── Dataset.csv
├── models/
│   └── best_model.pkl
├── reports/
│   ├── correlation_heatmap.png
│   ├── delivery_time_distribution.png
│   ├── distance_vs_time.png
│   └── vehicle_type_impact.png
├── app.py
├── train.py
├── utils.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a Python 3.x environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

## Train the Model

```
python train.py
```

This will:
- Clean and preprocess the dataset
- Generate EDA plots in the `reports/` folder
- Train and compare multiple regression models
- Save the best model to `models/best_model.pkl`

## Run the App

```
streamlit run app.py
```

The app provides:
- Delivery time prediction based on order inputs
- Model performance metrics (MAE, RMSE, R2)
- Feature importance visualization
- Interactive distance vs delivery time plot

## Feature Engineering

- Haversine distance between restaurant and delivery coordinates
- Rating x age interaction
- Simulated peak hour and traffic intensity for richer modeling

## Model Evaluation

Models compared:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor (if available)

Selection is based on the lowest RMSE on the test set.

## Future Improvements

- Add real traffic and weather data sources
- Build a larger monitoring dashboard
- Add SHAP explainability
- Deploy on a cloud platform for public access
