# Contrail Prediction Model

Predicting persistent contrail formation using CoCiP physics simulations, ERA5 atmospheric data, and machine learning classifiers.

---

## Table of Contents

- [About The Project](#about-the-project)
- [Running The Project](#running-the-project)
- [Usage](#usage)
- [Contact](#contact)

---

## About The Project

This project predicts persistent contrail formation by labeling flight waypoints with the Contrail Cirrus Prediction model (CoCiP) and training machine learning classifiers on ERA5 atmospheric data. 

Instead of running heavy physics simulations for every flight, this pipeline:
1. Pulls real meteorological data and maps Ice-Supersaturated Regions (ISSR).
2. Runs CoCiP on a synthetic fleet of flights to generate ground-truth labels.
3. Trains and compares Random Forest and XGBoost models using key atmospheric features (`rhi`, `air_temperature`, `altitude`, `specific_humidity`, `latitude`).

---

## Built With

- Python
- Google Colab
- PyContrails
- ERA5
- XGBoost
- Scikit-Learn

---

## Running The Project

This project was built entirely on **Google Colab**. You do not need to use the command line or install anything on your computer.

Click the button below to open the main notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fiona300/contrail-prediction-model/blob/main/03_contrail_classifier.ipynb)

**Note:** You will still need a free [Copernicus CDS API key](https://cds.climate.copernicus.eu/api-how-to) to download the weather data. The notebook will prompt you to paste it in.

---

## Usage

Run the notebooks in order inside Colab:

1. **`01_era5_issr_exploration.ipynb`**: Fetches ERA5 data and visualizes ISSR coverage.
2. **`02_cocip_single_flight.ipynb`**: Runs a CoCiP simulation on a single flight path.
3. **`03_contrail_classifier.ipynb`**: Generates a synthetic fleet, labels waypoints, and trains the ML models.

**Core Classification Logic:**
```python
import xgboost as xgb

# Label waypoints: 1 = contrail formed (energy forcing > 0), 0 = no contrail
df["contrail_formed"] = df["ef"].fillna(0).gt(0).astype(int)

# Define the 5 key atmospheric features
features = ["altitude", "air_temperature", "specific_humidity", "rhi", "latitude"]

# Train the XGBoost Classifier
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train[features], y_train)
```

---

## Contact

**Fiona Diana** – fionadianaofficial@gmail.com

Project Link: [https://github.com/Fiona300/contrail-prediction-model](https://github.com/Fiona300/contrail-prediction-model)
