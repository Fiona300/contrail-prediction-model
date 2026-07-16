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

1. **`01_contrail_conditions.ipynb`**: Fetches ERA5 data and visualizes ISSR coverage.
2. **`02_contrail_prediction.ipynb`**: Runs a CoCiP simulation on a single flight path.
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


## Model Accuracy and Evaluation

### CoCiP Sensitivity Analysis (Notebook 2)

CoCiP was run on flight AAL1158 (A320, ~3 hours, Texas to Virginia) on March 1, 2022, with 13 route variations:

| Variation | Segments | Total EF | vs Baseline |
|-----------|----------|----------|-------------|
| Baseline (~FL324) | 193 | 1.68×10¹³ | — |
| +1000 ft | 24 | 4.43×10¹¹ | **-97%** |
| +2000 ft | 55 | 5.19×10¹² | -69% |
| +1.0° lat | 9 | 1.18×10¹⁰ | **-99.9%** |
| +2.0° lat | 0 | 0 | **-100%** |
| −1000 ft | 188 | 7.19×10¹² | -57% |
| −2000 ft | 2,115 | 1.01×10¹⁵ | **+5900%** |

**Key insight:** Upward altitude shifts and northward lateral shifts reduce contrail energy forcing by 97–100%. Downward shifts can dramatically increase impact. Fuel burn trade-offs were not quantified in this analysis.

---

### Machine Learning Classifier (Notebook 3)

A Random Forest classifier was trained on 55 CoCiP-labeled flights to predict contrail formation from atmospheric variables:

| Model | AUC-ROC | Accuracy |
|-------|---------|----------|
| Random Forest | **0.996** | ~99% |
| XGBoost | 0.993 | ~99% |

**Features:** altitude, air temperature, specific humidity, RHI, latitude  
**Top predictor:** Relative Humidity over Ice (RHI) — confirms physical understanding

**Validation:** 80/20 random train-test split. High AUC indicates the model successfully learns the thermodynamic threshold for contrail formation.

---

### Evaluation Notes

1. **Class imbalance:** 11.4% positive rate with no SMOTE or class weights. Both models perform well regardless, but this could be explored in future work.

2. **No cross-validation:** Single 80/20 split. k-fold cross-validation would strengthen the AUC claims.

3. **Temporal/spatial leakage risk:** Random split on synthetic variations of one route means train and test share similar geographic and atmospheric patterns. This partially explains the near-perfect AUC.

4. **Single day, single route:** All data is March 1, 2022, along the DFW→IAD corridor. Generalizability to other seasons and regions is untested.

5. **RHI as top predictor:** Confirmed by feature importance plots. Physically sound — relative humidity over ice is the key thermodynamic threshold for persistent contrail formation.
---

## Contact

**Fiona Diana** – fionadianaofficial@gmail.com

Project Link: [https://github.com/Fiona300/contrail-prediction-model](https://github.com/Fiona300/contrail-prediction-model)
