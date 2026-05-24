# 🚦 AcciGuard: Smart Vehicle Accident Severity Prediction

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/ML--Model-LightGBM-blue?style=flat-square)](https://lightgbm.readthedocs.io/)
[![Leaflet.js](https://img.shields.io/badge/Frontend-Leaflet.js-green?style=flat-square&logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![pytest](https://img.shields.io/badge/Tests-pytest-e39738?style=flat-square&logo=pytest&logoColor=white)](https://docs.pytest.org/)

An interactive, high-performance web application designed to predict the severity of road accidents in real-time based on historical UK accident records (2008–2020).

---

## 🌟 Key Features

### 💻 1. Interactive Geospatial Dashboard
* **Leaflet.js Integration**: Pinpoint accident locations globally with a draggable marker that auto-syncs coordinates.
* **Antigravity Theme Engine**: Seamlessly toggle between Dark and Light modes with persistent local storage.
* **Glassmorphic UI**: Sleek, modern design with frosted glass panels and animated progress meters.

### ⚡ 2. High-Performance API Layer
* **FastAPI Core**: Minimal latency with asynchronous routing and automatic Swagger documentation.
* **Strict Validation**: Pydantic guards geographical and numeric limits (e.g., Latitude -90 to 90).

### 🧠 3. Machine Learning Pipeline
* **Bayesian Optimization**: Hyperparameters tuned via **Optuna** over 100+ trials.
* **Class Imbalance Handling**: Custom balanced weights to address the 83.5% "Slight" accident skew.
* **Risk Multipliers**: Compares scenarios against UK national averages (e.g., "5x more likely to be fatal").

---

## 📂 Project Architecture

```text
Road_Accident_Severity_Project/
├── 1-preprocessing.ipynb       # Data cleaning & processing
├── 2-EDA.ipynb                 # Exploratory data analysis
├── 3-Model_Building.ipynb      # Training & Optuna tuning
├── app.py                      # FastAPI backend & API
├── train_and_save_model.py     # Production training script
├── requirements.txt            # Project dependencies
├── data/                       # CSV datasets (ignored in git)
├── models/                     # Serialized production assets (.joblib)
├── src/                        # Core logic modules
├── static/                     # Frontend (HTML/CSS/JS)
└── tests/                      # Automated pytest suites
