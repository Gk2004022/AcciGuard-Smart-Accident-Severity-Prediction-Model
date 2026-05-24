# 🚦 AcciGuard: Smart Vehicle Accident Severity Prediction using Machine Learning

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/ML--Model-LightGBM-blue?style=flat-square)](https://lightgbm.readthedocs.io/)
[![Leaflet.js](https://img.shields.io/badge/Frontend-Leaflet.js-green?style=flat-square&logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![pytest](https://img.shields.io/badge/Tests-pytest-e39738?style=flat-square&logo=pytest&logoColor=white)](https://docs.pytest.org/)

An interactive, high-performance web application designed to predict the severity of road accidents in real-time based on historical UK accident records (spanning 2008 to 2020). The application integrates a bayesian-optimized machine learning backend with an advanced glassmorphic geospatial frontend dashboard.

---

## 🌟 Key Features

### 💻 1. Interactive Geospatial Dashboard (Frontend)
* **Interactive Leaflet.js Map**: Pinpoint accident locations globally. Dragging the marker automatically synchronizes the latitude and longitude inputs.
* **Antigravity Theme Engine**: Seamlessly toggle between dark and light themes using a floating, zero-gravity bobbing switcher widget. Icon states spin gracefully and map tiles swap dynamically.
* **Glassmorphic UI Elements**: A sleek, modern design powered by CSS variables, rich typography (Outfit), and frosted glass panels.
* **Dynamic Time & Temporal Settings**: Auto-derives temporal features like hour of the day, day of the week, weekend flags, and custom time bins (Morning, Afternoon, etc.).
* **Responsive Gauge Bars**: Visually represents prediction probabilities for **Fatal**, **Serious**, and **Slight** categories with custom animated progress meters.

### ⚡ 2. High-Performance API Layer (Backend)
* **FastAPI Core**: Minimal latency, asynchronous routing, and native support for automatic OpenAPI documentation (Swagger UI).
* **Pydantic Strict Validation**: Guards API boundaries by validating geographical limits (Latitude: -90 to 90, Longitude: -180 to 180) before processing.
* **Automated Feature Engineering**: Dynamically calculates temporal offsets and scales coordinates using a pre-saved scaling pipeline.

### 🧠 3. Advanced Machine Learning Pipeline
* **Bayesian Optimization via Optuna**: LightGBM hyperparameters tuned extensively over 100+ trials to identify optimal tree structures and learning speeds.
* **Addressing Class Imbalance**: Incorporates custom balanced class-weight metrics to counteract standard predictive bias on heavily skewed data.
* **Elevated Risk Multipliers**: Computes risk multipliers compared against UK national average baselines.

---

## 📂 Project Architecture

```text
Road_Accident_Severity_Project/
├── 1-preprocessing.ipynb          # Jupyter notebook for initial data cleaning
├── 2-EDA.ipynb                    # Notebook for exploratory data analysis
├── 3-Model_Building.ipynb         # Model building & Optuna study
├── app.py                         # FastAPI backend serving the dashboard & API
├── train_and_save_model.py        # Script to train and serialize ML assets
├── requirements.txt               # Project dependencies
├── data/                          # Folder for housing CSV datasets (ignored in git)
├── models/                        # Serialized production assets
│   ├── lightgbm_model.joblib      # Serialized LightGBM model
│   ├── model_features.joblib      # List of aligned features
│   └── scaler.joblib              # Serialized StandardScaler
├── src/                           # Core source modules
│   ├── model_pipeline.py          # Preprocessing & helper modules
│   └── statistical_tests.py       # Chi-square and ANOVA tests
├── static/                        # Frontend assets
│   ├── index.html                 # UI & Leaflet integration
│   ├── script.js                  # Frontend client handlers & API caller
│   └── style.css                  # Custom styling & Dark mode tokens
└── tests/                         # Production test suites
    └── test_api.py                # Automated pytest assertions

Installation Guide
1. Clone & Navigate
  cd Road_Accident_Severity_Project

2. Set Up a Virtual Environment (Recommended)
  # Windows
  python -m venv venv
  venv\Scripts\activate

  # macOS / Linux
  python3 -m venv venv
  source venv/bin/activate

3. Install Dependencies
  pip install -r requirements.txt

4. Train & Save the Machine Learning Model
  Note: This script requires data/Cleaned_Accident_data_2008_2020.csv to be in place.
  python train_and_save_model.py

5. Launch the Web Application Server
  uvicorn app:app --reload
  Navigate to: 👉 http://127.0.0.1:8000


