# 🚦 AcciGuard: Smart Vehicle Accident Severity Prediction using Machine Learning

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/ML--Model-LightGBM-blue?style=flat-square)](https://lightgbm.readthedocs.io/)
[![Leaflet.js](https://img.shields.io/badge/Frontend-Leaflet.js-green?style=flat-square&logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![pytest](https://img.shields.io/badge/Tests-pytest-e39738?style=flat-square&logo=pytest&logoColor=white)](https://docs.pytest.org/)

An interactive, high-performance web application designed to predict the severity of road accidents in real-time based on historical UK accident records (spanning 2008 to 2020). The application integrates a bayesian-optimized machine learning backend with an advanced glassmorphic geospatial frontend dashboard.

---

## 🌟 Key Features

### 💻 1. Interactive Geospatial Dashboard (Frontend)
- **Interactive Leaflet.js Map**: Pinpoint accident locations globally or search specific UK coordinates. Dragging the marker automatically synchronizes the latitude and longitude inputs.
- **Antigravity Theme Engine (Dark/Light Mode)**: Seamlessly toggle between dark and light themes using a floating, zero-gravity bobbing switcher widget. Icon states (Sun & Moon) spin gracefully, Leaflet map tiles swap dynamically (CartoDB Dark Matter vs. CartoDB Voyager) without page reloads, and preferences are persisted in the browser's local storage with flash prevention.
- **Glassmorphic UI Elements**: A sleek, modern design powered by CSS variables, rich typography (Outfit), linear gradients, frosted glass panels, and interactive input scale highlights.
- **Dynamic Time & Temporal Settings**: Simple ISO datetime inputs auto-derive temporal features like hour of the day, day of the week, weekend flags, and custom time bins (Morning, Afternoon, Evening, Night).
- **Responsive Gauge Bars**: Visually represents prediction probabilities for **Fatal**, **Serious**, and **Slight** categories with custom-styled animated progress meters.

### ⚡ 2. High-Performance API Layer (Backend)
- **FastAPI Core**: Minimal latency, asynchronous routing, and native support for automatic OpenAPI documentation (Swagger UI).
- **Pydantic Strict Validation**: Guards API boundaries by validating geographical limits (Latitude: -90 to 90, Longitude: -180 to 180) and numeric inputs before processing.
- **Automated Feature Engineering**: Dynamically calculates temporal offsets, weekend statuses, and scales coordinates/speed limits using a pre-saved scaling pipeline.

### 🧠 3. Advanced Machine Learning Pipeline
- **Bayesian Optimization via Optuna**: LightGBM hyperparameters tuned extensively over 100+ trials to identify optimal tree structures, node parameters, and learning speeds.
- **Addressing Class Imbalance**: Incorporates custom balanced class-weight metrics to counteract standard predictive bias on heavily skewed data (UK accidents are 83.5% Slight, 15.2% Serious, 1.3% Fatal).
- **One-Hot Alignment**: Automatically aligns incoming API requests with the exact 101 one-hot encoded features expected by the serialized LightGBM model.
- **Elevated Risk Multipliers**: Computes risk multipliers compared against UK national average baselines, highlighting if a scenario is e.g. 5x more likely to be fatal than the average UK accident.

---

## 📂 Project Architecture

The workspace is organized logically as follows:

```text
Road_Accident_Severity_Project/
├── 1-preprocessing.ipynb          # Jupyter notebook for initial data cleaning & processing
├── 2-EDA.ipynb                    # Notebook for exploratory data analysis and statistical testing
├── 3-Model_Building.ipynb        # Model building, hyperparameter tuning, and Optuna study
├── app.py                         # FastAPI backend serving the dashboard & /predict API
├── train_and_save_model.py        # Executable script to train and serialize the production ML assets
├── requirements.txt               # Pinpointed project dependencies
├── data/                          # Folder for housing cleaned CSV datasets (ignored in git)
├── models/                        # Serialized production assets
│   ├── lightgbm_model.joblib      # Serialized class-weighted LightGBM model
│   ├── model_features.joblib      # List of aligned One-Hot Encoded features
│   └── scaler.joblib              # Serialized StandardScaler fitted on numerics
├── src/                           # Core source modules
│   ├── model_pipeline.py          # Preprocessing functions & LightGBM helper modules
│   └── statistical_tests.py       # Chi-square and ANOVA statistical tests
├── static/                        # Frontend assets
│   ├── index.html                 # Glassmorphic user interface, Leaflet integration, and floating Antigravity Theme Switcher widget
│   ├── script.js                  # Frontend client handlers, map sync, API caller, and dynamic map tile theme swapping
│   └── style.css                  # Custom stylesheet with custom HSL styling, dark/light mode CSS variable tokens, and bobbing toggle button animations
└── tests/                         # Production test suites
    └── test_api.py                # Automated pytest assertions (endpoints, inputs, mock tests)
```

**Note on Temporary Files:**
Any files generated during diagnostics or by AI coding assistants (such as `probe_model.py`, `validate_predictions.py`, or `.md` artifact files like `implementation_plan.md` or `walkthrough.md`) are temporary and can be safely deleted at any time without harming the project. The core architecture relies strictly on the files listed above.

---

## 🚀 Installation & Quick-Start Guide

Follow these steps to set up and run the project locally on your system:

### 1. Clone & Navigate
Ensure you are in the project's root directory:
```bash
cd Road_Accident_Severity_Project
```

### 2. Set Up a Virtual Environment (Recommended)
Create and activate a clean Python virtual environment to manage dependencies:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries including LightGBM, FastAPI, Uvicorn, and Scikit-Learn:
```bash
pip install -r requirements.txt
```

### 4. Train & Save the Machine Learning Model
If you do not have the serialized `.joblib` files in the `models/` directory, run the script to preprocess the data, train the tuned model on the dataset, and serialize the assets:
```bash
python train_and_save_model.py
```
*Note: This script requires `data/Cleaned_Accident_data_2008_2020.csv` to be in place. Once run, it will generate three files under `models/`.*

### 5. Launch the Web Application Server
Launch the high-performance FastAPI server locally:
```bash
uvicorn app:app --reload
```
You should see:
```text
Loading serialized machine learning assets...
Assets successfully loaded! Service ready.
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 6. Open the Dashboard
Open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🔌 API Reference (`/predict`)

The backend exposes a highly robust predictive endpoint that can be queried by any external service or mobile app.

### Endpoint Details
* **Method**: `POST`
* **Path**: `/predict`
* **Content-Type**: `application/json`

### Example Request Body
```json
{
  "latitude": 51.5074,
  "longitude": -0.1278,
  "speed_limit": 30.0,
  "accident_datetime": "2026-05-18T17:30",
  "number_of_vehicles": "2",
  "number_of_casualties": "1",
  "urban_or_rural_area": "1",
  "light_conditions": "1",
  "weather_conditions": "1",
  "road_surface_conditions": "1",
  "road_type": "6",
  "junction_detail": "0",
  "first_road_class": "6"
}
```

### Example Response Body
```json
{
  "status": "success",
  "prediction": {
    "severity_class": "Slight",
    "probabilities": {
      "Fatal": 1.12,
      "Serious": 16.48,
      "Slight": 82.40
    },
    "risk_multipliers": {
      "Fatal_vs_Average": 0.9,
      "Serious_vs_Average": 1.1
    }
  },
  "meta": {
    "parsed_datetime": "2026-05-18 17:30:00",
    "extracted_hour": 17.0,
    "extracted_day_of_week": 2,
    "extracted_is_weekend": false,
    "extracted_time_of_day": "Evening"
  }
}
```

### 🛠️ Interactive Documentation (Swagger UI)
For interactive testing, view detailed schemas, and try out requests directly through your browser at:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 🧪 Testing & Verification

The project includes an automated unit test suite. Run tests using `pytest` to guarantee system stability and verify validation limits:

```bash
# Run all tests
pytest

# Run tests with detailed console print outputs
pytest -v
```

Tests validate:
1. **Home Endpoint Routing**: Verifies that the root path correctly serves static HTML/CSS files.
2. **Prediction Pipeline**: Verifies a valid payload returns exact class keys (`Fatal`, `Serious`, `Slight`) summing to ~100%.
3. **Pydantic Validation**: Asserts that sending an invalid latitude (e.g. `150.0`) triggers a `422 Unprocessable Entity` response.

---

## ⚙️ Model Hyperparameters (Bayesian-Tuned)
The underlying `LightGBM` model is optimized using the following parameters derived from automated Optuna trials:
- `n_estimators`: `140`
- `learning_rate`: `0.0927`
- `num_leaves`: `60`
- `min_child_samples`: `35`
- `class_weight`: `balanced` (specifically handles heavy class representation skew)
- `random_state`: `42`
#   A c c i G u a r d - S m a r t - A c c i d e n t - S e v e r i t y - P r e d i c t i o n - M o d e l  
 #   A c c i G u a r d - S m a r t - A c c i d e n t - S e v e r i t y - P r e d i c t i o n - M o d e l  
 