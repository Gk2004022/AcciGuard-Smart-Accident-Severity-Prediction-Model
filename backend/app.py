import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path 
# Define unscaled numeric medians and categorical defaults
DEFAULT_NUMERICS = {
    'police_force': 30.0,
    'local_authority_district': 322.0,
    'first_road_number': 38.0,
    'second_road_number': 0.0
}

DEFAULT_CATEGORICALS = {
    'junction_control': '30',
    'second_road_class': '-1',
    'pedestrian_crossing_human_control': '0',
    'pedestrian_crossing_physical_facilities': '0',
    'special_conditions_at_site': '0',
    'carriageway_hazards': '0',
    'did_police_officer_attend_scene_of_accident': '1',
    'trunk_road_flag': '2'
}

CATEGORICAL_COLS = [
    'number_of_vehicles', 'number_of_casualties', 'day_of_week', 
    'first_road_class', 'road_type', 'junction_detail', 'junction_control', 
    'second_road_class', 'pedestrian_crossing_human_control', 
    'pedestrian_crossing_physical_facilities', 'light_conditions', 
    'weather_conditions', 'road_surface_conditions', 'special_conditions_at_site', 
    'carriageway_hazards', 'urban_or_rural_area', 'did_police_officer_attend_scene_of_accident', 
    'trunk_road_flag', 'is_weekend', 'time_of_day'
]

NUMERIC_COLS = ['longitude', 'latitude', 'speed_limit', 'accident_hour']

# Pydantic schema for validation
class PredictionPayload(BaseModel):
    latitude: float = Field(..., description="Geographic Latitude", ge=-90.0, le=90.0)
    longitude: float = Field(..., description="Geographic Longitude", ge=-180.0, le=180.0)
    speed_limit: float = Field(..., description="Speed limit in mph", ge=10.0, le=80.0)
    accident_datetime: str = Field(..., description="ISO 8601 Datetime string (e.g. YYYY-MM-DDTHH:MM)")
    number_of_vehicles: str = Field(..., description="Number of vehicles involved ('1', '2', '3', '4+')")
    number_of_casualties: str = Field(..., description="Number of casualties ('1', '2', '3', '4', '5+')")
    urban_or_rural_area: str = Field(..., description="Area classification ('1' for Urban, '2' for Rural)")
    light_conditions: str = Field(..., description="Light conditions ('1', '4', '5', '6', '7')")
    weather_conditions: str = Field(..., description="Weather conditions ('1' to '9')")
    road_surface_conditions: str = Field(..., description="Road surface conditions ('1' to '9')")
    road_type: str = Field(..., description="Road configuration ('1' to '9')")
    junction_detail: str = Field(..., description="Junction detail ('0' to '99')")
    first_road_class: str = Field(..., description="First road classification ('1' to '6')")

# Initialize FastAPI App
app = FastAPI(
    title="AcciGuard: Smart Vehicle Accident Severity Prediction using Machine Learning",
    description="Real-time predictive API for road accident severity based on historical UK models.",
    version="1.0.0"
)

# Enable CORS for cross-domain requests (allow all origins for seamless external deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Set to False to prevent modern browsers from blocking wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model assets
scaler = None
model_features = None
model = None

# Get the absolute root path where app.py lives
BASE_DIR = Path(__file__).resolve().parent

BASE_DIR = Path(__file__).resolve().parent
@app.on_event("startup")
# BASE_DIR is already /opt/render/project/src/backend

@app.on_event("startup")
def load_assets():
    global scaler, model_features, model
    try:
        print("Loading serialized machine learning assets...")
        
        # 🟢 REMOVE "backend" FROM THE PATH LINKS HERE:
        scaler_path = BASE_DIR / "scaler.joblib"
        features_path = BASE_DIR / "model_features.joblib"
        model_path = BASE_DIR / "lightgbm_model.joblib"
        
        # Load the assets directly from the script's directory
        scaler = joblib.load(scaler_path)
        model_features = joblib.load(features_path)
        model = joblib.load(model_path)
        
        print("Assets successfully loaded! Service ready.")

    except Exception as e:
        print(f"CRITICAL: Failed to load machine learning assets from 'backend/'. Error: {e}")
        print("Please run 'python train_and_save_model.py' to generate the required joblib assets.")

@app.get("/")
async def serve_dashboard():
    """Serves the main dashboard page."""
    index_path = "frontend/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AcciGuard: Smart Vehicle Accident Severity Prediction using Machine Learning API is Active! Static files folder 'frontend/' not found."}

@app.post("/predict")
async def predict_severity(payload: PredictionPayload):
    """
    Receives accident characteristics, preprocesses, scales, encodes, 
    and predicts probabilities for accident severity.
    """
    global scaler, model_features, model
    
    if scaler is None or model_features is None or model is None:
        raise HTTPException(
            status_code=503, 
            detail="Machine learning model assets are not loaded on the server. Please check logs."
        )
        
    try:
        # 1. Parse temporal features from datetime string
        dt = pd.to_datetime(payload.accident_datetime)
        accident_hour = float(dt.hour)
        
        # Day of week conversion: Pandas 0 (Mon) -> 6 (Sun). UK Accident Data: 1 (Sun) -> 7 (Sat)
        day_of_week_val = int(((dt.dayofweek + 1) % 7) + 1)
        is_weekend_val = "1" if day_of_week_val in [1, 7] else "0"
        
        # Time of day binning: Morning [5-12], Afternoon [12-17], Evening [17-21], Night [21-5]
        h = dt.hour
        if 5 <= h < 12:
            time_of_day_val = "Morning"
        elif 12 <= h < 17:
            time_of_day_val = "Afternoon"
        elif 17 <= h < 21:
            time_of_day_val = "Evening"
        else:
            time_of_day_val = "Night"
            
        # 2. Build complete raw features dictionary including defaulted columns
        input_data = {
            # Scaled Numerics
            'latitude': payload.latitude,
            'longitude': payload.longitude,
            'speed_limit': payload.speed_limit,
            'accident_hour': accident_hour,
            
            # Unscaled Numeric Defaults
            'police_force': DEFAULT_NUMERICS['police_force'],
            'local_authority_district': DEFAULT_NUMERICS['local_authority_district'],
            'first_road_number': DEFAULT_NUMERICS['first_road_number'],
            'second_road_number': DEFAULT_NUMERICS['second_road_number'],
            
            # User Categoricals (Casted to String)
            'number_of_vehicles': str(payload.number_of_vehicles),
            'number_of_casualties': str(payload.number_of_casualties),
            'urban_or_rural_area': str(payload.urban_or_rural_area),
            'light_conditions': str(payload.light_conditions),
            'weather_conditions': str(payload.weather_conditions),
            'road_surface_conditions': str(payload.road_surface_conditions),
            'road_type': str(payload.road_type),
            'junction_detail': str(payload.junction_detail),
            'first_road_class': str(payload.first_road_class),
            
            # Derived Categoricals
            'day_of_week': str(day_of_week_val),
            'is_weekend': str(is_weekend_val),
            'time_of_day': str(time_of_day_val),
            
            # Defaulted Categoricals
            'junction_control': DEFAULT_CATEGORICALS['junction_control'],
            'second_road_class': DEFAULT_CATEGORICALS['second_road_class'],
            'pedestrian_crossing_human_control': DEFAULT_CATEGORICALS['pedestrian_crossing_human_control'],
            'pedestrian_crossing_physical_facilities': DEFAULT_CATEGORICALS['pedestrian_crossing_physical_facilities'],
            'special_conditions_at_site': DEFAULT_CATEGORICALS['special_conditions_at_site'],
            'carriageway_hazards': DEFAULT_CATEGORICALS['carriageway_hazards'],
            'did_police_officer_attend_scene_of_accident': DEFAULT_CATEGORICALS['did_police_officer_attend_scene_of_accident'],
            'trunk_road_flag': DEFAULT_CATEGORICALS['trunk_road_flag']
        }
        
        # 3. Convert to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # 4. Scale Numeric Columns using fitted scaler
        df_input[NUMERIC_COLS] = scaler.transform(df_input[NUMERIC_COLS])
        
        # 5. Build robust feature aligned dictionary initialized to 0.0
        input_dict = {feat: 0.0 for feat in model_features}
        
        # Fill numeric features (both scaled and unscaled defaults)
        all_numeric_features = NUMERIC_COLS + ['police_force', 'local_authority_district', 'first_road_number', 'second_road_number']
        for col in all_numeric_features:
            if col in input_dict:
                input_dict[col] = float(df_input.loc[0, col])
                
        # Fill categorical features (One-hot mapping)
        for col in CATEGORICAL_COLS:
            val = str(input_data[col])
            feature_name = f"{col}_{val}"
            if feature_name in input_dict:
                input_dict[feature_name] = 1.0
                
        # Convert to DataFrame
        df_aligned = pd.DataFrame([input_dict])
        
        # Downcast all columns to float32 for model inference compatibility
        for col in df_aligned.columns:
            df_aligned[col] = df_aligned[col].astype('float32')
                
        # 7. Model Inference (probabilities of Fatal [0], Serious [1], Slight [2])
        probs = model.predict_proba(df_aligned)[0]
        
        fatal_prob = float(probs[0])
        serious_prob = float(probs[1])
        slight_prob = float(probs[2])
        
        # Calculate risk multipliers compared to average UK baseline ratios
        # UK baseline: Fatal: 1.25%, Serious: 15.26%, Slight: 83.50%
        fatal_risk_mult = fatal_prob / 0.0125
        serious_risk_mult = serious_prob / 0.1526
        
        # -----------------------------------------------------------------------
        # Custom Severity Prediction Logic (Calibrated Probability Thresholds)
        # -----------------------------------------------------------------------
        # Model probability distribution (observed across diverse scenarios):
        #   Fatal  range: 13% - 79%, mean ~44%
        #   Serious range: 14% - 47%, mean ~28%
        #   Slight  range:  6% - 50%, mean ~28%
        #
        # Threshold rules (priority order — first match wins):
        #   1. FATAL   : fatal_prob >= 0.45  → clearly dominant fatal risk
        #   2. SERIOUS : serious_prob >= 0.38 → meaningful serious risk (not overshadowed by fatal)
        #   3. SLIGHT  : fallback for low-risk scenarios
        # -----------------------------------------------------------------------
        FATAL_THRESHOLD   = 0.45   # Only flag Fatal when it's clearly dominant
        SERIOUS_THRESHOLD = 0.38   # Catch Serious when fatal isn't dominant
        
        if fatal_prob >= FATAL_THRESHOLD:
            predicted_severity = "Fatal"
        elif serious_prob >= SERIOUS_THRESHOLD:
            predicted_severity = "Serious"
        else:
            predicted_severity = "Slight"
        
        return {
            "status": "success",
            "prediction": {
                "severity_class": predicted_severity,
                "probabilities": {
                    "Fatal": round(fatal_prob * 100, 2),
                    "Serious": round(serious_prob * 100, 2),
                    "Slight": round(slight_prob * 100, 2)
                },
                "risk_multipliers": {
                    "Fatal_vs_Average": round(fatal_risk_mult, 1),
                    "Serious_vs_Average": round(serious_risk_mult, 1)
                }
            },
            "meta": {
                "parsed_datetime": str(dt),
                "extracted_hour": accident_hour,
                "extracted_day_of_week": day_of_week_val,
                "extracted_is_weekend": is_weekend_val == "1",
                "extracted_time_of_day": time_of_day_val
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing prediction input: {str(e)}")

# Mount static files folder to serve styles, scripts, and map dependencies
static_dir = "frontend"
if os.path.exists(static_dir):
    app.mount("/frontend", StaticFiles(directory=static_dir), name="frontend")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="root_static")
