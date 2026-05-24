import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

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

def main():
    print("=== ROAD ACCIDENT SEVERITY: MODEL TRAINING & SERIALIZATION ===")
    
    os.makedirs('models', exist_ok=True)
    
    csv_path = 'data/Cleaned_Accident_data_2008_2020.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Cleaned dataset not found at {csv_path}. Please make sure the CSV exists.")
        return
        
    print(f"Loading cleaned dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded! Shape: {df.shape}")
    
    if 'accident_severity' in df.columns:
        df['target'] = df['accident_severity'] - 1
    elif 'accident_seriousness' in df.columns:
        df['target'] = df['accident_seriousness'].map({'Fatal': 0, 'Serious': 1, 'Not Serious': 2})
    else:
        raise ValueError("Could not find target column.")
        
    y = df['target'].astype(int)

    cols_to_drop = [
        'accident_severity', 'accident_seriousness', 'target', 
        'accident_datetime', 'season', 'accident_year', 'accident_month'
    ]
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    for col in NUMERIC_COLS:
        if col in X.columns and X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"Imputed missing values for numeric column '{col}' with median: {median_val}")

    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype(str)

    print("Fitting and transforming StandardScaler on numeric features...")
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[NUMERIC_COLS] = scaler.fit_transform(X[NUMERIC_COLS])

    print("Performing One-Hot Encoding on categorical features...")
    X_encoded = pd.get_dummies(X_scaled, columns=CATEGORICAL_COLS, drop_first=True)

    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'float64':
            X_encoded[col] = X_encoded[col].astype('float32')
        elif X_encoded[col].dtype == 'int64':
            X_encoded[col] = X_encoded[col].astype('int32')
            
    model_features = X_encoded.columns.tolist()
    print(f"One-Hot Encoding completed! Total features: {len(model_features)}")
    
    best_params = {
        'n_estimators': 140,
        'learning_rate': 0.0927309993184637,
        'num_leaves': 60,
        'min_child_samples': 35,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    print("\nTraining final Optuna-Tuned LightGBM Classifier on the entire dataset...")
    model = LGBMClassifier(**best_params)
    model.fit(X_encoded, y)
    print("Model training successfully completed!")
    
    print("\nSerializing fitted assets to 'models/' folder...")
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(model_features, 'models/model_features.joblib')
    joblib.dump(model, 'models/lightgbm_model.joblib')
    
    print("=== ALL ASSETS SUCCESSFULLY SERIALIZED ===")
    print("Files saved:")
    print(" - models/scaler.joblib (Fitted StandardScaler)")
    print(" - models/model_features.joblib (One-Hot Encoded Features List)")
    print(" - models/lightgbm_model.joblib (Trained LightGBM Model)")
    
if __name__ == '__main__':
    main()
