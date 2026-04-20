"""
train_and_save.py
-----------------
Trains the Isolation Forest V2 model on Plant 1 data and saves
the model artifacts to the models/ directory.

Run this script once before starting the API:
    python api/train_and_save.py

Output
------
    models/isolation_forest_v2.pkl
    models/scaler_v2.pkl
    models/p_rated_map.pkl
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Allow import from api/ when running from project root
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import engineer_features, compute_p_rated, FEATURES_V2

# ── Config ───────────────────────────────────────────────────────────────────
CONTAMINATION = 0.0157   # Matches baseline fleet flag rate (1.57%)
N_ESTIMATORS  = 100
RANDOM_STATE  = 42

DATA_DIR   = '.'          # Directory containing the CSV files
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_and_prepare(data_dir: str) -> pd.DataFrame:
    gen  = pd.read_csv(
        os.path.join(data_dir, 'Plant_1_Generation_data.csv'),
        parse_dates=['DATE_TIME']
    )
    sens = pd.read_csv(
        os.path.join(data_dir, 'Plant_1_Weather_Sensor_Data.csv'),
        parse_dates=['DATE_TIME']
    )

    # Merge
    merged = pd.merge(gen, sens, on=['DATE_TIME', 'PLANT_ID'], how='left')
    merged = merged.drop(columns=['SOURCE_KEY_y'])
    merged = merged.rename(columns={'SOURCE_KEY_x': 'SOURCE_KEY'})

    # Daylight filter
    daylight = merged[
        (merged['DC_POWER'] > 0) &
        (merged['IRRADIATION'] > 0)
    ].copy()

    # Drop rows with missing sensor readings
    daylight = daylight.dropna(subset=['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE'])

    print(f'Rows after daylight filter and cleaning: {len(daylight)}')
    return daylight


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print('Loading data...')
    df = load_and_prepare(DATA_DIR)

    print('Computing P_RATED per inverter...')
    p_rated_map = compute_p_rated(df)

    print('Engineering features...')
    df = engineer_features(df, p_rated_map)

    # Drop rows where any V2 feature is NaN
    df_model = df[FEATURES_V2 + ['SOURCE_KEY', 'DATE_TIME']].dropna()
    print(f'Rows used for training: {len(df_model)}')
    print(f'Features: {FEATURES_V2}')

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[FEATURES_V2])

    # Train
    print(f'\nTraining Isolation Forest V2 (contamination={CONTAMINATION})...')
    model = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE
    )
    model.fit(X_scaled)

    preds = model.predict(X_scaled)
    n_anomalies = (preds == -1).sum()
    print(f'Anomalies flagged: {n_anomalies} ({n_anomalies / len(df_model) * 100:.2f}%)')

    # Save artifacts
    model_path    = os.path.join(MODELS_DIR, 'isolation_forest_v2.pkl')
    scaler_path   = os.path.join(MODELS_DIR, 'scaler_v2.pkl')
    p_rated_path  = os.path.join(MODELS_DIR, 'p_rated_map.pkl')

    joblib.dump(model,      model_path)
    joblib.dump(scaler,     scaler_path)
    joblib.dump(p_rated_map, p_rated_path)

    print(f'\nArtifacts saved:')
    print(f'  {model_path}')
    print(f'  {scaler_path}')
    print(f'  {p_rated_path}')
    print('\nTraining complete. You can now start the API.')


if __name__ == '__main__':
    main()
