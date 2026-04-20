"""
model_utils.py
--------------
Shared feature engineering logic used by both:
  - train_and_save.py  (training)
  - main.py            (FastAPI inference)

Keeping this in one place ensures training and inference
use identical transformations.
"""

import numpy as np
import pandas as pd


# ── Constants (must match training) ─────────────────────────────────────────
TEMP_COEFFICIENT = 0.005   # 0.5% power loss per °C above 25°C (crystalline Si)
HIGH_IRR_QUANTILE = 0.90   # Top 10% irradiation used to estimate P_RATED
P_RATED_QUANTILE  = 0.95   # 95th‑percentile DC power at high irradiation = P_RATED

FEATURES_V2 = [
    'EFFICIENCY_RATIO',
    'IRR_NORM_DC',
    'TEMP_DERATING',
    'ROLLING_DEVIATION',
    'HOUR_SIN',
    'HOUR_COS',
]


def engineer_features(df: pd.DataFrame, p_rated_map: dict) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a DataFrame that already contains:
        DATE_TIME, SOURCE_KEY, DC_POWER, AC_POWER, IRRADIATION, MODULE_TEMPERATURE

    Parameters
    ----------
    df : pd.DataFrame
        Merged and daylight-filtered plant data.
    p_rated_map : dict
        {SOURCE_KEY: P_RATED_value} — computed during training and saved
        alongside the model so inference uses identical rated-power values.

    Returns
    -------
    pd.DataFrame with all FEATURES_V2 columns added.
    """
    df = df.copy()

    # 1. AC/DC efficiency ratio
    df['EFFICIENCY_RATIO'] = df['AC_POWER'] / df['DC_POWER']

    # 2. Irradiance-normalised DC power
    df['IRR_NORM_DC'] = df['DC_POWER'] / df['IRRADIATION']

    # 3. Temperature derating
    df['P_RATED']      = df['SOURCE_KEY'].map(p_rated_map)
    df['EXPECTED_DC']  = df['P_RATED'] * (
        1 - TEMP_COEFFICIENT * (df['MODULE_TEMPERATURE'] - 25)
    )
    df['TEMP_DERATING'] = (
        (df['EXPECTED_DC'] - df['DC_POWER']) / df['P_RATED']
    ) * 100

    # 4. Rolling 1-hour deviation (requires datetime index per inverter)
    df = df.sort_values(['SOURCE_KEY', 'DATE_TIME'])
    df['ROLLING_DC_MEAN'] = (
        df.set_index('DATE_TIME')
          .groupby('SOURCE_KEY')['DC_POWER']
          .transform(lambda x: x.rolling('1h').mean())
          .values
    )
    df['ROLLING_DEVIATION'] = df['DC_POWER'] - df['ROLLING_DC_MEAN']

    # 5. Cyclic hour encoding
    df['HOUR']     = df['DATE_TIME'].dt.hour
    df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)

    return df


def compute_p_rated(df: pd.DataFrame) -> dict:
    """
    Estimate rated DC power per inverter from training data.
    Uses the 95th-percentile DC output during the top-10% irradiation readings.

    Parameters
    ----------
    df : pd.DataFrame
        Daylight-filtered plant data with DC_POWER, IRRADIATION, SOURCE_KEY.

    Returns
    -------
    dict : {SOURCE_KEY: float}
    """
    high_irr_threshold = df['IRRADIATION'].quantile(HIGH_IRR_QUANTILE)
    p_rated = (
        df[df['IRRADIATION'] >= high_irr_threshold]
        .groupby('SOURCE_KEY')['DC_POWER']
        .quantile(P_RATED_QUANTILE)
        .to_dict()
    )
    return p_rated
