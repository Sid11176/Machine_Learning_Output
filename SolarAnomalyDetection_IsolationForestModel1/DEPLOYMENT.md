# Deployment Guide — Solar Inverter Anomaly Detection API

This guide walks you through setting up and running the FastAPI inference service locally, from a fresh clone of the repository.

---

## Prerequisites

- Python 3.9 or higher
- pip
- The two CSV files from Kaggle placed in the project root:
  - `Plant_1_Generation_data.csv`
  - `Plant_1_Weather_Sensor_Data.csv`

Download the dataset here: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/solar-anomaly-detection.git
cd solar-anomaly-detection
```

---

## Step 2 — Create a Virtual Environment

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Place the Dataset Files

Copy both CSV files into the project root (same folder as `README.md`):

```
solar-anomaly-detection/
├── Plant_1_Generation_data.csv       ← place here
├── Plant_1_Weather_Sensor_Data.csv   ← place here
├── README.md
├── requirements.txt
...
```

---

## Step 5 — Train and Save the Model

This step reads the CSV files, engineers all features, trains the Isolation Forest V2 model, and saves three artifacts into the `models/` folder.

```bash
python api/train_and_save.py
```

Expected output:

```
Loading data...
Rows after daylight filter and cleaning: XXXXX
Computing P_RATED per inverter...
Engineering features...
Rows used for training: XXXXX
Features: ['EFFICIENCY_RATIO', 'IRR_NORM_DC', 'TEMP_DERATING', 'ROLLING_DEVIATION', 'HOUR_SIN', 'HOUR_COS']

Training Isolation Forest V2 (contamination=0.0157)...
Anomalies flagged: XXX (1.57%)

Artifacts saved:
  models/isolation_forest_v2.pkl
  models/scaler_v2.pkl
  models/p_rated_map.pkl

Training complete. You can now start the API.
```

You only need to run this **once**. The saved artifacts are reloaded every time the API starts.

---

## Step 6 — Start the API Server

```bash
uvicorn api.main:app --reload
```

The `--reload` flag automatically restarts the server when you edit source files. Remove it in production.

Expected output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

## Step 7 — Test the API

### Option A: Interactive Docs (Recommended for first use)

Open your browser and go to:

```
http://127.0.0.1:8000/docs
```

This opens the auto-generated Swagger UI where you can test all endpoints interactively without writing any code.

---

### Option B: Health Check

```bash
curl http://127.0.0.1:8000/
```

Response:

```json
{
  "status": "online",
  "model": "Isolation Forest V2",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

### Option C: Single Reading Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date_time": "2020-05-15 10:15:00",
    "source_key": "sjndEbLyjtCKgGv",
    "dc_power": 8500.0,
    "ac_power": 8330.0,
    "irradiation": 0.75,
    "module_temperature": 42.0
  }'
```

Response:

```json
{
  "source_key": "sjndEbLyjtCKgGv",
  "date_time": "2020-05-15 10:15:00",
  "is_anomaly": false,
  "anomaly_label": 1,
  "anomaly_score": -0.082341,
  "features_used": {
    "EFFICIENCY_RATIO": 0.979882,
    "IRR_NORM_DC": 11333.333333,
    "TEMP_DERATING": 2.941176,
    "ROLLING_DEVIATION": 0.0,
    "HOUR_SIN": 0.866025,
    "HOUR_COS": 0.5
  }
}
```

**Field reference:**

| Field | Description |
|---|---|
| `is_anomaly` | `true` = flagged as anomaly |
| `anomaly_label` | `-1` = anomaly, `1` = normal (raw Isolation Forest output) |
| `anomaly_score` | Lower (more negative) = more anomalous |
| `features_used` | Engineered feature values used for prediction |

---

### Option D: Batch Prediction

Send multiple readings as a JSON array. For accurate `ROLLING_DEVIATION`, readings should be in **chronological order per inverter**.

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "date_time": "2020-05-15 06:00:00",
      "source_key": "sjndEbLyjtCKgGv",
      "dc_power": 120.0,
      "ac_power": 117.0,
      "irradiation": 0.05,
      "module_temperature": 22.0
    },
    {
      "date_time": "2020-05-15 10:15:00",
      "source_key": "sjndEbLyjtCKgGv",
      "dc_power": 8500.0,
      "ac_power": 8330.0,
      "irradiation": 0.75,
      "module_temperature": 42.0
    }
  ]'
```

Response includes a summary plus individual results:

```json
{
  "total_readings": 2,
  "total_anomalies": 1,
  "flag_rate_pct": 50.0,
  "results": [ ... ]
}
```

---

## Stopping the Server

Press `CTRL + C` in the terminal where uvicorn is running.

---

## Important Notes

### On `ROLLING_DEVIATION` for single readings
The rolling deviation feature requires a time-series context (the inverter's DC power history over the past hour). For the `/predict` single-reading endpoint, this is set to `0.0` (neutral). This means the model relies on the remaining 5 features for single-point inference, which is still valid — rolling deviation is one of six features.

For the most accurate predictions, use `/predict/batch` with a chronological sequence of readings per inverter.

### On unknown inverters
If you send a `source_key` not seen during training, the API falls back to the fleet mean `P_RATED` for temperature derating. All other features are computed directly from the reading values.

### On contamination rate
The model was trained with `contamination=0.0157` (1.57%), meaning approximately 1 in 64 readings will be flagged. This was set to match the statistical baseline flag rate, not from domain knowledge of actual fault frequency.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: Model artifacts not found` | `train_and_save.py` has not been run | Run `python api/train_and_save.py` first |
| `FileNotFoundError: Plant_1_Generation_data.csv` | CSV not in project root | Place both CSVs in the project root folder |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `Address already in use` | Port 8000 is taken | Use `--port 8001` flag: `uvicorn api.main:app --port 8001` |
