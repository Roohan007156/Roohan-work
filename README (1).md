# Grid Demand Forecasting AI — INFINIX'26

A runnable Streamlit prototype that forecasts the next 24 hours of hourly electricity demand and compares two machine-learning models:

1. Random Forest Regressor
2. Gradient Boosting Regressor

It includes a sample synthetic dataset, chronological model evaluation, MAE/RMSE/MAPE metrics, actual-versus-predicted plots, future 24-hour demand forecast, CSV export, and feature importance.

> **Important:** `data/sample_hourly_demand.csv` is synthetic demonstration data, not actual readings from a college or locality. For a real project, collect verified meter readings and use a genuine weather forecast for future temperature/humidity.

## A. Run on Windows

### 1. Install Python

Install Python 3.10 or newer from https://www.python.org/downloads/. During installation, enable **Add Python to PATH**.

### 2. Extract and open the project

Extract `grid_demand_forecasting.zip`. Open the extracted folder. In File Explorer, click the address bar, type `cmd`, and press Enter. This opens Command Prompt in that folder.

### 3. Create a virtual environment (recommended)

```bat
python -m venv .venv
.venv\Scripts\activate
```

If Windows blocks activation, run the commands below without activating the environment by using `.venv\Scripts\python -m pip ...` and `.venv\Scripts\python -m python -m streamlit run app.py`.

### 4. Install libraries

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Start the app

```bat
python -m streamlit run app.py
```

A browser tab should open automatically. If not, copy the local URL shown in the terminal (usually `http://localhost:8501`) into your browser. Keep the terminal running while presenting.

### 6. Demonstrate the system

1. Show historical demand from the included synthetic data.
2. Explain that the app trains and compares Random Forest and Gradient Boosting using a time-ordered train/test split.
3. Point out MAE, RMSE and MAPE, then show the actual-versus-predicted graph.
4. Show the next 24 hourly demand predictions, predicted peak, and downloadable CSV.
5. Explain the feature-importance chart (load lags, time, weather and academic calendar).
6. Toggle the holiday assumption and rerun/update the display to demonstrate a different calendar scenario.

## B. Generate another synthetic dataset (optional)

```bat
python generate_data.py
```

This writes `data/sample_hourly_demand.csv`. The generated records remain synthetic; do not describe them as real campus measurements.

## C. Use your own dataset

In the app sidebar, upload a CSV containing at least eight days of hourly data. Several weeks or months are strongly recommended. Required column names are exact:

| Column | Meaning | Example |
|---|---|---|
| `Timestamp` | Local date and hour | `2026-01-01 13:00:00` |
| `Load_kW` | Measured electricity demand in kW | `245.8` |
| `Temperature_C` | Temperature in °C | `31.2` |
| `Humidity_pct` | Relative humidity (%) | `67` |
| `Is_Holiday` | Optional: 1 for holiday, otherwise 0 | `0` |
| `Academic_Active` | Optional: 1 when campus is active, otherwise 0 | `1` |

`Is_Holiday` and `Academic_Active` are optional; the app estimates weekday academic activity when missing. Rows should represent hourly observations. The app resamples to hourly intervals and interpolates gaps, so review the input data before trusting its predictions.

## D. How the model works

- **Target:** hourly `Load_kW`.
- **Predictors:** hour, weekday, month, day of year, weekend flag, temperature, humidity, holiday/activity flags, load from 1 hour ago, 24 hours ago and 168 hours ago, and rolling mean demand over the previous 24 and 168 hours.
- **Evaluation:** chronological split (first 80% training, final 20% testing) to reduce time-series leakage.
- **Metrics:** MAE, RMSE and MAPE.
- **24-hour forecast:** the chosen model is retrained on all historical feature-ready rows, then predicts recursively. For each future hour, weather is estimated using the matching hour from the previous week; this is a simple fallback, not a real meteorological forecast.
- **Explainability:** tree-model feature importance shows which inputs the model relied on most. Importance is not proof of causation.

## E. Limitations and improvements to mention to judges

1. Replace synthetic data with campus smart-meter readings at hourly or 15-minute intervals.
2. Use a genuine forecast source for future temperature and humidity.
3. Add a verified academic, public-holiday and special-event calendar.
4. Compare against a simple baseline such as “same hour yesterday” and report performance on a held-out period.
5. Test on data from a different month or campus before operational deployment.
6. Forecast in kW (instantaneous/interval average demand); multiply by interval duration to calculate kWh energy for a one-hour interval.

## Project files

- `app.py` — interactive dashboard, model training, comparison and forecast.
- `generate_data.py` — reproducible synthetic dataset generator.
- `data/sample_hourly_demand.csv` — ready-to-use demonstration data.
- `requirements.txt` — Python dependencies.
