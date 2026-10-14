import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET = "Load_kW"
WEATHER = ["Temperature_C", "Humidity_pct"]
CALENDAR = ["Is_Holiday", "Academic_Active"]
FEATURES = [
    "Hour", "DayOfWeek", "DayOfYear", "Month", "IsWeekend",
    "Temperature_C", "Humidity_pct", "Is_Holiday", "Academic_Active",
    "Load_lag_1", "Load_lag_24", "Load_lag_168",
    "Load_rolling_mean_24", "Load_rolling_mean_168",
]


def normalise_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Check and standardise user or sample data to an hourly time series."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = ["Timestamp", TARGET, *WEATHER]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required CSV column(s): " + ", ".join(missing))

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for col in [TARGET, *WEATHER, *CALENDAR]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Timestamp", TARGET]).sort_values("Timestamp")
    if df.empty:
        raise ValueError("No valid timestamp and load rows were found.")

    df = df.set_index("Timestamp")
    aggregation = {TARGET: "mean", **{c: "mean" for c in WEATHER}}
    for col in CALENDAR:
        if col in df.columns:
            aggregation[col] = "max"
    df = df.groupby(level=0).agg(aggregation).sort_index().resample("h").mean()
    df[TARGET] = df[TARGET].interpolate(method="time").ffill().bfill()
    for col in WEATHER:
        df[col] = df[col].interpolate(method="time").ffill().bfill()
    if "Is_Holiday" not in df.columns:
        df["Is_Holiday"] = 0
    else:
        df["Is_Holiday"] = df["Is_Holiday"].fillna(0).round().clip(0, 1)
    if "Academic_Active" not in df.columns:
        df["Academic_Active"] = ((df.index.dayofweek < 5) & (df["Is_Holiday"] == 0)).astype(int)
    else:
        df["Academic_Active"] = df["Academic_Active"].fillna(0).round().clip(0, 1)
    df = df.reset_index().rename(columns={"index": "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if len(df) < 24 * 8:
        raise ValueError("Please provide at least 8 days of hourly data. A few months is better.")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("Timestamp").reset_index(drop=True)
    ts = pd.to_datetime(d["Timestamp"])
    d["Hour"] = ts.dt.hour
    d["DayOfWeek"] = ts.dt.dayofweek
    d["DayOfYear"] = ts.dt.dayofyear
    d["Month"] = ts.dt.month
    d["IsWeekend"] = (ts.dt.dayofweek >= 5).astype(int)
    d["Load_lag_1"] = d[TARGET].shift(1)
    d["Load_lag_24"] = d[TARGET].shift(24)
    d["Load_lag_168"] = d[TARGET].shift(168)
    past_load = d[TARGET].shift(1)
    d["Load_rolling_mean_24"] = past_load.rolling(24).mean()
    d["Load_rolling_mean_168"] = past_load.rolling(168).mean()
    return d


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    nonzero = np.abs(actual) > 1e-6
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    return {"MAE (kW)": mae, "RMSE (kW)": rmse, "MAPE (%)": mape}


def make_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=140, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.05, max_depth=3,
            loss="squared_error", random_state=42
        ),
    }


def train_and_compare(df: pd.DataFrame):
    feat = create_features(df).dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    if len(feat) < 100:
        raise ValueError("Not enough usable rows after creating lag features. Upload more history.")
    split = int(len(feat) * 0.8)
    train, test = feat.iloc[:split], feat.iloc[split:]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    reports, predictions = [], {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        reports.append({"Model": name, **calculate_metrics(y_test, pred)})
        predictions[name] = pred

    report_df = pd.DataFrame(reports).sort_values("RMSE (kW)").reset_index(drop=True)
    best_name = report_df.loc[0, "Model"]
    final_models = make_models()
    for model in final_models.values():
        model.fit(feat[FEATURES], feat[TARGET])
    validation = pd.DataFrame({
        "Timestamp": test["Timestamp"].to_numpy(),
        "Actual": y_test.to_numpy(),
        "Predicted": predictions[best_name],
    })
    return report_df, best_name, final_models, validation


def forecast_next_24_hours(df: pd.DataFrame, model, holiday_override: bool, classes_on_weekdays: bool) -> pd.DataFrame:
    """Recursive forecast; future weather is approximated from the same hour last week."""
    history = df.copy().sort_values("Timestamp").reset_index(drop=True)
    history["Timestamp"] = pd.to_datetime(history["Timestamp"])
    last_ts = history["Timestamp"].iloc[-1]
    future_times = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=24, freq="h")
    output = []
    weather_lookup = history.set_index("Timestamp")
    load_history = history[TARGET].astype(float).tolist()
    for ts in future_times:
        reference_ts = ts - pd.Timedelta(days=7)
        if reference_ts in weather_lookup.index:
            ref = weather_lookup.loc[reference_ts]
            if isinstance(ref, pd.DataFrame):
                ref = ref.iloc[-1]
            temp = float(ref["Temperature_C"])
            humidity = float(ref["Humidity_pct"])
        else:
            temp = float(history["Temperature_C"].iloc[-1])
            humidity = float(history["Humidity_pct"].iloc[-1])

        is_holiday = int(holiday_override)
        is_weekend = int(ts.dayofweek >= 5)
        academic_active = int(classes_on_weekdays and ts.dayofweek < 5 and not is_holiday)
        row = {
            "Hour": ts.hour,
            "DayOfWeek": ts.dayofweek,
            "DayOfYear": ts.dayofyear,
            "Month": ts.month,
            "IsWeekend": is_weekend,
            "Temperature_C": temp,
            "Humidity_pct": humidity,
            "Is_Holiday": is_holiday,
            "Academic_Active": academic_active,
            "Load_lag_1": load_history[-1],
            "Load_lag_24": load_history[-24] if len(load_history) >= 24 else load_history[-1],
            "Load_lag_168": load_history[-168] if len(load_history) >= 168 else load_history[0],
            "Load_rolling_mean_24": float(np.mean(load_history[-24:])),
            "Load_rolling_mean_168": float(np.mean(load_history[-168:])),
        }
        pred = max(0.0, float(model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
        output.append({
            "Timestamp": ts,
            "Predicted_Demand_kW": round(pred, 2),
            "Temperature_C_assumed": round(temp, 1),
            "Humidity_pct_assumed": round(humidity, 1),
        })
        load_history.append(pred)
        history = pd.concat([history, pd.DataFrame([{
            "Timestamp": ts, TARGET: pred,
            "Temperature_C": temp, "Humidity_pct": humidity,
            "Is_Holiday": is_holiday, "Academic_Active": academic_active,
        }])], ignore_index=True)
    return pd.DataFrame(output)
