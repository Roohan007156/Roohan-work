import streamlit as st
import pandas as pd

from forecast import (
    normalise_dataset,
    train_and_compare,
    forecast_next_24_hours,
)

st.set_page_config(
    page_title="GridCast AI",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ GridCast AI")
st.subheader("Electricity Demand Forecasting Dashboard")
st.write(
    "Upload hourly electricity-demand data to compare machine-learning models "
    "and forecast the next 24 hours."
)

with st.sidebar:
    st.header("Forecast Settings")
    holiday_override = st.checkbox("Next 24h is a holiday", value=False)
    classes_on_weekdays = st.checkbox("Classes active on weekdays", value=True)

    st.markdown("---")
    st.markdown("### Required CSV columns")
    st.code(
        "Timestamp\n"
        "Load_kW\n"
        "Temperature_C\n"
        "Humidity_pct\n"
        "Is_Holiday (optional)\n"
        "Academic_Active (optional)"
    )

uploaded_file = st.file_uploader(
    "Upload your hourly electricity-demand CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.info(
        "Upload a CSV file to start. The dataset should contain at least 8 days "
        "of hourly data; several weeks or months are recommended."
    )
    st.stop()

try:
    raw = pd.read_csv(uploaded_file)
    df = normalise_dataset(raw)

    st.success(
        f"Loaded {len(df):,} hourly records from "
        f"{df['Timestamp'].min()} to {df['Timestamp'].max()}."
    )

    with st.expander("View uploaded data"):
        st.dataframe(df.tail(20), use_container_width=True)

    with st.spinner("Training Random Forest and Gradient Boosting models..."):
        report_df, best_name, final_models, validation = train_and_compare(df)

    st.header("Model Performance")

    cols = st.columns(len(report_df))
    for i, row in report_df.iterrows():
        with cols[i]:
            st.metric("Model", row["Model"])
            st.metric("RMSE", f"{row['RMSE (kW)']:.2f} kW")
            st.metric("MAE", f"{row['MAE (kW)']:.2f} kW")
            st.metric("MAPE", f"{row['MAPE (%)']:.2f}%")

    st.success(f"Best model based on RMSE: **{best_name}**")

    st.subheader("Actual vs Predicted Demand")
    validation_chart = validation.set_index("Timestamp")[["Actual", "Predicted"]]
    st.line_chart(validation_chart)

    st.subheader("Next 24-Hour Forecast")

    best_model = final_models[best_name]
    forecast_df = forecast_next_24_hours(
        df,
        best_model,
        holiday_override=holiday_override,
        classes_on_weekdays=classes_on_weekdays,
    )

    st.line_chart(
        forecast_df.set_index("Timestamp")[["Predicted_Demand_kW"]]
    )

    st.dataframe(forecast_df, use_container_width=True)

    csv_data = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download 24-hour forecast CSV",
        data=csv_data,
        file_name="gridcast_next_24h_forecast.csv",
        mime="text/csv",
    )

except Exception as exc:
    st.error("The application could not process this dataset.")
    st.exception(exc)
