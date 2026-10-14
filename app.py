
import streamlit as st
import pandas as pd

from forecast import (
    normalise_dataset,
    train_and_compare,
    forecast_next_24_hours,
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="GridCast AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM DASHBOARD STYLE
# ---------------------------------------------------------

st.markdown(
    """
    <style>

     /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    /* KPI cards */
    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }

    .kpi-title {
        font-size: 14px;
        color: #666;
        font-weight: 600;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* Readable text */
    body {
        color: #1f2937 !important;
    }

    .main-title {
        color: #111827 !important;
    }

    .subtitle {
        color: #4b5563 !important;
        font-weight: 500 !important;
    }

    .section-title {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    .kpi-title {
        color: #374151 !important;
        font-weight: 700 !important;
    }

    .kpi-value {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    .stMarkdown, .stText, p, label {
        color: #1f2937 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #374151 !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #374151 !important;
        font-weight: 600 !important;
    }

    /* Info box */
    .info-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

   # ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## ⚡ GridCast AI")

    st.markdown(
        "### Institutional Campus "
        "Electricity Forecasting"
    )

    st.markdown("---")

    st.markdown("### Forecast Settings")

    holiday_override = st.checkbox(
        "Next 24h is a holiday",
        value=False,
    )

    classes_on_weekdays = st.checkbox(
        "Classes active on weekdays",
        value=True,
    )

    st.markdown("---")

    st.markdown("### Upload Data")

    uploaded_file = st.file_uploader(
        "Upload hourly electricity-demand CSV",
        type=["csv"],
    )

    st.markdown("---")

    st.markdown("### Required columns")

    st.code(
        "Timestamp\n"
        "Load_kW\n"
        "Temperature_C\n"
        "Humidity_pct"
    )

    st.markdown("**Optional columns:**")

    st.code(
        "Is_Holiday\n"
        "Academic_Active"
    )

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">⚡ GridCast AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Institutional Campus Electricity Demand Forecasting System'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# NAVIGATION STYLE
# ---------------------------------------------------------

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.button("📊 Dashboard", use_container_width=True)

with nav2:
    st.button("🔮 Forecast", use_container_width=True)

with nav3:
    st.button("📈 History", use_container_width=True)

with nav4:
    st.button("🤖 Models", use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

if uploaded_file is None:

    st.info(
        "Upload your campus electricity CSV from the sidebar "
        "to start the forecasting dashboard."
    )

    st.markdown(
        """
        <div class="info-box">

        ### Welcome to GridCast AI ⚡

        This dashboard is designed for institutional campus
        electricity-demand forecasting using machine learning.

        <br>

        <b>Current capabilities:</b>

        <ul>
        <li>Electricity demand data processing</li>
        <li>Random Forest forecasting</li>
        <li>Gradient Boosting forecasting</li>
        <li>Model comparison</li>
        <li>24-hour electricity demand prediction</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()

# ---------------------------------------------------------
# READ DATA
# ---------------------------------------------------------

try:

    raw = pd.read_csv(uploaded_file)

    df = normalise_dataset(raw)

    st.success(
        f"Dataset loaded successfully: "
        f"{len(df):,} hourly records"
    )

    # -----------------------------------------------------
    # TRAIN MODELS
    # -----------------------------------------------------

    with st.spinner(
        "Training Random Forest and Gradient Boosting models..."
    ):

        report_df, best_name, final_models, validation = (
            train_and_compare(df)
        )

    # -----------------------------------------------------
    # KPI CALCULATIONS
    # -----------------------------------------------------

    current_load = float(
        df["Load_kW"].iloc[-1]
    )

    # -----------------------------------------------------
    # CREATE 24-HOUR FORECAST
    # -----------------------------------------------------

    best_model = final_models[best_name]

    forecast_df = forecast_next_24_hours(
        df,
        best_model,
        holiday_override=holiday_override,
        classes_on_weekdays=classes_on_weekdays,
    )

    # -----------------------------------------------------
    # FORECAST KPIs
    # -----------------------------------------------------

    forecast_peak = float(
        forecast_df["Predicted_Demand_kW"].max()
    )

    forecast_average = float(
        forecast_df["Predicted_Demand_kW"].mean()
    )

    peak_index = forecast_df[
        "Predicted_Demand_kW"
    ].idxmax()

    peak_time = forecast_df.loc[
        peak_index,
        "Timestamp"
    ]

    peak_time_string = peak_time.strftime("%H:%M")

    # Hourly kW summed across 24 hours gives approximate kWh.
    daily_energy = float(
        forecast_df["Predicted_Demand_kW"].sum()
    )

    best_row = report_df[
        report_df["Model"] == best_name
    ].iloc[0]

    model_mape = float(
        best_row["MAPE (%)"]
    )

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Campus Energy Overview'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">CURRENT LOAD</div>
                <div class="kpi-value">{current_load:.1f} kW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">FORECAST PEAK</div>
                <div class="kpi-value">{forecast_peak:.1f} kW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">PEAK HOUR</div>
                <div class="kpi-value">{peak_time_string}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">DAILY ENERGY</div>
                <div class="kpi-value">{daily_energy:,.0f} kWh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">AVERAGE FORECAST</div>
                <div class="kpi-value">{forecast_average:.1f} kW</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">MODEL MAPE</div>
                <div class="kpi-value">{model_mape:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # DATA INFORMATION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Dataset Overview'
        '</div>',
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "First timestamp",
            str(df["Timestamp"].min()),
        )

    with d2:
        st.metric(
            "Latest timestamp",
            str(df["Timestamp"].max()),
        )

    with d3:
        st.metric(
            "Records",
            f"{len(df):,}",
        )

 # -----------------------------------------------------
# STAGE 4 - ELECTRICITY DEMAND HISTORY
# -----------------------------------------------------

 st.markdown(
    '<div class="section-title">'
    '📈 Electricity Demand History'
    '</div>',
    unsafe_allow_html=True,
 )

 st.write(
    "Explore historical campus electricity demand "
    "over different time periods."
 )


 # -----------------------------------------------------
 # HISTORY PERIOD SELECTOR
 # -----------------------------------------------------

 history_period = st.radio(
    "Select history period",
    ["24 Hours", "7 Days", "30 Days"],
    horizontal=True,
 )

 # -----------------------------------------------------
 # FILTER DATA
 # -----------------------------------------------------

 history_df = df.copy()

 latest_timestamp = history_df["Timestamp"].max()

 if history_period == "24 Hours":

    start_time = latest_timestamp - pd.Timedelta(hours=24)

 elif history_period == "7 Days":

    start_time = latest_timestamp - pd.Timedelta(days=7)

 else:

    start_time = latest_timestamp - pd.Timedelta(days=30)


 history_df = history_df[
     history_df["Timestamp"] >= start_time
 ].copy()


# -----------------------------------------------------
# HISTORY KPIs
# -----------------------------------------------------

history_peak = float(
    history_df["Load_kW"].max()
)

history_minimum = float(
    history_df["Load_kW"].min()
)

history_average = float(
    history_df["Load_kW"].mean()
)

history_energy = float(
    history_df["Load_kW"].sum()
)


h1, h2, h3, h4 = st.columns(4)


with h1:

    st.metric(
        "Peak Demand",
        f"{history_peak:.1f} kW"
    )


with h2:

    st.metric(
        "Minimum Demand",
        f"{history_minimum:.1f} kW"
    )


with h3:

    st.metric(
        "Average Demand",
        f"{history_average:.1f} kW"
    )


with h4:

    st.metric(
        "Energy",
        f"{history_energy:,.0f} kWh"
    )


# -----------------------------------------------------
# DEMAND HISTORY CHART
# -----------------------------------------------------

history_chart = history_df.set_index(
    "Timestamp"
)[["Load_kW"]]


st.line_chart(
    history_chart,
    use_container_width=True,
)


# -----------------------------------------------------
# WEATHER HISTORY
# -----------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '🌡️ Load & Weather'
    '</div>',
    unsafe_allow_html=True,
)

weather_chart = history_df.set_index(
    "Timestamp"
)[
    [
        "Temperature_C",
        "Humidity_pct",
    ]
]


st.line_chart(
    weather_chart,
    use_container_width=True,
)


# -----------------------------------------------------
# HISTORY DATA TABLE
# -----------------------------------------------------

with st.expander("View historical data"):

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # MODEL PERFORMANCE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Machine Learning Model Performance'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        f"🏆 Best model based on RMSE: {best_name}"
    )

    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Actual vs Predicted Demand'
        '</div>',
        unsafe_allow_html=True,
    )

    validation_chart = validation.set_index(
        "Timestamp"
    )[["Actual", "Predicted"]]

    st.line_chart(
        validation_chart,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # 24 HOUR FORECAST
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🔮 Next 24-Hour Electricity Forecast'
        '</div>',
        unsafe_allow_html=True,
    )

    forecast_chart = forecast_df.set_index(
        "Timestamp"
    )[["Predicted_Demand_kW"]]

    st.line_chart(
        forecast_chart,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # FORECAST TABLE
    # -----------------------------------------------------

    st.markdown(
        "### Hour-by-Hour Forecast"
    )

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    csv_data = forecast_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download 24-hour Forecast CSV",
        data=csv_data,
        file_name="gridcast_next_24h_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

except Exception as exc:

    st.error(
        "The application could not process this dataset."
    )

    st.exception(exc)
