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

   /* Main page text */
body {
    color: #1f2937 !important;
}

/* Main title */
.main-title {
    color: #111827 !important;
    font-weight: 800 !important;
    font-size: 42px !important;
}

/* Subtitle */
.subtitle {
    color: #4b5563 !important;
    font-weight: 500 !important;
    font-size: 18px !important;
}

/* Section headings */
.section-title {
    color: #111827 !important;
    font-weight: 800 !important;
    font-size: 24px !important;
}

/* KPI title */
.kpi-title {
    color: #374151 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}

/* KPI value */
.kpi-value {
    color: #111827 !important;
    font-weight: 800 !important;
    font-size: 30px !important;
}

/* Normal Streamlit text */
.stMarkdown,
.stText,
p,
label {
    color: #374151 !important;
}

/* Metric text */
[data-testid="stMetricLabel"] {
    color: #374151 !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 800 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] {
    color: #1f2937 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: #374151 !important;
    font-weight: 500 !important;
}

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
        "Humidity_pct\n"
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

    average_load = float(
        df["Load_kW"].mean()
    )

    maximum_load = float(
        df["Load_kW"].max()
    )

    minimum_load = float(
        df["Load_kW"].min()
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

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    Latest Load
                </div>
                <div class="kpi-value">
                    {current_load:.1f} kW
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    Average Load
                </div>
                <div class="kpi-value">
                    {average_load:.1f} kW
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    Maximum Recorded Load
                </div>
                <div class="kpi-value">
                    {maximum_load:.1f} kW
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    Best Model MAPE
                </div>
                <div class="kpi-value">
                    {model_mape:.2f}%
                </div>
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
    # ACTUAL DEMAND
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Electricity Demand History'
        '</div>',
        unsafe_allow_html=True,
    )

    history_chart = df.set_index(
        "Timestamp"
    )[["Load_kW"]]

    st.line_chart(
        history_chart,
        use_container_width=True,
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

    best_model = final_models[best_name]

    forecast_df = forecast_next_24_hours(
        df,
        best_model,
        holiday_override=holiday_override,
        classes_on_weekdays=classes_on_weekdays,
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
