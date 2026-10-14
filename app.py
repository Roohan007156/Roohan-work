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
