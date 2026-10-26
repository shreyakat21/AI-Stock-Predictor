"""Interactive Streamlit app: LSTM stock forecasting + AI market insights."""

from __future__ import annotations

from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from finance_agent import get_insights
from stock_predictor import PredictionResult, run_prediction

st.set_page_config(page_title="AI Stock Predictor", page_icon="\U0001F4C8", layout="wide")

st.title("\U0001F4C8 AI Stock Predictor")
st.caption(
    "Train an LSTM on any ticker, forecast future prices, and get AI-generated market insights."
)

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Ticker symbol", value="AAPL", help="Any symbol yfinance supports, e.g. AAPL, TSLA, NVDA").strip().upper()
    years_back = st.slider("Years of history", min_value=1, max_value=10, value=5)
    lookback = st.slider("Lookback window (days)", min_value=10, max_value=120, value=60, step=5)
    epochs = st.slider("Training epochs", min_value=5, max_value=100, value=25, step=5)
    forecast_days = st.slider("Forecast future days", min_value=0, max_value=30, value=7)
    run = st.button("Run Prediction", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.result_symbol = None


@st.cache_data(show_spinner=False, ttl=3600)
def cached_prediction(symbol: str, start_date: str, end_date: str, lookback: int, epochs: int, forecast_days: int) -> PredictionResult:
    return run_prediction(symbol, start_date, end_date, lookback=lookback, epochs=epochs, forecast_days=forecast_days)


tab_predict, tab_insights = st.tabs(["\U0001F52E Prediction", "\U0001F916 AI Insights"])

with tab_predict:
    if run:
        if not symbol:
            st.error("Enter a ticker symbol.")
        else:
            end_date = date.today().isoformat()
            start_date = (date.today() - timedelta(days=365 * years_back)).isoformat()
            with st.spinner(f"Training LSTM on {symbol} ({years_back}y history, {epochs} epochs)..."):
                try:
                    st.session_state.result = cached_prediction(
                        symbol, start_date, end_date, lookback, epochs, forecast_days
                    )
                    st.session_state.result_symbol = symbol
                except ValueError as exc:
                    st.session_state.result = None
                    st.error(str(exc))
                except Exception as exc:  # yfinance / network / training failures
                    st.session_state.result = None
                    st.error(f"Prediction failed: {exc}")

    result: PredictionResult | None = st.session_state.result

    if result is None:
        st.info("Set your options in the sidebar and click **Run Prediction**.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"${result.rmse:.2f}")
        col2.metric("MAE", f"${result.mae:.2f}")
        if len(result.future_predicted):
            delta = result.future_predicted[-1] - result.actual[-1]
            col3.metric(
                f"{len(result.future_predicted)}-day forecast",
                f"${result.future_predicted[-1]:.2f}",
                f"{delta:+.2f}",
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.dates, y=result.actual, name="Actual", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=result.dates, y=result.predicted, name="Predicted", line=dict(color="#ff7f0e")))
        if len(result.future_predicted):
            fig.add_trace(
                go.Scatter(
                    x=result.future_dates,
                    y=result.future_predicted,
                    name="Forecast",
                    line=dict(color="#2ca02c", dash="dash"),
                )
            )
        fig.update_layout(
            title=f"{result.symbol} — Actual vs Predicted vs Forecast",
            xaxis_title="Date",
            yaxis_title="Close Price (USD)",
            hovermode="x unified",
            template="plotly_white",
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_insights:
    st.subheader("AI-generated market insights")
    default_question = f"Summarize the latest fundamentals and analyst recommendations for {symbol or 'AAPL'}. Show in tables."
    question = st.text_area("Ask the AI analyst", value=default_question, height=100)
    if st.button("Get Insights"):
        with st.spinner("Asking the AI analyst..."):
            try:
                st.markdown(get_insights(symbol or "AAPL", question))
            except Exception as exc:
                st.error(f"Could not fetch insights: {exc}")
