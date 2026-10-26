# AI Stock Predictor

An interactive app that trains an LSTM neural network on any stock ticker to
backtest and forecast closing prices, paired with an AI agent (Groq +
Phidata) that answers natural-language questions about a stock's
fundamentals and analyst recommendations.

## Features

- **Any ticker, live** — type any symbol yfinance supports (`AAPL`, `TSLA`,
  `NVDA`, ...) and the app fetches fresh historical data on demand.
- **Tunable model** — adjust the history window, LSTM lookback, training
  epochs, and forecast horizon from the sidebar.
- **Future forecasting** — recursively predicts N business days beyond the
  latest known price, not just a backtest on historical data.
- **Interactive charts** — actual vs. predicted vs. forecast, with RMSE/MAE
  metrics, rendered with Plotly.
- **AI market insights** — ask a Groq-powered agent to summarize
  fundamentals and analyst recommendations for any company or ticker.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your [Groq API key](https://console.groq.com/keys) (only needed for
   the AI Insights tab):

   ```bash
   cp .env.example .env
   # then edit .env and set GROQ_API_KEY
   ```

## Run the app

```bash
streamlit run app.py
```

This opens the interactive dashboard in your browser:

- **Prediction tab** — pick a ticker and model settings in the sidebar,
  click **Run Prediction**, and get a backtested chart plus an optional
  future-price forecast.
- **AI Insights tab** — ask questions like "compare TSLA and MSFT
  fundamentals" and get a markdown-formatted, table-rich answer.

## Using the pieces standalone

Both the model and the agent are importable, and each still runs as a
script for a quick CLI demo:

```bash
python stock_predictor.py      # trains on AAPL, plots actual/predicted/forecast with matplotlib
python finance_agent.py        # compares TSLA and MSFT fundamentals in the terminal
python finance_agent.py "What is Nvidia's forward P/E?"
```

```python
from stock_predictor import run_prediction

result = run_prediction("TSLA", start_date="2018-01-01", lookback=60, epochs=25, forecast_days=10)
print(result.rmse, result.mae)
```

## How the model works

1. **Fetch** — historical daily closing prices via `yfinance`.
2. **Prepare** — scale prices with `MinMaxScaler` and slice into
   overlapping `lookback`-day windows.
3. **Train** — a 2-layer LSTM (with dropout) on 80% of the windows, early
   stopping on validation loss.
4. **Backtest** — predict the held-out 20% and report RMSE/MAE against
   actual prices.
5. **Forecast** — recursively feed each new prediction back in as input to
   project `forecast_days` beyond the last known price.

## Notes

- Model training happens on every new ticker/settings combination and is
  cached per session (`st.cache_data`) to avoid retraining on every
  interaction.
- Stock price forecasts are for educational purposes only and are not
  financial advice.
