"""Core LSTM stock price prediction logic.

Can be used as a library (see `run_prediction`) or run directly as a script
for a one-off prediction + matplotlib plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


def fetch_stock_data(symbol: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """Fetch historical daily closing prices for `symbol`."""
    data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'. Check the ticker and try again.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data[["Close"]]


def prepare_data(values: np.ndarray, lookback: int = 60):
    """Scale `values` and build (X, y) sliding-window sequences for training."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])

    return np.array(X), np.array(y), scaler, scaled


def build_lstm_model(input_shape, units: int = 50, dropout: float = 0.2):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(model, X_train, y_train, batch_size=32, epochs=50, validation_split=0.2, patience=10, verbose=0):
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    return model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=verbose,
    )


def forecast_future(model, last_window: np.ndarray, scaler: MinMaxScaler, days_ahead: int) -> np.ndarray:
    """Recursively forecast `days_ahead` closing prices beyond the known data.

    Each predicted day is fed back in as input for the next day's prediction.
    """
    window = last_window.copy()
    preds_scaled = []
    for _ in range(days_ahead):
        x = window.reshape(1, window.shape[0], 1)
        next_scaled = model.predict(x, verbose=0)[0, 0]
        preds_scaled.append(next_scaled)
        window = np.append(window[1:], next_scaled)
    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()


@dataclass
class PredictionResult:
    symbol: str
    dates: pd.DatetimeIndex
    actual: np.ndarray
    predicted: np.ndarray
    rmse: float
    mae: float
    future_dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    future_predicted: np.ndarray = field(default_factory=lambda: np.array([]))


def run_prediction(
    symbol: str,
    start_date: str,
    end_date: str | None = None,
    lookback: int = 60,
    epochs: int = 25,
    forecast_days: int = 0,
) -> PredictionResult:
    """Train an LSTM on `symbol` and return backtested + optional future predictions."""
    data = fetch_stock_data(symbol, start_date, end_date)
    values = data.values

    min_required = lookback + 20
    if len(values) <= min_required:
        raise ValueError(
            f"Not enough history for '{symbol}' with lookback={lookback} "
            f"(got {len(values)} days, need > {min_required}). "
            "Pick an earlier start date or a smaller lookback."
        )

    X, y, scaler, scaled_data = prepare_data(values, lookback)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_lstm_model((X_train.shape[1], 1))
    train_model(model, X_train, y_train, epochs=epochs)

    predictions = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mae = float(mean_absolute_error(actual, predictions))

    future_dates = pd.DatetimeIndex([])
    future_predicted = np.array([])
    if forecast_days > 0:
        last_window = scaled_data[-lookback:, 0]
        future_predicted = forecast_future(model, last_window, scaler, forecast_days)
        future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    return PredictionResult(
        symbol=symbol,
        dates=data.index[-len(actual):],
        actual=actual.flatten(),
        predicted=predictions.flatten(),
        rmse=rmse,
        mae=mae,
        future_dates=future_dates,
        future_predicted=future_predicted,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run_prediction(
        symbol="AAPL",
        start_date="2015-01-01",
        end_date="2023-12-31",
        lookback=60,
        epochs=50,
        forecast_days=10,
    )

    print(f"RMSE: {result.rmse:.2f}  MAE: {result.mae:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(result.dates, result.actual, label="Actual")
    plt.plot(result.dates, result.predicted, label="Predicted")
    if len(result.future_predicted):
        plt.plot(result.future_dates, result.future_predicted, label="Forecast", linestyle="--")
    plt.title(f"{result.symbol} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()
