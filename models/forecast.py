import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def forecast_energy(data):
    series = pd.Series(data).dropna().astype(float)

    if series.empty:
        return pd.Series([0.0, 0.0, 0.0], dtype=float)

    if len(series) < 2:
        last_value = float(series.iloc[-1]) if not series.empty else 0.0
        return pd.Series([last_value, last_value, last_value], dtype=float)

    try:
        model = ARIMA(series, order=(2, 1, 2))
        forecast = model.fit().forecast(steps=3)
        return pd.Series(forecast, dtype=float)
    except Exception:
        slope = (series.iloc[-1] - series.iloc[0]) / max(len(series) - 1, 1)
        fallback = [series.iloc[-1] + slope * (step + 1) for step in range(3)]
        return pd.Series(fallback, dtype=float)