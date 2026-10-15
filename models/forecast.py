"""
ARIMA-based energy consumption forecasting module.
Provides time series forecasting with automatic model selection.
"""

import json
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from typing import Tuple, List, Dict, Optional

# Suppress statsmodels warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=ValueWarning)


class EnergyForecaster:
    """
    ARIMA-based energy consumption forecaster with seasonal pattern overlay.

    Combines ARIMA trend with historical seasonal patterns to prevent saturation.
    """

    # ARIMA orders to try (p, d, q)
    CANDIDATE_ORDERS = [
        (1, 0, 0),  # AR(1)
        (0, 1, 1),  # IMA(1,1)
        (1, 1, 0),  # ARI(1,1)
        (1, 1, 1),  # ARIMA(1,1,1)
        (2, 1, 1),  # ARIMA(2,1,1)
        (2, 1, 2),  # ARIMA(2,1,2)
    ]

    def __init__(self, data: pd.Series, dates: pd.Series = None):
        """
        Initialize with historical consumption data.

        Args:
            data: Consumption values
            dates: Corresponding datetime index (for seasonal pattern extraction)
        """
        self.raw_data = data
        self.raw_dates = dates
        self.data = self._clean_data(data)
        self.model = None
        self.fitted = None
        self.order = None
        self.seasonal_patterns = {}
        self.trend_component = 0

        if dates is not None:
            self._extract_seasonal_patterns(data, dates)

    @staticmethod
    def _clean_data(data: pd.Series) -> pd.Series:
        """Clean and validate input data."""
        series = pd.to_numeric(pd.Series(data), errors='coerce')
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        series = series[series > 0]
        return series.reset_index(drop=True)

    def _extract_seasonal_patterns(self, data: pd.Series, dates: pd.Series):
        """Extract seasonal patterns from historical data."""
        df = pd.DataFrame({'date': pd.to_datetime(dates), 'value': pd.to_numeric(data, errors='coerce')})
        df = df.dropna()

        if df.empty:
            return

        # Extract time components
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['dayofyear'] = df['date'].dt.dayofyear

        # Calculate overall mean and std
        overall_mean = df['value'].mean()
        overall_std = df['value'].std()

        # Hourly pattern (24 values) - only if meaningful variation exists
        if df['hour'].nunique() > 1:
            hourly = df.groupby('hour')['value'].mean()
            hourly_std = hourly.std()
            # Only use if hourly variation is significant (>2% of mean)
            if hourly_std > overall_mean * 0.02:
                self.seasonal_patterns['hourly'] = (hourly / overall_mean).to_dict()
                self.seasonal_patterns['hourly_strength'] = hourly_std / overall_mean

        # Day of week pattern (7 values)
        if df['dayofweek'].nunique() > 1:
            weekly = df.groupby('dayofweek')['value'].mean()
            weekly_std = weekly.std()
            if weekly_std > overall_mean * 0.02:
                self.seasonal_patterns['weekly'] = (weekly / overall_mean).to_dict()
                self.seasonal_patterns['weekly_strength'] = weekly_std / overall_mean

        # Monthly pattern (12 values) - typically has strong seasonality
        if df['month'].nunique() > 1:
            monthly = df.groupby('month')['value'].mean()
            monthly_std = monthly.std()
            self.seasonal_patterns['monthly'] = (monthly / overall_mean).to_dict()
            self.seasonal_patterns['monthly_strength'] = monthly_std / overall_mean

        # Day of year pattern for fine-grained seasonality
        if len(df) > 365:
            daily_year = df.groupby('dayofyear')['value'].mean()
            # Smooth with rolling average to reduce noise
            daily_year = daily_year.rolling(7, min_periods=1, center=True).mean()
            self.seasonal_patterns['yearly'] = (daily_year / overall_mean).to_dict()
            self.seasonal_patterns['yearly_strength'] = daily_year.std() / overall_mean

        # Store the overall coefficient of variation for scaling
        self.seasonal_patterns['cv'] = overall_std / overall_mean if overall_mean > 0 else 0.1

        # Calculate trend from recent data
        recent = df.tail(min(len(df), 90))
        if len(recent) > 7:
            x = np.arange(len(recent))
            y = recent['value'].values
            self.trend_component = np.polyfit(x, y, 1)[0]

    def fit(self) -> 'EnergyForecaster':
        """Fit the best ARIMA model using AIC criterion."""
        if len(self.data) < 5:
            return self

        best_aic = np.inf
        best_model = None
        best_order = None

        train_data = self.data.tail(5000)

        for order in self.CANDIDATE_ORDERS:
            try:
                model = ARIMA(
                    train_data,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                fitted = model.fit()

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
                    best_order = order
            except:
                continue

        self.fitted = best_model
        self.order = best_order
        return self

    def predict(self, steps: int, start_date: pd.Timestamp = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecast with confidence intervals and seasonal adjustment.

        Args:
            steps: Number of steps to forecast
            start_date: Starting date for forecast (used for seasonal patterns)

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if steps <= 0:
            return np.array([]), np.array([]), np.array([])

        if self.fitted is None or len(self.data) < 5:
            return self._linear_forecast(steps)

        try:
            forecast = self.fitted.get_forecast(steps=steps)
            pred = forecast.predicted_mean.values
            conf = forecast.conf_int(alpha=0.1)

            lower = conf.iloc[:, 0].values
            upper = conf.iloc[:, 1].values

            # Apply seasonal adjustment to prevent saturation
            pred, lower, upper = self._apply_seasonal_adjustment(
                pred, lower, upper, steps, start_date
            )

            # Ensure non-negative values
            pred = np.maximum(pred, 0)
            lower = np.maximum(lower, 0)
            upper = np.maximum(upper, pred)

            return pred, lower, upper
        except:
            return self._linear_forecast(steps)

    def _apply_seasonal_adjustment(
        self,
        pred: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        steps: int,
        start_date: pd.Timestamp = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply historical pattern variation to forecasts.

        Since ARIMA often saturates to a flat line, we use historical data
        directly as a pattern template, shifted to match ARIMA's predicted level.
        """
        if start_date is None:
            start_date = pd.Timestamp.now()

        # Get recent historical data as pattern template
        template_size = min(len(self.data), max(steps, 168))
        recent_data = self.data.tail(template_size).values

        hist_mean = float(np.mean(recent_data))
        hist_std = float(np.std(recent_data)) if len(recent_data) > 1 else hist_mean * 0.1

        # ARIMA predicted level (may be flat/saturated)
        arima_level = float(np.mean(pred))

        # Calculate level shift: difference between ARIMA prediction and historical mean
        level_shift = arima_level - hist_mean

        adjusted_pred = np.zeros(steps)
        adjusted_lower = np.zeros(steps)
        adjusted_upper = np.zeros(steps)

        for i in range(steps):
            # Get historical value at same position in cycle
            # This directly uses the oscillation pattern from real data
            pattern_idx = i % len(recent_data)
            hist_value = recent_data[pattern_idx]

            # Shift historical pattern to ARIMA's predicted level
            # This preserves the variation shape while adjusting the baseline
            adjusted_pred[i] = hist_value + level_shift

            # Add small random variation to prevent exact repetition
            noise = np.random.normal(0, hist_std * 0.03)
            adjusted_pred[i] = max(0.1, adjusted_pred[i] + noise)

            # Confidence intervals based on historical variance
            ci_range = hist_std * 1.645
            adjusted_lower[i] = max(0, adjusted_pred[i] - ci_range)
            adjusted_upper[i] = adjusted_pred[i] + ci_range

        return adjusted_pred, adjusted_lower, adjusted_upper

    def _is_hourly_data(self) -> bool:
        """Check if the data appears to be hourly."""
        return 'hourly' in self.seasonal_patterns and len(self.seasonal_patterns.get('hourly', {})) > 12

    def _linear_forecast(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple linear trend fallback with seasonal variation."""
        if len(self.data) == 0:
            zeros = np.zeros(steps)
            return zeros, zeros, zeros

        last = float(self.data.iloc[-1])
        recent = self.data.tail(min(len(self.data), 30)).values
        mean_val = float(np.mean(recent))

        if len(recent) > 1:
            slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        else:
            slope = 0

        # Add variation based on historical standard deviation
        std_val = float(np.std(recent)) if len(recent) > 1 else mean_val * 0.1

        pred = []
        for i in range(steps):
            # Base prediction with trend
            base = last + slope * (i + 1)
            # Add cyclic variation to prevent flat line
            cycle = std_val * 0.3 * np.sin(2 * np.pi * i / 24)  # Daily cycle
            pred.append(max(0, base + cycle))

        pred = np.array(pred)
        lower = pred - std_val
        upper = pred + std_val

        return pred, np.maximum(lower, 0), upper


def infer_frequency(dates: pd.Series) -> Dict:
    """
    Detect the time frequency of the data.

    Returns:
        Dict with keys: kind, step, label, seasonal_period
    """
    dates = pd.to_datetime(pd.Series(dates), errors='coerce').dropna()
    if len(dates) < 2:
        return {'kind': 'daily', 'step': pd.Timedelta(days=1), 'label': 'Daily', 'seasonal_period': 7}

    dates = dates.sort_values().unique()
    diffs = pd.Series(dates[1:] - dates[:-1])
    diffs = diffs[diffs > pd.Timedelta(0)]

    if len(diffs) == 0:
        return {'kind': 'daily', 'step': pd.Timedelta(days=1), 'label': 'Daily', 'seasonal_period': 7}

    median = diffs.median()

    if median >= pd.Timedelta(days=25):
        return {'kind': 'monthly', 'step': None, 'label': 'Monthly', 'seasonal_period': 12}
    elif pd.Timedelta(days=6) <= median <= pd.Timedelta(days=8):
        return {'kind': 'weekly', 'step': pd.Timedelta(days=7), 'label': 'Weekly', 'seasonal_period': 52}
    elif pd.Timedelta(hours=20) <= median <= pd.Timedelta(hours=28):
        return {'kind': 'daily', 'step': pd.Timedelta(days=1), 'label': 'Daily', 'seasonal_period': 7}
    elif median < pd.Timedelta(days=1):
        hours = round(median.total_seconds() / 3600)
        return {'kind': 'hourly', 'step': pd.Timedelta(hours=max(1, hours)), 'label': f'{hours}H', 'seasonal_period': 24}
    else:
        return {'kind': 'irregular', 'step': median, 'label': f'~{median.days}D', 'seasonal_period': 7}


def generate_forecast(
    df: pd.DataFrame,
    horizon_days: int = 30,
    period: str = 'month',
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    recursive: bool = True
) -> List[Dict]:
    """
    Generate energy consumption forecast with recursive/rolling support.

    Args:
        df: DataFrame with 'date' and 'total_usage' columns
        horizon_days: Number of days to forecast
        period: Forecast period type ('day', 'week', '15d', '30d', 'month', '3month', 'custom')
        start_date: Custom start date (for period='custom')
        end_date: Custom end date (for period='custom')
        recursive: If True, use rolling forecast for distant dates (marking early steps as Provisional)

    Returns:
        List of dicts with date, consumption, lower, upper, status
    """
    if df is None or df.empty or 'date' not in df.columns:
        return []

    # Prepare data
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['total_usage'] = pd.to_numeric(df['total_usage'], errors='coerce')
    df = df.dropna(subset=['date', 'total_usage'])
    df = df.sort_values('date')

    if df.empty:
        return []

    # Sort by date but preserve all data points (don't aggregate - keeps variation)
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    freq = infer_frequency(df['date'])

    # Calculate forecast horizon
    latest_data = df['date'].max()
    today = pd.Timestamp.now().normalize()  # Today at midnight

    # Handle custom date range
    if period == 'custom' and start_date is not None and end_date is not None:
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        horizon = (end_date - start_date).days + 1
    else:
        period_days = {'day': 1, 'week': 7, '15d': 15, '30d': 30, 'month': 30, '3month': 90}
        horizon = period_days.get(period, horizon_days)
        start_date = None
        end_date = None

    # Calculate steps based on frequency
    if freq['step'] is not None:
        steps = max(1, int(pd.Timedelta(days=horizon) / freq['step']))
    else:
        steps = horizon  # Monthly data

    # Fit model and generate forecast with seasonal patterns
    forecaster = EnergyForecaster(df['total_usage'], dates=df['date'])
    forecaster.fit()
    model_order = forecaster.order

    # Start forecast from TODAY (not from after last data point)
    # This shows predictions for current/recent dates which is more useful
    if freq['kind'] == 'hourly':
        forecast_start = pd.Timestamp.now().floor('h')  # Current hour
    else:
        forecast_start = today  # Today at midnight

    pred, lower, upper = forecaster.predict(steps, start_date=forecast_start)

    # Generate forecast dates
    results = []

    # Determine provisional threshold (steps beyond direct model confidence)
    provisional_threshold = min(steps // 3, 30) if recursive and steps > 30 else steps

    # For custom range, use user-specified dates
    if period == 'custom' and start_date is not None:
        # Re-calculate with custom start
        custom_start = pd.Timestamp(start_date)
        custom_end = pd.Timestamp(end_date)

        if freq['step']:
            total_steps = max(1, int((custom_end - custom_start) / freq['step'])) + 1
        else:
            total_steps = (custom_end - custom_start).days + 1

        pred, lower, upper = forecaster.predict(total_steps, start_date=custom_start)

        for i in range(len(pred)):
            if freq['kind'] == 'monthly':
                next_date = custom_start + pd.DateOffset(months=i)
            elif freq['step']:
                next_date = custom_start + freq['step'] * i
            else:
                next_date = custom_start + pd.Timedelta(days=i)

            if next_date > custom_end:
                break

            status = _get_forecast_status(i, provisional_threshold, recursive)
            results.append({
                'date': next_date.strftime('%Y-%m-%d %H:%M:%S'),
                'consumption': round(float(pred[i]), 4),
                'lower': round(float(lower[i]), 4),
                'upper': round(float(upper[i]), 4),
                'status': status,
                'model_order': model_order
            })
    else:
        # Standard period-based forecast - START FROM TODAY
        for i in range(len(pred)):
            if freq['kind'] == 'monthly':
                next_date = forecast_start + pd.DateOffset(months=i)
            elif freq['step']:
                next_date = forecast_start + freq['step'] * i
            else:
                next_date = forecast_start + pd.Timedelta(days=i)

            status = _get_forecast_status(i, provisional_threshold, recursive)
            results.append({
                'date': next_date.strftime('%Y-%m-%d %H:%M:%S'),
                'consumption': round(float(pred[i]), 4),
                'lower': round(float(lower[i]), 4),
                'upper': round(float(upper[i]), 4),
                'status': status,
                'model_order': model_order
            })

    return results


def _get_forecast_status(step_index: int, provisional_threshold: int, recursive: bool) -> str:
    """Determine forecast status based on step position."""
    if step_index == 0:
        return 'Predicted'
    elif recursive and step_index > provisional_threshold:
        return 'Provisional'
    else:
        return 'Forecast'


def generate_next_month_forecast(df: pd.DataFrame, as_of: pd.Timestamp = None) -> List[Dict]:
    """Generate forecast for the next calendar month."""
    if df is None or df.empty:
        return []

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['total_usage'] = pd.to_numeric(df['total_usage'], errors='coerce')
    df = df.dropna(subset=['date', 'total_usage']).sort_values('date')

    if df.empty:
        return []

    # Get next month boundaries
    ref = pd.Timestamp(as_of) if as_of else pd.Timestamp.now()
    next_month = (ref + pd.DateOffset(months=1)).replace(day=1)
    month_end = (next_month + pd.DateOffset(months=1)) - pd.Timedelta(days=1)

    freq = infer_frequency(df['date'])

    # Calculate steps for next month
    if freq['kind'] == 'monthly':
        steps = 1
    elif freq['step']:
        days_in_month = (month_end - next_month).days + 1
        steps = max(1, int(days_in_month * 24 / (freq['step'].total_seconds() / 3600)))
    else:
        steps = 30

    # Generate forecast
    daily = df.groupby('date')['total_usage'].mean()
    forecaster = EnergyForecaster(daily)
    forecaster.fit()
    pred, lower, upper = forecaster.predict(steps)

    results = []
    for i in range(len(pred)):
        if freq['kind'] == 'monthly':
            date = next_month
        elif freq['step']:
            date = next_month + freq['step'] * i
        else:
            date = next_month + pd.Timedelta(days=i)

        if date > month_end:
            break

        results.append({
            'date': date.strftime('%Y-%m-%d %H:%M:%S'),
            'consumption': round(float(pred[i]), 4),
            'lower': round(float(lower[i]), 4),
            'upper': round(float(upper[i]), 4),
            'status': 'Forecast'
        })

    return results


def forecast_summary(forecast_rows: List[Dict]) -> Dict:
    """Calculate summary statistics for a forecast."""
    if not forecast_rows:
        return {'total': 0, 'average': 0, 'min': 0, 'max': 0, 'count': 0}

    values = [r['consumption'] for r in forecast_rows]
    return {
        'total': round(sum(values), 2),
        'average': round(np.mean(values), 2),
        'min': round(min(values), 2),
        'max': round(max(values), 2),
        'count': len(values)
    }


# Backward compatibility
def forecast_energy(data, steps=3, seasonal_period=None):
    """Legacy function for backward compatibility."""
    forecaster = EnergyForecaster(data)
    forecaster.fit()
    pred, _, _ = forecaster.predict(steps)
    return pd.Series(pred, dtype=float)


def generate_recursive_forecast(df, start_date=None, end_date=None, horizon_days=30,
                                period=None, as_of=None, include_bridge=False):
    """Legacy function for backward compatibility."""
    return generate_forecast(df, horizon_days=horizon_days, period=period or 'month')


def forecast_start_date(df, as_of=None):
    """Get the start date for forecasting."""
    if df is None or df.empty or 'date' not in df.columns:
        return None
    dates = pd.to_datetime(df['date'], errors='coerce').dropna()
    if dates.empty:
        return None
    return dates.max() + pd.Timedelta(days=1)


# =============================================================================
# Forecast History Persistence
# =============================================================================

class ForecastHistory:
    """Manages forecast history persistence per house."""

    def __init__(self, house_path: Path):
        """Initialize with house directory path."""
        self.house_path = Path(house_path)
        self.history_file = self.house_path / "forecast_history.json"

    def _load_history(self) -> List[Dict]:
        """Load existing history from disk."""
        if not self.history_file.exists():
            return []
        try:
            data = json.loads(self.history_file.read_text(encoding='utf-8'))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []

    def _save_history(self, history: List[Dict]):
        """Atomically save history to disk."""
        self.house_path.mkdir(exist_ok=True, parents=True)
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8',
            dir=self.house_path, prefix='history-', suffix='.tmp',
            delete=False
        ) as f:
            tmp_path = Path(f.name)
            json.dump(history, f, indent=2)
        tmp_path.replace(self.history_file)

    def save_forecast(
        self,
        forecast_rows: List[Dict],
        model_order: tuple = None,
        period: str = 'month'
    ) -> int:
        """
        Save new forecast to history.

        Args:
            forecast_rows: List of forecast dicts with date, consumption, etc.
            model_order: ARIMA model order used (p, d, q)
            period: Forecast period type

        Returns:
            Number of new records saved
        """
        if not forecast_rows:
            return 0

        history = self._load_history()
        generated_at = datetime.now().isoformat()
        model_str = f"ARIMA{model_order}" if model_order else "ARIMA"

        existing_dates = {r['date'] for r in history}
        new_records = []

        for row in forecast_rows:
            if row['date'] not in existing_dates:
                new_records.append({
                    'date': row['date'],
                    'consumption': row['consumption'],
                    'lower': row.get('lower'),
                    'upper': row.get('upper'),
                    'status': 'Provisional',
                    'generated_at': generated_at,
                    'model_used': model_str,
                    'period': period,
                    'actual': None,
                    'confirmed_at': None
                })

        if new_records:
            history.extend(new_records)
            history.sort(key=lambda x: x['date'])
            self._save_history(history)

        return len(new_records)

    def confirm_with_actual(self, actual_df: pd.DataFrame) -> int:
        """
        Update forecasts with actual data when it becomes available.

        Args:
            actual_df: DataFrame with 'date' and 'total_usage' columns

        Returns:
            Number of records updated
        """
        if actual_df is None or actual_df.empty:
            return 0

        history = self._load_history()
        if not history:
            return 0

        actual_df = actual_df.copy()
        actual_df['date'] = pd.to_datetime(actual_df['date'], errors='coerce')
        actual_df['total_usage'] = pd.to_numeric(actual_df['total_usage'], errors='coerce')
        actual_df = actual_df.dropna(subset=['date', 'total_usage'])

        actual_by_date = {}
        for _, row in actual_df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d %H:%M:%S')
            actual_by_date[date_str] = float(row['total_usage'])

        updated = 0
        for record in history:
            if record['status'] == 'Provisional' and record['date'] in actual_by_date:
                record['actual'] = actual_by_date[record['date']]
                record['status'] = 'Confirmed'
                record['confirmed_at'] = datetime.now().isoformat()
                updated += 1

        if updated > 0:
            self._save_history(history)

        return updated

    def get_history(
        self,
        start_date: str = None,
        end_date: str = None,
        status: str = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Retrieve forecast history with optional filters.

        Args:
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            status: Filter by status ('Provisional' or 'Confirmed')
            limit: Maximum records to return

        Returns:
            List of history records
        """
        history = self._load_history()

        if start_date:
            history = [r for r in history if r['date'] >= start_date]
        if end_date:
            history = [r for r in history if r['date'] <= end_date]
        if status:
            history = [r for r in history if r['status'] == status]

        return history[:limit]

    def get_accuracy_stats(self) -> Dict:
        """
        Calculate forecast accuracy statistics.

        Returns:
            Dict with MAPE, RMSE, and other accuracy metrics
        """
        history = self._load_history()
        confirmed = [r for r in history if r['status'] == 'Confirmed' and r['actual'] is not None]

        if not confirmed:
            return {'count': 0, 'mape': None, 'rmse': None, 'mae': None}

        predictions = np.array([r['consumption'] for r in confirmed])
        actuals = np.array([r['actual'] for r in confirmed])

        errors = actuals - predictions
        abs_errors = np.abs(errors)
        pct_errors = np.abs((actuals - predictions) / actuals) * 100

        return {
            'count': len(confirmed),
            'mape': round(float(np.mean(pct_errors)), 2),
            'rmse': round(float(np.sqrt(np.mean(errors ** 2))), 4),
            'mae': round(float(np.mean(abs_errors)), 4),
            'bias': round(float(np.mean(errors)), 4)
        }

    def cleanup_old(self, days: int = 365) -> int:
        """Remove history records older than specified days."""
        history = self._load_history()
        cutoff = (datetime.now() - pd.Timedelta(days=days)).isoformat()

        original_count = len(history)
        history = [r for r in history if r['generated_at'] >= cutoff]

        if len(history) < original_count:
            self._save_history(history)

        return original_count - len(history)
