"""
Energy usage analysis module.
Provides comprehensive analysis of consumption patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from analysis.preprocessing import get_appliance_columns, get_appliance_display_name, COMMON_APPLIANCES


def analyze_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of energy consumption data.

    Returns dict with:
    - Basic stats: total, average, min, max, latest
    - Appliance breakdown
    - Trends and patterns
    - Anomaly detection
    """
    if df is None or df.empty or 'date' not in df.columns:
        return _empty_analysis()

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['total_usage'] = pd.to_numeric(df['total_usage'], errors='coerce')
    df = df.dropna(subset=['date', 'total_usage']).sort_values('date')

    if df.empty:
        return _empty_analysis()

    # Basic statistics
    total = float(df['total_usage'].sum())
    average = float(df['total_usage'].mean())
    minimum = float(df['total_usage'].min())
    maximum = float(df['total_usage'].max())
    latest = float(df['total_usage'].iloc[-1])
    latest_date = df['date'].iloc[-1]

    # Appliance analysis
    appliance_cols = get_appliance_columns(df)
    appliance_data = _analyze_appliances(df, appliance_cols)

    # Temporal patterns
    monthly = _analyze_monthly(df)
    hourly = _analyze_hourly(df)
    weekly = _analyze_weekly(df)

    # Trend analysis
    trend = _calculate_trend(df['total_usage'])
    trend_direction = _get_trend_direction(trend)

    # Anomaly detection
    anomalies = _detect_anomalies(df)

    # Peak analysis
    peak_month = monthly['peak_month'] if monthly else pd.NaT
    peak_hour = hourly.get('peak_hour')
    peak_day = weekly.get('peak_day')

    return {
        # Basic stats
        'total_consumption': total,
        'average_consumption': average,
        'min_consumption': minimum,
        'max_consumption': maximum,
        'latest_actual': latest,
        'latest_date': latest_date,
        'record_count': len(df),

        # Appliance data
        'appliance_rows': appliance_data['rows'],
        'percentage': appliance_data['percentages'],
        'highest_appliance': appliance_data['highest'],

        # Patterns
        'peak_month': peak_month,
        'peak_hour': peak_hour,
        'peak_day': peak_day,
        'monthly_patterns': monthly.get('patterns', []),
        'hourly_patterns': hourly.get('patterns', []),
        'weekly_patterns': weekly.get('patterns', []),

        # Trend
        'trend': trend_direction,
        'trend_slope': trend,
        'trend_percent': round(trend / average * 100, 2) if average else 0,

        # Anomalies
        'anomalies': anomalies,
        'anomaly_count': len(anomalies),

        # For charts
        'history': [
            {'date': row['date'], 'consumption': float(row['total_usage'])}
            for _, row in df[['date', 'total_usage']].iterrows()
        ]
    }


def _empty_analysis() -> Dict:
    """Return empty analysis structure."""
    return {
        'total_consumption': 0,
        'average_consumption': 0,
        'min_consumption': 0,
        'max_consumption': 0,
        'latest_actual': 0,
        'latest_date': pd.NaT,
        'record_count': 0,
        'appliance_rows': [],
        'percentage': {},
        'highest_appliance': 'N/A',
        'peak_month': pd.NaT,
        'peak_hour': None,
        'peak_day': None,
        'monthly_patterns': [],
        'hourly_patterns': [],
        'weekly_patterns': [],
        'trend': 'Insufficient data',
        'trend_slope': 0,
        'trend_percent': 0,
        'anomalies': [],
        'anomaly_count': 0,
        'history': []
    }


def _analyze_appliances(df: pd.DataFrame, appliance_cols: List[str]) -> Dict:
    """Analyze appliance-level consumption."""
    rows = []
    percentages = {}
    totals = {}

    # Calculate totals for each appliance
    for col in appliance_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if not values.empty and values.sum() > 0:
            name = get_appliance_display_name(col)
            totals[name] = float(values.sum())

    grand_total = sum(totals.values()) if totals else 0

    # Create rows for available appliances
    for name, total in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        pct = round(total / grand_total * 100, 2) if grand_total else 0
        percentages[name] = pct
        rows.append({
            'appliance': name,
            'consumption': round(total, 2),
            'percentage': pct,
            'availability': 'Available'
        })

    # Add common appliances that are not available
    available_names = set(totals.keys())
    for common in COMMON_APPLIANCES:
        if common not in available_names:
            rows.append({
                'appliance': common,
                'consumption': 'N/A',
                'percentage': 'N/A',
                'availability': 'Not Available'
            })

    # Sort: available first, then alphabetically
    rows.sort(key=lambda x: (x['availability'] != 'Available', x['appliance']))

    highest = max(totals, key=totals.get) if totals else 'N/A'

    return {
        'rows': rows,
        'percentages': percentages,
        'highest': highest,
        'totals': totals
    }


def _analyze_monthly(df: pd.DataFrame) -> Dict:
    """Analyze monthly consumption patterns."""
    if df.empty:
        return {'peak_month': pd.NaT, 'patterns': []}

    df = df.copy()
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month')['total_usage'].sum()

    if monthly.empty:
        return {'peak_month': pd.NaT, 'patterns': []}

    peak_month = monthly.idxmax().to_timestamp()

    patterns = [
        {'period': str(period), 'consumption': round(float(value), 2)}
        for period, value in monthly.items()
    ]

    return {
        'peak_month': peak_month,
        'patterns': patterns,
        'average': float(monthly.mean()),
        'std': float(monthly.std()) if len(monthly) > 1 else 0
    }


def _analyze_hourly(df: pd.DataFrame) -> Dict:
    """Analyze hourly consumption patterns (for sub-daily data)."""
    if df.empty:
        return {'peak_hour': None, 'patterns': []}

    df = df.copy()
    df['hour'] = df['date'].dt.hour

    # Check if we have meaningful hourly variation
    if df['hour'].nunique() < 4:
        return {'peak_hour': None, 'patterns': []}

    hourly = df.groupby('hour')['total_usage'].mean()

    if hourly.empty:
        return {'peak_hour': None, 'patterns': []}

    peak_hour = int(hourly.idxmax())

    patterns = [
        {'hour': int(hour), 'consumption': round(float(value), 4)}
        for hour, value in hourly.items()
    ]

    return {
        'peak_hour': peak_hour,
        'patterns': patterns,
        'morning_avg': float(hourly[6:12].mean()) if len(hourly) > 12 else 0,
        'evening_avg': float(hourly[18:22].mean()) if len(hourly) > 22 else 0
    }


def _analyze_weekly(df: pd.DataFrame) -> Dict:
    """Analyze day-of-week consumption patterns."""
    if df.empty:
        return {'peak_day': None, 'patterns': []}

    df = df.copy()
    df['dayofweek'] = df['date'].dt.dayofweek
    daily = df.groupby('dayofweek')['total_usage'].mean()

    if daily.empty or len(daily) < 3:
        return {'peak_day': None, 'patterns': []}

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    peak_day = days[int(daily.idxmax())]

    patterns = [
        {'day': days[int(dow)], 'consumption': round(float(value), 4)}
        for dow, value in daily.items()
    ]

    return {
        'peak_day': peak_day,
        'patterns': patterns,
        'weekday_avg': float(daily[daily.index < 5].mean()) if (daily.index < 5).any() else 0,
        'weekend_avg': float(daily[daily.index >= 5].mean()) if (daily.index >= 5).any() else 0
    }


def _calculate_trend(series: pd.Series) -> float:
    """Calculate linear trend slope."""
    if len(series) < 2:
        return 0

    x = np.arange(len(series))
    y = series.values.astype(float)

    try:
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except:
        return 0


def _get_trend_direction(slope: float) -> str:
    """Convert slope to trend direction."""
    if abs(slope) < 0.01:
        return 'Stable'
    elif slope > 0:
        return 'Increasing'
    else:
        return 'Decreasing'


def _detect_anomalies(df: pd.DataFrame, threshold: float = 2.5) -> List[Dict]:
    """
    Detect anomalous consumption readings using Z-score method.

    Args:
        df: DataFrame with date and total_usage
        threshold: Z-score threshold for anomaly detection

    Returns:
        List of anomalous readings
    """
    if len(df) < 10:
        return []

    mean = df['total_usage'].mean()
    std = df['total_usage'].std()

    if std == 0:
        return []

    anomalies = []
    for _, row in df.iterrows():
        z_score = (row['total_usage'] - mean) / std
        if abs(z_score) > threshold:
            anomalies.append({
                'date': row['date'],
                'consumption': round(float(row['total_usage']), 2),
                'z_score': round(float(z_score), 2),
                'type': 'High' if z_score > 0 else 'Low'
            })

    return anomalies[:10]  # Return top 10 anomalies


def get_insights(analysis: Dict) -> List[str]:
    """Generate human-readable insights from analysis."""
    insights = []

    # Trend insight
    trend = analysis.get('trend', 'Stable')
    trend_pct = analysis.get('trend_percent', 0)
    if trend == 'Increasing' and trend_pct > 5:
        insights.append(f"Your energy usage is trending upward by {trend_pct:.1f}% per reading period.")
    elif trend == 'Decreasing' and trend_pct < -5:
        insights.append(f"Great news! Your usage is decreasing by {abs(trend_pct):.1f}% per reading period.")

    # Peak appliance insight
    highest = analysis.get('highest_appliance')
    if highest and highest != 'N/A':
        pct = analysis.get('percentage', {}).get(highest, 0)
        if pct > 30:
            insights.append(f"{highest} is your largest consumer at {pct:.1f}% of total usage.")

    # Anomaly insight
    anomaly_count = analysis.get('anomaly_count', 0)
    if anomaly_count > 0:
        insights.append(f"Detected {anomaly_count} unusual consumption reading(s) in your data.")

    # Peak time insights
    peak_hour = analysis.get('peak_hour')
    if peak_hour is not None:
        time_str = f"{peak_hour}:00"
        insights.append(f"Peak usage typically occurs around {time_str}.")

    return insights


def build_appliance_summary(df: pd.DataFrame) -> List[Dict]:
    """Build appliance summary for backward compatibility."""
    return analyze_usage(df).get('appliance_rows', [])
