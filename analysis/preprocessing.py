"""
Data preprocessing module for energy consumption datasets.
Handles CSV loading, date parsing, and appliance detection.
"""

import re
import pandas as pd
import numpy as np

# Column categories
DATE_COLUMNS = {'date', 'datetime', 'timestamp', 'time', 'record_date'}
IGNORE_COLUMNS = {'house_id', 'house_name', 'city', 'location', 'region', 'address',
                  'company', 'type', 'status', 'category', 'year', 'day', 'month',
                  'season', 'festival', 'holiday'}
TOTAL_COLUMNS = {'total_usage', 'total_kwh', 'total_consumption', 'electricity',
                 'consumption', 'energy', 'kwh', 'usage', 'meter_reading', 'reading',
                 'electricity_consumption', 'energy_consumption', 'electricity_bill'}

# Known appliances with display names
APPLIANCES = {
    'ac': 'Air Conditioner',
    'airconditioner': 'Air Conditioner',
    'air_conditioner': 'Air Conditioner',
    'refrigerator': 'Refrigerator',
    'fridge': 'Refrigerator',
    'lights': 'Lights',
    'light': 'Lights',
    'fan': 'Fan',
    'fans': 'Fan',
    'washing_machine': 'Washing Machine',
    'washingmachine': 'Washing Machine',
    'tv': 'Television',
    'television': 'Television',
    'microwave': 'Microwave',
    'heater': 'Heater',
    'water_heater': 'Water Heater',
    'waterheater': 'Water Heater',
    'geyser': 'Geyser',
    'computer': 'Computer',
    'laptop': 'Laptop',
    'dryer': 'Dryer',
    'dishwasher': 'Dishwasher',
    'oven': 'Oven',
    'iron': 'Iron',
    'router': 'Router',
    'motorpump': 'Motor Pump',
    'motor_pump': 'Motor Pump',
}

COMMON_APPLIANCES = ['Air Conditioner', 'Refrigerator', 'Lights', 'Fan',
                     'Television', 'Washing Machine', 'Microwave']


def normalize_name(name: str) -> str:
    """Convert column name to lowercase with underscores."""
    if not name:
        return ''
    text = str(name).strip().lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def get_appliance_display_name(column: str) -> str:
    """Get human-readable appliance name."""
    normalized = normalize_name(column)
    if normalized in APPLIANCES:
        return APPLIANCES[normalized]
    return column.replace('_', ' ').title()


def detect_date_column(df: pd.DataFrame) -> str:
    """Find the date/timestamp column in the dataframe."""
    normalized_cols = {normalize_name(c): c for c in df.columns}

    # Check known date column names
    for name in DATE_COLUMNS:
        if name in normalized_cols:
            return normalized_cols[name]

    # Check for columns containing 'date' or 'time'
    for norm, orig in normalized_cols.items():
        if 'date' in norm or 'time' in norm:
            return orig

    # Try to parse each column as dates
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().mean() > 0.8:
                return col
        except:
            continue

    return None


def detect_total_column(df: pd.DataFrame) -> str:
    """Find the total consumption column."""
    normalized_cols = {normalize_name(c): c for c in df.columns}

    for name in TOTAL_COLUMNS:
        if name in normalized_cols:
            col = normalized_cols[name]
            values = pd.to_numeric(df[col], errors='coerce')
            if values.notna().any():
                return col

    return None


def detect_appliance_columns(df: pd.DataFrame, exclude: set = None) -> list:
    """Find columns that represent appliance usage."""
    exclude = exclude or set()
    exclude = {normalize_name(c) for c in exclude}
    exclude.update(normalize_name(c) for c in DATE_COLUMNS)
    exclude.update(normalize_name(c) for c in IGNORE_COLUMNS)
    exclude.update(normalize_name(c) for c in TOTAL_COLUMNS)

    appliances = []
    for col in df.columns:
        normalized = normalize_name(col)
        if normalized in exclude:
            continue

        # Check if it's a known appliance or has numeric values
        if normalized in APPLIANCES:
            appliances.append(col)
        else:
            values = pd.to_numeric(df[col], errors='coerce')
            if values.notna().any() and values.mean() > 0:
                appliances.append(col)

    return appliances


def load_and_preprocess(path, house_id: str = None, house_name: str = None) -> pd.DataFrame:
    """
    Load and preprocess an energy consumption CSV file.

    Returns a DataFrame with columns:
    - date: datetime
    - total_usage: float (kWh)
    - [appliance columns]: float (kWh)
    - house_id: str
    - house_name: str
    """
    if path is None:
        raise ValueError("Dataset path is required.")

    # Load CSV
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Find date column
    date_col = detect_date_column(df)
    if date_col is None:
        raise ValueError("No date/timestamp column found in dataset.")

    # Parse dates
    # Handle month + year combination
    if normalize_name(date_col) == 'month' and 'year' in [normalize_name(c) for c in df.columns]:
        year_col = [c for c in df.columns if normalize_name(c) == 'year'][0]
        try:
            df['_parsed_date'] = pd.to_datetime(
                df[year_col].astype(str) + '-' + df[date_col].astype(str) + '-01',
                errors='coerce'
            )
        except:
            df['_parsed_date'] = pd.to_datetime(
                df[date_col].astype(str) + ' ' + df[year_col].astype(str),
                errors='coerce'
            )
        date_col = '_parsed_date'
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Remove rows with invalid dates
    df = df.dropna(subset=[date_col]).copy()
    if df.empty:
        raise ValueError("No valid dates found in dataset.")

    # Rename date column
    if date_col != 'date':
        df = df.rename(columns={date_col: 'date'})

    # Find total consumption and appliance columns
    total_col = detect_total_column(df)
    appliance_cols = detect_appliance_columns(df, exclude={'date', total_col} if total_col else {'date'})

    # Calculate or use total_usage
    if total_col:
        df['total_usage'] = pd.to_numeric(df[total_col], errors='coerce')
        if total_col != 'total_usage':
            df = df.drop(columns=[total_col], errors='ignore')
    elif appliance_cols:
        # Sum appliance columns
        appliance_values = df[appliance_cols].apply(pd.to_numeric, errors='coerce')
        df['total_usage'] = appliance_values.sum(axis=1, min_count=1)
    else:
        raise ValueError("No consumption data found. Need either a total column or appliance columns.")

    # Convert appliance columns to numeric
    for col in appliance_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove negative or missing consumption
    df = df[df['total_usage'] > 0].dropna(subset=['total_usage']).copy()
    if df.empty:
        raise ValueError("No valid consumption values found.")

    # Add house identifiers
    df['house_id'] = house_id or 'house'
    df['house_name'] = house_name or df['house_id']

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    return df


def get_appliance_columns(df: pd.DataFrame) -> list:
    """Get list of appliance columns from a preprocessed dataframe."""
    exclude = {'date', 'total_usage', 'house_id', 'house_name'}
    exclude.update(IGNORE_COLUMNS)

    appliances = []
    for col in df.columns:
        normalized = normalize_name(col)
        if normalized in exclude or normalized in TOTAL_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            appliances.append(col)

    return appliances


def humanize_appliance_name(name: str) -> str:
    """Alias for get_appliance_display_name for backward compatibility."""
    return get_appliance_display_name(name)
