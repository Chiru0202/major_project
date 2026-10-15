"""
Energy Dashboard Flask Application
Provides energy consumption analysis, ARIMA forecasting, and recommendations.
"""

import json
import tempfile
from datetime import datetime, date, time, timedelta
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, redirect

from analysis.analyzer import analyze_usage
from analysis.preprocessing import load_and_preprocess
from models.forecast import (
    generate_forecast,
    generate_next_month_forecast,
    forecast_summary,
    infer_frequency,
)
from recommendation.advisor import generate_advice, generate_balancing_recommendation

# Flask app setup
app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HOUSE_DIR = DATA_DIR / "houses"
REGISTRY_PATH = DATA_DIR / "houses.json"
DEFAULT_DATASET = BASE_DIR / "dataset" / "appliance_usage_dataset.csv"

# Ensure directories exist
HOUSE_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# Utility Functions
# =============================================================================

def slugify(text: str) -> str:
    """Create URL-safe slug from text."""
    text = str(text or "house").strip().lower()
    text = "".join(c if c.isalnum() else "_" for c in text)
    text = "_".join(p for p in text.split("_") if p)
    return text or "house"


def house_title(house_id: str) -> str:
    """Create display title from house ID."""
    if not house_id:
        return "House"
    text = str(house_id).replace("_", " ")
    # Capitalize properly while keeping numbers
    return " ".join(word.capitalize() if word.isalpha() else word for word in text.split())


def house_sort_key(house: dict) -> tuple:
    """Sort key for house list (numeric IDs first)."""
    hid = str(house.get("id", "")).lower()
    digits = "".join(c for c in hid if c.isdigit())
    if digits:
        return (0, int(digits), hid)
    return (1, hid, hid)


# =============================================================================
# House Registry Management
# =============================================================================

def read_registry() -> list:
    """Read house registry from disk, initializing if needed."""
    if not REGISTRY_PATH.exists():
        initialize_registry()

    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Registry must be a list")
        return data
    except (json.JSONDecodeError, ValueError):
        initialize_registry()
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(entries: list):
    """Atomically save house registry."""
    REGISTRY_PATH.parent.mkdir(exist_ok=True, parents=True)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8",
        dir=REGISTRY_PATH.parent,
        prefix="houses-", suffix=".tmp",
        delete=False
    ) as f:
        tmp_path = Path(f.name)
        json.dump(entries, f, indent=2)

    tmp_path.replace(REGISTRY_PATH)


def initialize_registry():
    """Initialize registry from default dataset."""
    entries = []

    if DEFAULT_DATASET.exists():
        df = pd.read_csv(DEFAULT_DATASET)

        if "house_id" not in df.columns:
            raise ValueError("Default dataset must have house_id column")

        house_ids = sorted(set(str(v) for v in df["house_id"].dropna()))

        for idx, hid in enumerate(house_ids, 1):
            safe_id = slugify(hid) or f"house_{idx}"
            house_path = HOUSE_DIR / safe_id
            house_path.mkdir(exist_ok=True, parents=True)

            # Save house-specific data
            house_df = df[df["house_id"].astype(str) == hid].copy()
            if house_df.empty:
                continue

            dataset_path = house_path / "energy.csv"
            house_df.to_csv(dataset_path, index=False)

            entries.append({
                "id": safe_id,
                "name": house_title(hid),
                "dataset": str(dataset_path),
            })

    if not entries:
        raise ValueError("No houses found in default dataset")

    save_registry(entries)


def get_house_options() -> list:
    """Get sorted list of house options."""
    return sorted(read_registry(), key=house_sort_key)


def get_house_by_id(house_id: str) -> dict:
    """Find house by ID or name."""
    if not house_id:
        return None

    house_id_lower = str(house_id).strip().lower()
    for house in read_registry():
        if house_id_lower in {
            str(house.get("id", "")).lower(),
            str(house.get("name", "")).lower(),
        }:
            return house
    return None


def add_house(name: str, uploaded_file) -> str:
    """Add or update a house from uploaded CSV."""
    if not uploaded_file or not uploaded_file.filename:
        raise ValueError("Please select a CSV file to upload.")

    raw_name = (name or uploaded_file.filename.rsplit(".", 1)[0]).strip()
    if not raw_name:
        raise ValueError("Please enter a house name.")

    house_id = slugify(raw_name)
    house_name = house_title(raw_name)
    house_path = HOUSE_DIR / house_id
    house_path.mkdir(exist_ok=True, parents=True)
    dataset_path = house_path / "energy.csv"

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        dir=house_path, prefix="upload-", suffix=".csv", delete=False
    ) as f:
        tmp_path = Path(f.name)

    try:
        uploaded_file.save(tmp_path)

        # Validate and preprocess
        new_df = load_and_preprocess(tmp_path, house_id=house_id, house_name=house_name)

        # Merge with existing data if present
        if dataset_path.exists():
            existing_df = load_and_preprocess(dataset_path, house_id=house_id, house_name=house_name)
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            merged.to_csv(tmp_path, index=False)

        tmp_path.replace(dataset_path)

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(str(e))

    # Update registry
    registry = read_registry()
    for entry in registry:
        if str(entry.get("id", "")).lower() == house_id.lower():
            entry.update({"name": house_name, "dataset": str(dataset_path)})
            break
    else:
        registry.append({
            "id": house_id,
            "name": house_name,
            "dataset": str(dataset_path),
        })

    save_registry(sorted(registry, key=house_sort_key))
    return house_id


def append_usage_data(house_id: str, uploaded_file) -> tuple:
    """Append new usage data from uploaded CSV to existing house data."""
    house = get_house_by_id(house_id)
    if not house:
        raise ValueError(f"House '{house_id}' not found.")

    if not uploaded_file or not uploaded_file.filename:
        raise ValueError("Please select a CSV file to upload.")

    dataset_path = Path(house.get("dataset", ""))
    if not dataset_path.exists():
        raise ValueError("No existing data found for this house.")

    house_path = dataset_path.parent
    house_name = house.get("name", house_id)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        dir=house_path, prefix="append-", suffix=".csv", delete=False
    ) as f:
        tmp_path = Path(f.name)

    try:
        uploaded_file.save(tmp_path)

        # Load and validate new data
        new_df = load_and_preprocess(tmp_path, house_id=house_id, house_name=house_name)
        new_count = len(new_df)

        # Load existing data
        existing_df = load_and_preprocess(dataset_path, house_id=house_id, house_name=house_name)
        existing_count = len(existing_df)

        # Merge: new data overwrites existing for same timestamps
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        merged.to_csv(tmp_path, index=False)

        tmp_path.replace(dataset_path)
        final_count = len(merged)
        added_count = final_count - existing_count

        return added_count, final_count

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(str(e))


def add_manual_usage(house_id: str, usage_date: str, usage_value: float, duration: str = "hourly") -> bool:
    """Add a single manual usage reading to house data."""
    house = get_house_by_id(house_id)
    if not house:
        raise ValueError(f"House '{house_id}' not found.")

    dataset_path = Path(house.get("dataset", ""))
    if not dataset_path.exists():
        raise ValueError("No existing data found for this house.")

    house_name = house.get("name", house_id)

    try:
        # Parse the date based on duration type
        if duration == "hourly":
            parsed_date = datetime.strptime(usage_date, "%Y-%m-%dT%H:%M")
        elif duration == "daily":
            parsed_date = datetime.strptime(usage_date, "%Y-%m-%d")
        elif duration == "monthly":
            parsed_date = datetime.strptime(usage_date + "-01", "%Y-%m-%d")
        else:
            parsed_date = datetime.strptime(usage_date, "%Y-%m-%dT%H:%M")

        # Load existing data
        existing_df = load_and_preprocess(dataset_path, house_id=house_id, house_name=house_name)

        # Create new row
        new_row = pd.DataFrame([{
            "date": parsed_date,
            "total_usage": float(usage_value),
            "house_id": house_id,
            "house_name": house_name,
        }])

        # Merge
        merged = pd.concat([existing_df, new_row], ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        merged.to_csv(dataset_path, index=False)

        return True

    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")
    except Exception as e:
        raise ValueError(str(e))


# =============================================================================
# Forecast Helpers
# =============================================================================

def normalize_period(period: str) -> str:
    """Normalize period name."""
    period = (period or "month").strip().lower()
    aliases = {
        "day": "day", "daily": "day", "tomorrow": "day",
        "week": "week", "weekly": "week", "7d": "week",
        "15d": "15d", "15": "15d",
        "30d": "30d", "30": "30d",
        "month": "month", "monthly": "month",
        "3month": "3month", "3months": "3month", "quarter": "3month", "90d": "3month",
        "custom": "custom",
    }
    return aliases.get(period, "month")


def get_period_label(period: str, start_date=None, end_date=None) -> str:
    """Get display label for period."""
    if period == "custom" and start_date and end_date:
        return f"Custom ({start_date} to {end_date})"
    labels = {
        "day": "Tomorrow",
        "week": "7-Day",
        "15d": "15-Day",
        "30d": "30-Day",
        "month": "Next Month",
        "3month": "Next 3 Months",
        "custom": "Custom Range",
    }
    return labels.get(period, "Month") + " Forecast"


def get_total_label(period: str) -> str:
    """Get label for forecast total."""
    labels = {
        "day": "Tomorrow Total",
        "week": "7-Day Total",
        "15d": "15-Day Total",
        "30d": "30-Day Total",
        "month": "Next Month Total",
        "3month": "3-Month Total",
        "custom": "Selected Range Total",
    }
    return labels.get(period, "Forecast Total")


def parse_custom_dates(request_values) -> tuple:
    """Parse custom date range from request."""
    start_str = request_values.get("start_date", "").strip()
    end_str = request_values.get("end_date", "").strip()

    if not start_str or not end_str:
        return None, None

    try:
        start = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)
    except ValueError:
        raise ValueError("Invalid date format. Please select valid dates from the calendar.")

    today = date.today()

    # Validation checks
    if start > end:
        raise ValueError("Start date must be before or equal to end date.")

    if start < today:
        raise ValueError("Start date cannot be in the past. Please select a future date.")

    # Check if range is too large (max 365 days)
    diff_days = (end - start).days
    if diff_days > 365:
        raise ValueError(f"Date range is {diff_days} days. Maximum allowed is 365 days.")

    # Check if dates are too far in the future (max 2 years)
    max_future = today + timedelta(days=730)
    if end > max_future:
        raise ValueError("End date cannot be more than 2 years in the future.")

    return (
        datetime.combine(start, time.min),
        datetime.combine(end, time.max)
    )


# =============================================================================
# Routes
# =============================================================================

@app.route("/", methods=["GET", "POST"])
def home():
    """Main dashboard route."""
    error_message = None
    success_message = None
    selected_house_id = request.values.get("house_id", "")
    period = normalize_period(request.values.get("period", "month"))
    preferred_appliance = request.values.get("preferred_appliance", "")
    additional_kwh = request.values.get("additional_kwh", "")

    # Handle POST requests
    if request.method == "POST":
        action = request.form.get("action", "")

        # Add new house
        if action == "add_house":
            uploaded = request.files.get("house_file")
            if uploaded and uploaded.filename:
                try:
                    house_name = request.form.get("new_house_name") or uploaded.filename.rsplit(".", 1)[0]
                    selected_house_id = add_house(house_name, uploaded)
                    success_message = f"House '{house_name}' added successfully!"
                except ValueError as e:
                    error_message = str(e)

        # Append usage data from file
        elif action == "append_data":
            uploaded = request.files.get("append_file")
            target_house = request.form.get("append_house_id", selected_house_id)
            if uploaded and uploaded.filename:
                try:
                    added, total = append_usage_data(target_house, uploaded)
                    selected_house_id = target_house
                    success_message = f"Added {added} new readings. Total: {total} records."
                except ValueError as e:
                    error_message = str(e)

        # Add manual usage entry
        elif action == "manual_entry":
            try:
                target_house = request.form.get("manual_house_id", selected_house_id)
                usage_date = request.form.get("usage_date", "")
                usage_value = request.form.get("usage_value", "")
                duration = request.form.get("duration_type", "hourly")

                if not usage_date or not usage_value:
                    raise ValueError("Please enter both date and usage value.")

                add_manual_usage(target_house, usage_date, float(usage_value), duration)
                selected_house_id = target_house
                success_message = f"Added {usage_value} kWh for {usage_date}."
            except ValueError as e:
                error_message = str(e)

        # Legacy: simple file upload without action
        else:
            uploaded = request.files.get("house_file")
            if uploaded and uploaded.filename:
                try:
                    house_name = request.form.get("new_house_name") or uploaded.filename.rsplit(".", 1)[0]
                    selected_house_id = add_house(house_name, uploaded)
                except ValueError as e:
                    error_message = str(e)

    # Get house data
    house_options = get_house_options()
    house = get_house_by_id(selected_house_id)

    if house is None and house_options:
        house = house_options[0]
    elif house is None:
        error_message = error_message or "No houses found. Please upload a dataset."
        house = {"id": "demo", "name": "Demo House", "dataset": ""}

    selected_house_id = house["id"]
    selected_house_name = house.get("name", house_title(selected_house_id))

    # Load and analyze data
    now = datetime.now()
    today = date.today()

    try:
        dataset_path = house.get("dataset")
        if dataset_path and Path(dataset_path).exists():
            df = load_and_preprocess(dataset_path, house_id=selected_house_id, house_name=selected_house_name)
        else:
            df = pd.DataFrame()
    except Exception as e:
        error_message = error_message or f"Error loading data: {e}"
        df = pd.DataFrame()

    # Analyze
    analysis = analyze_usage(df)
    frequency = infer_frequency(df["date"]) if not df.empty else {"label": "Unknown"}

    # Calculate data age
    latest_date = analysis.get("latest_date", pd.NaT)
    if pd.notna(latest_date):
        data_age = (pd.Timestamp(today) - pd.Timestamp(latest_date).normalize()).days
    else:
        data_age = None

    # Parse custom dates if needed
    custom_start, custom_end = None, None
    if period == "custom":
        try:
            custom_start, custom_end = parse_custom_dates(request.values)
            if custom_start is None or custom_end is None:
                error_message = error_message or "Please select both start and end dates for custom range."
        except ValueError as e:
            error_message = error_message or str(e)

    # Generate forecasts
    if df.empty:
        forecast_rows = []
        next_month_rows = []
    else:
        try:
            if period == "custom" and custom_start and custom_end:
                forecast_rows = generate_forecast(
                    df, period=period,
                    start_date=pd.Timestamp(custom_start),
                    end_date=pd.Timestamp(custom_end)
                )
            else:
                forecast_rows = generate_forecast(df, period=period)
            next_month_rows = generate_next_month_forecast(df)
        except Exception as e:
            error_message = error_message or f"Forecast error: {e}"
            forecast_rows = []
            next_month_rows = []

    # Calculate forecast summary
    summary = forecast_summary(forecast_rows)

    # Select default appliance if needed
    available_appliances = [
        r["appliance"] for r in analysis.get("appliance_rows", [])
        if r["availability"] == "Available"
    ]
    if preferred_appliance not in available_appliances and available_appliances:
        preferred_appliance = available_appliances[0]

    # Generate recommendations
    print(f"DEBUG: peak_hour={analysis.get('peak_hour')}, anomaly_count={analysis.get('anomaly_count')}")
    advice = generate_advice(
        analysis,
        user_preference=preferred_appliance or None,
        period=period,
        forecast_value=summary["average"] if summary["count"] > 0 else None
    )
    print(f"DEBUG: advice={advice}")

    # Generate balancing recommendation
    balancing = ""
    if preferred_appliance and not df.empty:
        try:
            kwh = float(additional_kwh) if additional_kwh else None
            balancing = generate_balancing_recommendation(df, preferred_appliance, additional_kwh=kwh)
        except ValueError:
            balancing = "Please enter a valid kWh amount."

    # Prepare chart data (sample if too large)
    historical_df = df.copy() if not df.empty else pd.DataFrame()
    if len(historical_df) > 2000:
        step = len(historical_df) // 2000 + 1
        historical_df = historical_df.iloc[::step]

    historical_rows = [
        {
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d %H:%M"),
            "consumption": round(float(row["total_usage"]), 4)
        }
        for _, row in historical_df[["date", "total_usage"]].iterrows()
    ] if not historical_df.empty else []

    # Usage breakdown (sorted ascending)
    usage_breakdown = [
        {"appliance": name, "percentage": pct}
        for name, pct in sorted(analysis.get("percentage", {}).items(), key=lambda x: x[1])
    ]

    return render_template(
        "index.html",
        # Page info
        title=f"{selected_house_name} Energy Dashboard",
        error_message=error_message,
        success_message=success_message,
        today_label=today.isoformat(),
        generated_at=now.strftime("%Y-%m-%d %H:%M"),

        # House selection
        house_options=[{"id": h["id"], "name": h["name"]} for h in house_options],
        selected_house=selected_house_id,
        selected_house_name=selected_house_name,

        # Period selection
        selected_period=period,
        start_date=request.values.get("start_date", ""),
        end_date=request.values.get("end_date", ""),

        # Appliance selection
        preferred_appliance=preferred_appliance,
        additional_kwh=additional_kwh,

        # Analysis data
        analysis=analysis,
        frequency_label=frequency.get("label", "Unknown"),
        latest_actual_date=latest_date.strftime("%Y-%m-%d %H:%M") if pd.notna(latest_date) else "N/A",
        data_age_days=data_age,
        peak_month_label=analysis["peak_month"].strftime("%b %Y") if pd.notna(analysis.get("peak_month")) else "N/A",

        # Forecast data
        forecast_rows=forecast_rows,
        next_month_rows=next_month_rows,
        forecast_label=get_period_label(period, request.values.get("start_date"), request.values.get("end_date")),
        forecast_total_label=get_total_label(period),
        forecast_total=summary["total"],
        forecast_value=summary["average"],
        forecast_sample_count=summary["count"],

        # Recommendations
        advice=advice,
        balancing_recommendation=balancing,

        # Chart data
        appliance_rows=analysis.get("appliance_rows", []),
        historical_rows=historical_rows,
        historical_actual_count=len(df),
        usage_breakdown=usage_breakdown,

        # Summary
        dataset_summary={
            "house_count": len(house_options),
            "total_usage": round(analysis.get("total_consumption", 0), 2),
            "avg_usage": round(analysis.get("average_consumption", 0), 2),
        }
    )


@app.route("/<path:legacy>", methods=["GET"])
def redirect_legacy(legacy):
    """Handle legacy URLs."""
    if "=" in legacy and "&" in legacy:
        return redirect("/?" + legacy)
    return app.send_static_file(legacy)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
