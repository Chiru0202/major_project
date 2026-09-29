from pathlib import Path

from flask import Flask, redirect, render_template, request
import pandas as pd

from models.forecast import forecast_energy
from analysis.analyzer import analyze_usage
from recommendation.advisor import generate_advice
from analysis.preprocessing import load_and_preprocess

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
UPLOAD_DIR = DATASET_DIR / "custom_houses"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)


def clean_column_name(name):
    cleaned = str(name).strip().lower().replace(" ", "_")
    cleaned = cleaned.replace("-", "_")
    return cleaned


def normalize_columns(df):
    mapping = {
        "air_conditioner": "ac",
        "air_conditioning": "ac",
        "ac": "ac",
        "refrigerator": "fridge",
        "fridge": "fridge",
        "light": "lights",
        "lights": "lights",
        "fan": "fans",
        "fans": "fans",
        "washing_machine": "washing_machine",
        "washingmachine": "washing_machine",
        "tv": "tv",
        "television": "tv",
    }
    df.columns = [clean_column_name(col) for col in df.columns]
    rename_map = {col: mapping.get(col, col) for col in df.columns}
    return df.rename(columns=rename_map)


def normalize_period(period):
    value = (period or "month").strip().lower()
    if value in {"day", "daily", "1d", "next_day"}:
        return "day"
    if value in {"week", "weekly", "7d", "next_week"}:
        return "week"
    return "month"


def get_dataset(path):
    df = pd.read_csv(path)
    df = normalize_columns(df)
    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    appliance_cols = ["ac", "fridge", "lights", "fans", "washing_machine", "tv"]
    for col in appliance_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    if "house_id" not in df.columns:
        df["house_id"] = path.stem.replace("custom_", "House") if "custom" in path.stem.lower() else "House_1"
    df["total_usage"] = df[appliance_cols].sum(axis=1, numeric_only=True)
    return df


def forecast_for_period(data, period):
    if data.empty:
        return 0.0, "Next Month"

    if period == "day":
        series = data.resample("D", on="timestamp")["total_usage"].sum()
        if series.empty:
            return 0.0, "Next Day"
        recent = series.tail(7)
        value = float(recent.mean()) if not recent.empty else 0.0
        return value, "Next Day"
    if period == "week":
        series = data.resample("W-MON", on="timestamp")["total_usage"].sum()
        if series.empty:
            return 0.0, "Next Week"
        recent = series.tail(4)
        value = float(recent.mean()) if not recent.empty else 0.0
        return value, "Next Week"
    series = data.resample("MS", on="timestamp")["total_usage"].sum()
    if series.empty:
        return 0.0, "Next Month"
    recent = series.tail(6)
    value = float(recent.mean()) if not recent.empty else 0.0
    return value, "Next Month"


def house_sort_key(name):
    text = str(name).strip()
    if text.lower() == "all":
        return (0, 0)
    digits = ''.join(ch for ch in text if ch.isdigit())
    if digits:
        return (1, int(digits))
    return (2, text)


def get_house_options():
    default_path = DATASET_DIR / "appliance_usage_dataset.csv"
    default_df = get_dataset(default_path)
    custom_paths = sorted(UPLOAD_DIR.glob("*.csv"))
    option_list = sorted(default_df["house_id"].dropna().unique().tolist())
    for custom_path in custom_paths:
        custom_df = get_dataset(custom_path)
        custom_df["house_id"] = custom_path.stem
        option_list.extend(sorted(custom_df["house_id"].dropna().unique().tolist()))
    unique_houses = sorted(set(option_list), key=house_sort_key)
    return unique_houses


@app.route("/<path:legacy_query>", methods=["GET"])
def redirect_legacy_query(legacy_query):
    if "=" in legacy_query and "&" in legacy_query:
        return redirect("/?" + legacy_query)
    return app.send_static_file(legacy_query)


@app.route('/', methods=['GET', 'POST'])
def home():
    default_path = DATASET_DIR / "appliance_usage_dataset.csv"
    raw_df = get_dataset(default_path)
    for custom_path in sorted(UPLOAD_DIR.glob("*.csv")):
        custom_df = get_dataset(custom_path)
        custom_df["house_id"] = custom_path.stem
        raw_df = pd.concat([raw_df, custom_df], ignore_index=True)
    house_options = get_house_options()

    uploaded_file = request.files.get("house_file")
    new_house_name = (request.form.get("new_house_name") or "").strip()
    selected_house = (request.form.get("house_id") or request.args.get("house_id") or "all").strip()
    house_input = (request.form.get("house_input") or request.args.get("house_input") or "").strip()
    if house_input:
        selected_house = house_input
    error_message = None

    if uploaded_file and uploaded_file.filename:
        if not new_house_name:
            new_house_name = uploaded_file.filename.rsplit(".", 1)[0]
        safe_name = new_house_name.replace(" ", "_")
        saved_path = UPLOAD_DIR / f"{safe_name}.csv"
        uploaded_file.save(saved_path)
        try:
            custom_df = get_dataset(saved_path)
            custom_df["house_id"] = safe_name
            raw_df = pd.concat([raw_df, custom_df], ignore_index=True)
            selected_house = safe_name
            house_options = get_house_options()
        except Exception as exc:
            error_message = f"Could not process the uploaded CSV: {exc}"
            if saved_path.exists():
                saved_path.unlink(missing_ok=True)
            selected_house = (request.form.get("house_id") or request.args.get("house_id") or "all").strip()

    preferred_appliance = (request.form.get("preferred_appliance") or request.args.get("preferred_appliance") or "AC").strip()

    if selected_house in {"", "all"}:
        filtered_df = raw_df.copy()
        house_label = "All Houses"
        selected_house = "all"
    else:
        filtered_df = raw_df[raw_df["house_id"] == selected_house].copy()
        house_label = selected_house.replace("_", " ")

    period = normalize_period(request.form.get("period") or request.args.get("period"))
    forecast_value, _ = forecast_for_period(filtered_df, period)
    forecast_label_map = {
        "day": "Day Forecast",
        "week": "Week Forecast",
        "month": "Month Forecast",
    }
    forecast_label = forecast_label_map.get(period, "Month Forecast")

    if filtered_df.empty:
        analysis = {
            "percentage": {
                "AC": 0.0,
                "Fridge": 0.0,
                "Lights": 0.0,
                "Fans": 0.0,
                "Washing Machine": 0.0,
                "TV": 0.0,
            },
            "peak_month": pd.NaT,
            "average_consumption": 0.0,
            "total_consumption": 0.0,
        }
    else:
        if selected_house == "all":
            analysis = analyze_usage(load_and_preprocess(str(default_path)))
        else:
            usage_ratio = filtered_df[["ac", "fridge", "lights", "fans", "washing_machine", "tv"]].sum()
            usage_total = usage_ratio.sum()
            percent_data = {}
            if usage_total:
                labels = {
                    "ac": "AC",
                    "fridge": "Fridge",
                    "lights": "Lights",
                    "fans": "Fans",
                    "washing_machine": "Washing Machine",
                    "tv": "TV",
                }
                percent_data = {labels.get(k, k): round((v / usage_total) * 100, 2) for k, v in usage_ratio.items()}
            monthly_usage = filtered_df.set_index("timestamp")["total_usage"].resample("MS").sum()
            analysis = {
                "percentage": percent_data,
                "peak_month": monthly_usage.idxmax() if not monthly_usage.empty else pd.NaT,
                "average_consumption": float(filtered_df["total_usage"].mean()),
                "total_consumption": float(filtered_df["total_usage"].sum()),
            }

    advice = generate_advice(analysis, user_preference=preferred_appliance, period=period, forecast_value=forecast_value)
    peak_month = analysis["peak_month"]
    peak_month_label = peak_month.strftime("%b %Y") if pd.notna(peak_month) else "N/A"

    if selected_house == "all":
        title = "Energy Dashboard"
    else:
        title = f"{house_label} Energy Insights"

    return render_template(
        "index.html",
        title=title,
        house_options=house_options,
        selected_house=selected_house,
        selected_period=period,
        preferred_appliance=preferred_appliance,
        analysis=analysis,
        peak_month_label=peak_month_label,
        forecast={forecast_label: round(forecast_value, 2)},
        advice=advice,
        house_label=house_label,
        forecast_label=forecast_label,
        error_message=error_message,
        dataset_summary={
            "house_count": raw_df["house_id"].nunique(),
            "total_usage": round(float(raw_df["total_usage"].sum()), 2),
            "avg_usage": round(float(raw_df["total_usage"].mean()), 2),
        },
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)