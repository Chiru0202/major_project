import pandas as pd


def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError("The dataset must include a 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").copy()

    appliance_cols = ["ac", "fridge", "lights", "fans", "washing_machine", "tv"]
    for col in appliance_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("timestamp")
    monthly_usage = df[appliance_cols].resample("MS").sum()
    monthly_usage["total_usage"] = monthly_usage.sum(axis=1)

    return monthly_usage