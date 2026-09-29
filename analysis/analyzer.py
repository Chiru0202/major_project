import pandas as pd


def analyze_usage(df):
    appliance_cols = ["ac", "fridge", "lights", "fans", "washing_machine", "tv"]
    usage = df[appliance_cols].sum()

    if usage.empty or usage.sum() == 0:
        percentage = {"AC": 0.0, "Fridge": 0.0, "Lights": 0.0, "Fans": 0.0, "Washing Machine": 0.0, "TV": 0.0}
    else:
        labels = {
            "ac": "AC",
            "fridge": "Fridge",
            "lights": "Lights",
            "fans": "Fans",
            "washing_machine": "Washing Machine",
            "tv": "TV",
        }
        percentage = ((usage / usage.sum()) * 100).rename(index=labels)
        percentage = percentage.to_dict()

    peak_month = df["total_usage"].idxmax()
    total_consumption = df["total_usage"].sum()
    average_consumption = df["total_usage"].mean()

    return {
        "percentage": percentage,
        "peak_month": peak_month,
        "average_consumption": average_consumption,
        "total_consumption": total_consumption,
    }