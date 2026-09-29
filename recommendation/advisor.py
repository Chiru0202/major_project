import pandas as pd


def _normalize_preference(preference):
    if preference is None:
        return None

    lookup = {
        "ac": "AC",
        "air conditioner": "AC",
        "airconditioner": "AC",
        "conditioner": "AC",
        "fridge": "Fridge",
        "refrigerator": "Fridge",
        "lights": "Lights",
        "light": "Lights",
        "fans": "Fans",
        "fan": "Fans",
        "washing machine": "Washing Machine",
        "washingmachine": "Washing Machine",
        "tv": "TV",
        "television": "TV",
    }

    normalized = str(preference).strip().lower().replace("_", " ")
    return lookup.get(normalized, preference)


def _device_action(device):
    actions = {
        "AC": "raise the thermostat by 1-2 degrees and use a timer instead of cooling empty rooms",
        "Fridge": "keep the door closed, check the seal, and avoid placing hot food inside",
        "Lights": "switch to LEDs and turn lights off in rooms that are not occupied",
        "Fans": "use lower speed when comfortable and switch fans off when nobody is in the room",
        "Washing Machine": "run full loads and use cold or eco cycles where suitable",
        "TV": "reduce standby time and enable the television's energy-saving mode",
    }
    return actions.get(device, "avoid unnecessary runtime and switch it off when idle")


def _offset_action(device):
    actions = {
        "AC": "close doors and windows while it runs and use the timer",
        "Fridge": "check the door seal and avoid repeated door opening",
        "Lights": "turn off unused lights and make better use of daylight",
        "Fans": "switch fans off in empty rooms or use a lower speed",
        "Washing Machine": "combine loads and use an eco cycle",
        "TV": "turn it fully off instead of leaving it on standby",
    }
    return actions.get(device, "switch it off when it is not needed")


def generate_advice(analysis, user_preference=None, period="month", forecast_value=None):
    advice = []
    usage = analysis.get("percentage", {})
    normalized_pref = _normalize_preference(user_preference)

    for device, percent in usage.items():
        percent_value = float(percent)
        if percent_value > 35:
            if device == normalized_pref:
                advice.append(f"{device} leads usage at {percent_value:.2f}%. To keep using it comfortably, {_device_action(device)}.")
            else:
                advice.append(f"{device} accounts for {percent_value:.2f}% of usage. The quickest saving is to {_device_action(device)}.")
        elif percent_value < 5:
            advice.append(f"{device} is only {percent_value:.2f}% of usage, so changing it will have limited impact. Prioritize the larger loads first.")

    peak = analysis.get("peak_month")
    if peak is None or pd.isna(peak):
        advice.append("There is not enough history to identify a reliable peak month yet.")
    else:
        advice.append(f"Your highest recorded consumption was in {peak.strftime('%B')}. Compare appliance routines during that period before changing everything at once.")

    if forecast_value is not None:
        avg = float(analysis.get("average_consumption", 0) or 0)
        if avg and forecast_value > avg * 1.15:
            advice.append(f"The {period} outlook is {forecast_value:.2f} kWh, above the recent average. Delay washing and other flexible loads until demand is lower.")
        elif avg and forecast_value < avg * 0.85:
            advice.append(f"The {period} outlook is {forecast_value:.2f} kWh, below the recent average. Keep the current routine and watch whether the saving continues.")

    if usage:
        ranked_devices = sorted(usage, key=usage.get, reverse=True)
        highest = ranked_devices[0]
        if normalized_pref and normalized_pref in usage and len(ranked_devices) > 1:
            alternative = next((device for device in ranked_devices if device != normalized_pref), None)
            alternative_percent = float(usage[alternative])
            advice.append(
                f"If you increase {normalized_pref}, offset it through {alternative} ({alternative_percent:.2f}%): {_offset_action(alternative)}."
            )
        elif normalized_pref and normalized_pref in usage:
            advice.append(f"{normalized_pref} is the only tracked appliance in this dataset, so use it in shorter intervals and monitor the next forecast.")
        else:
            advice.append(f"Focus on optimizing {highest}, as it consumes the most energy.")

    return advice