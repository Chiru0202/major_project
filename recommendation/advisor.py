"""
Energy recommendation engine.
Generates actionable advice based on usage patterns and forecasts.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from analysis.preprocessing import get_appliance_columns, get_appliance_display_name


# Appliance-specific energy saving tips
APPLIANCE_TIPS = {
    'Air Conditioner': {
        'reduce': [
            "Set thermostat 1-2°C higher to save up to 10% on cooling costs",
            "Use a timer to avoid cooling empty rooms",
            "Clean filters monthly for optimal efficiency",
            "Close doors and windows while AC is running",
            "Use ceiling fans alongside AC to distribute cool air"
        ],
        'optimize': "Consider using AC only during peak heat hours (12-4 PM)"
    },
    'Refrigerator': {
        'reduce': [
            "Check door seals for air leaks",
            "Keep fridge at 3-4°C and freezer at -18°C",
            "Avoid placing hot food directly inside",
            "Defrost regularly if not auto-defrost",
            "Keep fridge well-stocked but not overcrowded"
        ],
        'optimize': "Position away from heat sources like ovens or direct sunlight"
    },
    'Lights': {
        'reduce': [
            "Switch to LED bulbs (use 75% less energy)",
            "Turn off lights when leaving a room",
            "Use natural daylight when possible",
            "Install motion sensors or timers",
            "Use task lighting instead of overhead lights"
        ],
        'optimize': "Clean light fixtures regularly for maximum brightness"
    },
    'Fan': {
        'reduce': [
            "Use lower speed settings when comfortable",
            "Turn off fans in unoccupied rooms",
            "Clean blades regularly for efficiency",
            "Use fans instead of AC when possible"
        ],
        'optimize': "Set ceiling fans to run counter-clockwise in summer"
    },
    'Television': {
        'reduce': [
            "Enable energy-saving mode in settings",
            "Turn off completely instead of standby",
            "Reduce screen brightness",
            "Use a smart power strip to cut phantom power"
        ],
        'optimize': "Consider screen size - smaller TVs use less power"
    },
    'Washing Machine': {
        'reduce': [
            "Run full loads only",
            "Use cold water cycles when possible",
            "Use eco or quick wash modes",
            "Clean the filter regularly"
        ],
        'optimize': "Wash during off-peak hours if you have time-of-use pricing"
    },
    'Microwave': {
        'reduce': [
            "Use microwave instead of oven for small portions",
            "Cover food to reduce heating time",
            "Clean regularly for efficient operation"
        ],
        'optimize': "Defrost food in fridge overnight instead of using microwave"
    },
    'Water Heater': {
        'reduce': [
            "Lower thermostat to 50-55°C",
            "Insulate hot water pipes",
            "Fix any leaking faucets promptly",
            "Use cold water for laundry when possible"
        ],
        'optimize': "Consider a timer to heat water only when needed"
    },
    'Computer': {
        'reduce': [
            "Enable sleep mode after 10 minutes of inactivity",
            "Use power management settings",
            "Turn off monitors when not in use",
            "Unplug chargers when devices are fully charged"
        ],
        'optimize': "Consider a laptop over desktop - they use 80% less power"
    }
}

# Default tips for unknown appliances
DEFAULT_TIPS = {
    'reduce': [
        "Turn off when not in use",
        "Use energy-saving modes if available",
        "Unplug when not needed to avoid standby power",
        "Regular maintenance improves efficiency"
    ],
    'optimize': "Review usage patterns and reduce during peak hours"
}


def generate_advice(
    analysis: Dict,
    user_preference: str = None,
    period: str = 'month',
    forecast_value: float = None
) -> List[str]:
    """
    Generate personalized energy-saving recommendations.

    Args:
        analysis: Output from analyze_usage()
        user_preference: Appliance the user wants to prioritize
        period: Forecast period being viewed
        forecast_value: Average forecasted consumption

    Returns:
        List of recommendation strings
    """
    advice = []
    percentages = analysis.get('percentage', {})
    avg_consumption = analysis.get('average_consumption', 0)

    # 1. Forecast-based advice
    if forecast_value is not None and avg_consumption > 0:
        diff_pct = (forecast_value - avg_consumption) / avg_consumption * 100
        if diff_pct > 15:
            advice.append(
                f"⚠️ Your {period} forecast is {diff_pct:.0f}% above average. "
                "Consider reducing usage of high-consumption appliances."
            )
        elif diff_pct < -10:
            advice.append(
                f"✓ Great! Your {period} forecast is {abs(diff_pct):.0f}% below average. "
                "Keep up the good energy habits!"
            )

    # 2. High-consumption appliance advice
    for appliance, pct in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
        if pct > 35:
            tips = APPLIANCE_TIPS.get(appliance, DEFAULT_TIPS)
            tip = tips['reduce'][0] if tips['reduce'] else tips['optimize']
            advice.append(
                f"💡 {appliance} uses {pct:.0f}% of your energy. Tip: {tip}"
            )
            break  # Only show for top consumer

    # 3. User preference handling
    if user_preference:
        normalized_pref = _normalize_appliance_name(user_preference)
        if normalized_pref in percentages:
            pref_pct = percentages[normalized_pref]

            # Find offsetting appliances
            other_appliances = [
                (name, pct) for name, pct in percentages.items()
                if name != normalized_pref and pct > 5
            ]

            if other_appliances:
                offset_name, offset_pct = max(other_appliances, key=lambda x: x[1])
                offset_tips = APPLIANCE_TIPS.get(offset_name, DEFAULT_TIPS)
                tip = offset_tips['reduce'][0] if offset_tips['reduce'] else offset_tips['optimize']
                advice.append(
                    f"🔄 To offset {user_preference} usage, reduce {offset_name} ({offset_pct:.0f}%): {tip}"
                )

    # 4. Trend-based advice
    trend = analysis.get('trend', 'Stable')
    if trend == 'Increasing':
        advice.append(
            "📈 Your usage is trending upward. Review your habits and check for "
            "inefficient appliances or behavior changes."
        )
    elif trend == 'Decreasing':
        advice.append(
            "📉 Your usage is decreasing - your conservation efforts are working!"
        )


    # 6. Peak time advice
    peak_hour = analysis.get('peak_hour')
    if peak_hour is not None:
        if 18 <= peak_hour <= 21:
            advice.append(
                "🕐 Peak usage is during evening hours (6-9 PM). "
                "Shift some activities to off-peak times if possible."
            )
        elif 9 <= peak_hour <= 17:
            advice.append(
                "🕐 Peak usage is during work hours. Consider energy audits "
                "for daytime appliances."
            )

    # 7. Low-impact appliance note
    low_impact = [name for name, pct in percentages.items() if pct < 5]
    if low_impact and len(advice) < 5:
        advice.append(
            f"ℹ️ {', '.join(low_impact[:2])} have minimal impact (<5% each). "
            "Focus optimization on larger consumers first."
        )

    # 8. Additional practical tips based on top consumers
    if len(advice) < 4:
        top_appliances = sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:2]
        for appliance, pct in top_appliances:
            if len(advice) >= 4:
                break
            tips = APPLIANCE_TIPS.get(appliance, DEFAULT_TIPS)
            if tips['reduce'] and len(tips['reduce']) > 1:
                tip = tips['reduce'][1]  # Use second tip for variety
                advice.append(f"💡 {appliance} tip: {tip}")

    # 9. General energy saving tip
    if len(advice) < 3:
        advice.append(
            "🌱 Use natural light during daytime and switch to LED bulbs "
            "to reduce lighting costs by up to 75%."
        )

    # Ensure we have at least some advice
    if not advice:
        advice.append(
            "📊 Your energy usage appears balanced. Continue monitoring "
            "and consider seasonal adjustments."
        )

    return advice[:6]  # Limit to 6 recommendations


def _normalize_appliance_name(name: str) -> str:
    """Normalize appliance name for comparison."""
    if not name:
        return ''
    return get_appliance_display_name(name)


def generate_balancing_recommendation(
    frame: pd.DataFrame,
    target_appliance: str,
    additional_kwh: float = None
) -> str:
    """
    Generate recommendations for balancing increased usage of an appliance.

    Args:
        frame: Energy consumption DataFrame
        target_appliance: Appliance user wants to use more
        additional_kwh: Additional kWh per reading period

    Returns:
        Recommendation string
    """
    if frame is None or frame.empty:
        return "Insufficient data to calculate balancing recommendations."

    appliances = get_appliance_columns(frame)
    target_normalized = _normalize_appliance_name(target_appliance)

    # Find target appliance column
    target_col = None
    for col in appliances:
        if _normalize_appliance_name(col) == target_normalized:
            target_col = col
            break

    if target_col is None:
        return f"{target_appliance} is not available in this dataset."

    # Calculate target's current usage
    target_values = pd.to_numeric(frame[target_col], errors='coerce').dropna()
    if target_values.empty:
        return f"No valid readings found for {target_appliance}."

    target_avg = float(target_values.mean())

    # Calculate additional usage
    if additional_kwh is None or additional_kwh <= 0:
        additional_kwh = target_avg * 0.1  # Default: 10% increase

    # Find flexible appliances to offset
    flexible_appliances = ['Lights', 'Fan', 'Television', 'Washing Machine', 'Microwave']
    offsets = []

    for col in appliances:
        if col == target_col:
            continue

        name = _normalize_appliance_name(col)
        if name not in flexible_appliances:
            continue

        values = pd.to_numeric(frame[col], errors='coerce').dropna()
        if values.empty:
            continue

        avg = float(values.mean())
        capacity = avg * 0.2  # Can reduce up to 20%

        if capacity > 0.01:
            tips = APPLIANCE_TIPS.get(name, DEFAULT_TIPS)
            tip = tips['reduce'][0] if tips['reduce'] else "Reduce usage when possible"
            offsets.append({
                'name': name,
                'capacity': capacity,
                'tip': tip
            })

    offsets.sort(key=lambda x: x['capacity'], reverse=True)

    # Build recommendation
    result = f"📊 Increasing {target_appliance} by {additional_kwh:.2f} kWh per reading.\n\n"

    if not offsets:
        result += "No flexible appliances found to offset this increase. "
        result += "Consider overall energy conservation measures."
        return result

    remaining = additional_kwh
    actions = []

    for offset in offsets:
        if remaining <= 0:
            break

        reduction = min(offset['capacity'], remaining)
        actions.append(f"• {offset['name']}: Reduce by {reduction:.2f} kWh - {offset['tip']}")
        remaining -= reduction

    result += "**Suggested offsets:**\n" + "\n".join(actions)

    if remaining > 0:
        result += f"\n\n⚠️ Remaining {remaining:.2f} kWh cannot be offset by flexible appliances. "
        result += "Total consumption will increase."
    else:
        result += "\n\n✓ These changes will maintain your current consumption level."

    return result


def get_appliance_tips(appliance: str) -> Dict[str, Any]:
    """Get energy-saving tips for a specific appliance."""
    name = _normalize_appliance_name(appliance)
    return APPLIANCE_TIPS.get(name, DEFAULT_TIPS)


def get_quick_wins(analysis: Dict) -> List[Dict]:
    """
    Identify quick energy-saving opportunities.

    Returns list of actions with estimated savings.
    """
    wins = []
    percentages = analysis.get('percentage', {})
    total = analysis.get('total_consumption', 0)

    for appliance, pct in sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:3]:
        if pct < 10:
            continue

        tips = APPLIANCE_TIPS.get(appliance, DEFAULT_TIPS)
        estimated_savings = total * (pct / 100) * 0.1  # 10% of appliance usage

        wins.append({
            'appliance': appliance,
            'current_share': pct,
            'action': tips['reduce'][0] if tips['reduce'] else tips['optimize'],
            'estimated_savings': round(estimated_savings, 2),
            'difficulty': 'Easy' if appliance in ['Lights', 'Fan', 'Television'] else 'Medium'
        })

    return wins
