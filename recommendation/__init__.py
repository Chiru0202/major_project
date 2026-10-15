"""Energy recommendation module."""

from recommendation.advisor import (
    generate_advice,
    generate_balancing_recommendation,
    get_appliance_tips,
    get_quick_wins,
)

__all__ = [
    "generate_advice",
    "generate_balancing_recommendation",
    "get_appliance_tips",
    "get_quick_wins",
]
