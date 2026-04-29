"""
tools.py — Mock Tool Definitions
=================================
These are the ONLY tools your agent is allowed to use.
Do NOT modify this file. Do NOT replace with real API calls.

Each tool simulates real-world conditions:
  - get_weather()   : fails ~20% of the time (network timeout simulation)
  - check_calendar(): weekends always blocked; weekdays 70% free
  - get_price()     : returns an error DICT (not an exception) for unknown items

Your agent must handle all failure modes gracefully.
"""

import random
import time


# ---------------------------------------------------------------------------
# Tool 1: Weather
# ---------------------------------------------------------------------------

def get_weather(city: str) -> dict:
    """
    Returns current weather conditions for a given city.

    Args:
        city (str): Name of the city, e.g. "Mumbai", "Delhi", "Bangalore"

    Returns:
        dict: {
            "city": str,
            "condition": "sunny" | "cloudy" | "rainy" | "windy",
            "temp_c": float,
            "humidity_pct": int
        }

    Raises:
        Exception: ~20% of calls raise a timeout error — handle this!
    """
    time.sleep(0.3)  # simulate network latency

    if random.random() < 0.20:
        raise Exception(f"WeatherAPIError: Request timed out for city='{city}'")

    conditions = ["sunny", "cloudy", "rainy", "windy"]
    return {
        "city": city,
        "condition": random.choice(conditions),
        "temp_c": round(random.uniform(5.0, 40.0), 1),
        "humidity_pct": random.randint(20, 95),
    }


# ---------------------------------------------------------------------------
# Tool 2: Calendar
# ---------------------------------------------------------------------------

def check_calendar(date: str, duration_minutes: int) -> dict:
    """
    Checks whether a time slot is available on a given date.

    Args:
        date (str)            : Date in YYYY-MM-DD format, e.g. "2025-05-12"
        duration_minutes (int): Length of the required slot in minutes

    Returns:
        dict: {
            "date": str,
            "duration_minutes": int,
            "available": bool,
            "reason": str | None   # None when available, explanation when blocked
        }

    Notes:
        - Weekends (Sat/Sun) are ALWAYS blocked.
        - Weekdays are available ~70% of the time.
        - This tool does NOT raise exceptions — check the "available" field.
    """
    time.sleep(0.2)

    from datetime import datetime
    try:
        d = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {
            "date": date,
            "duration_minutes": duration_minutes,
            "available": False,
            "reason": f"Invalid date format '{date}' — expected YYYY-MM-DD",
        }

    if d.weekday() >= 5:
        return {
            "date": date,
            "duration_minutes": duration_minutes,
            "available": False,
            "reason": "Weekend — office closed",
        }

    available = random.random() < 0.70
    return {
        "date": date,
        "duration_minutes": duration_minutes,
        "available": available,
        "reason": None if available else "Slot already booked by another team",
    }


# ---------------------------------------------------------------------------
# Tool 3: Pricing
# ---------------------------------------------------------------------------

PRICE_CATALOG = {
    "solar panel":   299.99,
    "inverter":      149.50,
    "battery pack":  499.00,
    "mounting kit":   89.00,
    "cable set":      34.99,
}


def get_price(item: str, quantity: int) -> dict:
    """
    Returns pricing information for an item from the product catalog.

    Args:
        item (str)    : Product name, e.g. "solar panel", "battery pack"
        quantity (int): Number of units

    Returns on success:
        dict: {
            "item": str,
            "quantity": int,
            "unit_price": float,
            "total": float,
            "currency": "USD"
        }

    Returns on failure (unknown item — NOT an exception):
        dict: {
            "error": str,
            "available_items": list[str]
        }

    Notes:
        - This tool NEVER raises an exception.
        - Always check whether "error" key is present in the response.
        - Item matching is case-insensitive and strips whitespace.
    """
    time.sleep(0.1)

    key = item.lower().strip()
    if key not in PRICE_CATALOG:
        return {
            "error": f"Item '{item}' not found in catalog",
            "available_items": list(PRICE_CATALOG.keys()),
        }

    unit_price = PRICE_CATALOG[key]
    return {
        "item": key,
        "quantity": quantity,
        "unit_price": unit_price,
        "total": round(unit_price * quantity, 2),
        "currency": "USD",
    }


# ---------------------------------------------------------------------------
# Tool registry — useful if you want to pass tools to an LLM dynamically
# ---------------------------------------------------------------------------

TOOLS = {
    "get_weather": get_weather,
    "check_calendar": check_calendar,
    "get_price": get_price,
}

TOOL_DESCRIPTIONS = {
    "get_weather": "Get current weather for a city. May fail with a timeout — retry if needed.",
    "check_calendar": "Check if a date+duration slot is free. Returns availability and reason.",
    "get_price": "Look up unit price and total for an item and quantity. Returns error dict if item unknown.",
}
