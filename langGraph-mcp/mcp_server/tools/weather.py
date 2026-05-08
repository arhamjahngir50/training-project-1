FAKE_WEATHER_DB = {
    "lahore":     {"temp": 38, "condition": "Sunny",         "humidity": "30%"},
    "karachi":    {"temp": 33, "condition": "Humid",          "humidity": "75%"},
    "islamabad":  {"temp": 28, "condition": "Partly Cloudy",  "humidity": "50%"},
    "london":     {"temp": 14, "condition": "Rainy",          "humidity": "85%"},
    "new york":   {"temp": 22, "condition": "Clear",          "humidity": "45%"},
}

def get_weather(city: str) -> dict:
    """Return fake weather data for a city."""
    key = city.lower().strip()
    if key in FAKE_WEATHER_DB:
        return {"city": city, **FAKE_WEATHER_DB[key]}
    return {"error": f"No weather data found for '{city}'"}