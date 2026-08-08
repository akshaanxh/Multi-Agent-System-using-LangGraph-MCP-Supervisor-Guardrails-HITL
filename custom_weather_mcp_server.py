import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

mcp = FastMCP("Weather MCP Server")

REQUEST_TIMEOUT_SECONDS = 20


def _wmo_code_to_string(code: int) -> str:
    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return mapping.get(code, "Unknown")


def _request_json(
    url: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        details = ""
        failed_response = getattr(exc, "response", None)
        if failed_response is not None:
            details = f" Response: {failed_response.text[:500]}"
        raise RuntimeError(
            f"API request failed: {exc}.{details}"
        ) from exc


def _geocode(city: str) -> tuple[float, float, str]:
    """Call Open-Meteo Geocoding API to resolve coordinates for a city."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    data = _request_json(url, {"name": city, "count": 1, "format": "json"})
    results = data.get("results")
    if not results:
        raise ValueError(f"Could not find coordinates for city: {city}")
    result = results[0]
    return float(result["latitude"]), float(result["longitude"]), result.get("name", city)


@mcp.tool()
def get_current_weather(
    city: str,
) -> dict[str, Any]:
    """Return the current weather for a city."""
    city = city.strip()
    if not city:
        raise ValueError("city cannot be empty")

    lat, lon, resolved_city = _geocode(city)

    data = _request_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
    )

    current = data.get("current", {})

    return {
        "city": resolved_city,
        "temperature_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "condition": _wmo_code_to_string(current.get("weather_code", 0)),
        "wind_speed": current.get("wind_speed_10m"),
    }


@mcp.tool()
def get_forecast(
    city: str,
) -> dict[str, Any]:
    """
    Return the first five hourly
    forecast entries for a city.
    """
    city = city.strip()
    if not city:
        raise ValueError("city cannot be empty")

    lat, lon, resolved_city = _geocode(city)

    data = _request_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,weather_code",
            "timezone": "auto",
        },
    )

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])[:5]
    temps = hourly.get("temperature_2m", [])[:5]
    codes = hourly.get("weather_code", [])[:5]

    forecast = []
    for i in range(len(times)):
        forecast.append({
            "datetime": times[i],
            "temperature_c": temps[i] if i < len(temps) else None,
            "condition": _wmo_code_to_string(codes[i]) if i < len(codes) else "Unknown",
        })

    return {
        "city": resolved_city,
        "forecast": forecast,
    }


if __name__ == "__main__":
    mcp.run(
        transport="stdio",
    )