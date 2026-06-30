"""OpenWeather API client for match-day weather features."""
import httpx
from typing import Dict, Optional
from config import settings

OW_BASE = "https://api.openweathermap.org/data/2.5"


class WeatherClient:
    def __init__(self):
        self.key = settings.openweather_api_key

    async def get_weather(self, lat: float, lon: float) -> Dict:
        """Fetch current weather at a lat/lon. Returns temperature, rain, wind."""
        if not self.key:
            return {"temp": 18.0, "rain_1h": 0.0, "wind_speed": 5.0, "description": "unknown"}
        params = {"lat": lat, "lon": lon, "appid": self.key, "units": "metric"}
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{OW_BASE}/weather", params=params)
            r.raise_for_status()
            d = r.json()
            return {
                "temp": d["main"]["temp"],
                "rain_1h": d.get("rain", {}).get("1h", 0.0),
                "wind_speed": d["wind"]["speed"],
                "description": d["weather"][0]["description"],
            }


weather_client = WeatherClient()
