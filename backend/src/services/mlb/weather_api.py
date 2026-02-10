"""Weather API client using Open-Meteo (free, no API key required)."""

import httpx
import structlog
from datetime import datetime, date
from dataclasses import dataclass
from typing import Any

logger = structlog.get_logger()


@dataclass
class WeatherData:
    """Weather data for a game location."""
    temperature: int  # Fahrenheit
    wind_speed: int  # MPH
    wind_direction: int  # Degrees (0 = North, 90 = East, etc.)
    humidity: int  # Percentage
    precipitation_probability: int  # Percentage
    weather_code: int  # WMO weather code
    is_clear: bool


# MLB Stadium coordinates (latitude, longitude)
STADIUM_COORDS = {
    "Angel Stadium": (33.8003, -117.8827),
    "Chase Field": (33.4455, -112.0667),
    "Oriole Park at Camden Yards": (39.2838, -76.6218),
    "Fenway Park": (42.3467, -71.0972),
    "Wrigley Field": (41.9484, -87.6553),
    "Great American Ball Park": (39.0979, -84.5082),
    "Progressive Field": (41.4962, -81.6852),
    "Coors Field": (39.7559, -104.9942),
    "Comerica Park": (42.3390, -83.0485),
    "Minute Maid Park": (29.7573, -95.3555),
    "Kauffman Stadium": (39.0517, -94.4803),
    "Dodger Stadium": (34.0739, -118.2400),
    "loanDepot park": (25.7781, -80.2196),
    "American Family Field": (43.0280, -87.9712),
    "Target Field": (44.9817, -93.2776),
    "Citi Field": (40.7571, -73.8458),
    "Yankee Stadium": (40.8296, -73.9262),
    "Oakland Coliseum": (37.7516, -122.2005),
    "Citizens Bank Park": (39.9061, -75.1665),
    "PNC Park": (40.4469, -80.0057),
    "Petco Park": (32.7076, -117.1570),
    "Oracle Park": (37.7786, -122.3893),
    "T-Mobile Park": (47.5914, -122.3325),
    "Busch Stadium": (38.6226, -90.1928),
    "Tropicana Field": (27.7682, -82.6534),
    "Globe Life Field": (32.7513, -97.0825),
    "Rogers Centre": (43.6414, -79.3894),
    "Nationals Park": (38.8730, -77.0074),
    "Truist Park": (33.8907, -84.4678),
    "Guaranteed Rate Field": (41.8299, -87.6338),
}

# Wind direction impact on scoring
# Positive = helps hitters (out to CF), Negative = helps pitchers (in from OF)
def calculate_wind_factor(wind_speed: int, wind_direction: int, stadium: str) -> float:
    """
    Calculate wind impact factor on scoring.

    Args:
        wind_speed: Wind speed in MPH
        wind_direction: Direction in degrees (0 = N, 90 = E, etc.)
        stadium: Stadium name for orientation adjustment

    Returns:
        Factor where >1.0 helps hitters, <1.0 helps pitchers
    """
    if wind_speed < 5:
        return 1.0  # Light wind, minimal impact

    # Most stadiums face roughly NE, so wind from SW helps hitters
    # Simplify: out (blowing toward outfield) helps hitters
    # This is a rough approximation; real model would use stadium orientation

    # Convert to radial factor (-1 to 1 based on direction)
    # Assume 225 degrees (SW) is "out to center" for average park
    import math
    optimal_direction = 225  # SW wind blowing out
    direction_diff = abs(wind_direction - optimal_direction)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    # cos gives 1 at optimal, -1 at opposite
    direction_factor = math.cos(math.radians(direction_diff))

    # Scale by wind speed (max impact around 15-20 mph)
    speed_factor = min(wind_speed / 15.0, 1.5)

    # Calculate total impact (typical range: 0.95 to 1.10)
    impact = direction_factor * speed_factor * 0.05
    return 1.0 + impact


class WeatherAPIClient:
    """Client for Open-Meteo weather API."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    async def get_weather_for_game(
        self,
        venue_name: str,
        game_time: datetime,
    ) -> WeatherData | None:
        """
        Fetch weather forecast for a specific game.

        Args:
            venue_name: Stadium name
            game_time: Game start time (UTC)

        Returns:
            WeatherData object or None if location unknown
        """
        coords = STADIUM_COORDS.get(venue_name)
        if not coords:
            logger.warning("Unknown stadium for weather", venue=venue_name)
            return None

        lat, lon = coords

        # Get forecast for the game day
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,weather_code,wind_speed_10m,wind_direction_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "America/New_York",  # Games reported in ET typically
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

            # Find the hour closest to game time
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            if not times:
                return None

            # Find index of closest hour to game time
            game_hour = game_time.strftime("%Y-%m-%dT%H:00")
            try:
                idx = times.index(game_hour)
            except ValueError:
                # Fall back to first hour of game day
                game_date_str = game_time.strftime("%Y-%m-%d")
                for i, t in enumerate(times):
                    if t.startswith(game_date_str):
                        idx = i
                        break
                else:
                    return None

            temperature = int(hourly["temperature_2m"][idx])
            humidity = int(hourly["relative_humidity_2m"][idx])
            precip_prob = int(hourly["precipitation_probability"][idx])
            weather_code = int(hourly["weather_code"][idx])
            wind_speed = int(hourly["wind_speed_10m"][idx])
            wind_direction = int(hourly["wind_direction_10m"][idx])

            # Clear weather: codes 0-3 are clear/partly cloudy
            is_clear = weather_code <= 3

            logger.debug(
                "Fetched weather",
                venue=venue_name,
                temp=temperature,
                wind=wind_speed,
            )

            return WeatherData(
                temperature=temperature,
                wind_speed=wind_speed,
                wind_direction=wind_direction,
                humidity=humidity,
                precipitation_probability=precip_prob,
                weather_code=weather_code,
                is_clear=is_clear,
            )

        except Exception as e:
            logger.warning("Failed to fetch weather", venue=venue_name, error=str(e))
            return None

    async def get_weather_batch(
        self,
        games: list[tuple[str, datetime]],
    ) -> dict[str, WeatherData | None]:
        """
        Fetch weather for multiple games.

        Args:
            games: List of (venue_name, game_time) tuples

        Returns:
            Dict mapping venue to WeatherData
        """
        results = {}
        for venue, game_time in games:
            results[venue] = await self.get_weather_for_game(venue, game_time)
        return results

    def calculate_weather_factor(
        self,
        weather: WeatherData,
        venue_name: str,
    ) -> float:
        """
        Calculate combined weather impact on run scoring.

        Args:
            weather: Weather data
            venue_name: Stadium name

        Returns:
            Factor where >1.0 favors hitters, <1.0 favors pitchers
        """
        factor = 1.0

        # Temperature impact
        # Warmer = better for hitters (ball travels further)
        if weather.temperature >= 85:
            factor += 0.03
        elif weather.temperature >= 75:
            factor += 0.02
        elif weather.temperature >= 65:
            factor += 0.01
        elif weather.temperature <= 50:
            factor -= 0.02
        elif weather.temperature <= 40:
            factor -= 0.04

        # Wind impact
        wind_factor = calculate_wind_factor(
            weather.wind_speed,
            weather.wind_direction,
            venue_name,
        )
        factor *= wind_factor

        # Humidity impact (minimal)
        if weather.humidity >= 80:
            factor -= 0.01  # Heavy air

        return round(factor, 3)


async def test_weather_api():
    """Test function to verify weather API."""
    client = WeatherAPIClient()

    # Test with Yankee Stadium
    from datetime import timedelta
    game_time = datetime.now() + timedelta(hours=3)

    print("Fetching weather for Yankee Stadium...")
    weather = await client.get_weather_for_game("Yankee Stadium", game_time)

    if weather:
        print(f"Temperature: {weather.temperature}F")
        print(f"Wind: {weather.wind_speed} MPH from {weather.wind_direction} degrees")
        print(f"Humidity: {weather.humidity}%")
        print(f"Precipitation chance: {weather.precipitation_probability}%")
        print(f"Clear: {weather.is_clear}")

        factor = client.calculate_weather_factor(weather, "Yankee Stadium")
        print(f"Weather factor: {factor}")
    else:
        print("Could not fetch weather")

    return weather


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_weather_api())
