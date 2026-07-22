import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path
from datetime import date
import tempfile


def fetch_weather_for_year_range(
    latitude: float = 52.57,
    longitude: float = 30.19,
    start_year: int = 2022,
    end_year: int = 2026,
    timezone: str = "Europe/Moscow",
) -> pd.DataFrame:
    """Вдохновитесь материалами по этой ссылочке, чтобы создать функцию для сбора данных.
    Ссылочка-референс: https://open-meteo.com/en/docs/historical-weather-api?latitude=52.57&longitude=30.19&start_date=2022-04-30&timezone=Europe%2FMoscow&hourly=temperature_2m,apparent_temperature,wind_speed_10m,wind_direction_10m,relative_humidity_2m#api_response"""
    if start_year > end_year:
        raise ValueError("start_year должен быть не больше end_year")

    current_year = date.today().year
    current_day = date.today().isoformat()
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31" if end_year < current_year else current_day

    cache_path = Path(tempfile.gettempdir()) / "openmeteo_cache"
    cache_session = requests_cache.CachedSession(str(cache_path), expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "apparent_temperature",
            "wind_speed_10m",
            "relative_humidity_2m",
        ],
        "timezone": timezone,
        "wind_speed_unit": "ms",
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }

    variable_names = [
        "temperature_2m",
        "apparent_temperature",
        "wind_speed_10m",
        "relative_humidity_2m",
    ]
    for index, name in enumerate(variable_names):
        hourly_data[name] = hourly.Variables(index).ValuesAsNumpy()

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe = hourly_dataframe.dropna().reset_index(drop=True)

    return hourly_dataframe


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)