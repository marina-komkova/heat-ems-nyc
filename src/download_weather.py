"""
Загрузка ежедневной погоды по пяти районам Нью-Йорка.

Источник: Open-Meteo Historical Weather API (реанализ ERA5, бесплатно, без ключа)
https://open-meteo.com/en/docs/historical-weather-api

Для каждого района берём точку в его центре и скачиваем дневные показатели:
максимальную и минимальную температуру, «ощущаемую» температуру (учитывает влажность)
и среднюю влажность. Ночная температура (минимум) важна: если ночью не остывает,
организм не успевает восстановиться.

Запуск:  uv run python src/download_weather.py
"""

import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "raw" / "weather_daily.parquet"

# Названия районов — точно как в колонке borough данных скорой, чтобы потом легко соединить таблицы.
BOROUGHS = {
    "MANHATTAN": (40.7831, -73.9712),
    "BROOKLYN": (40.6782, -73.9442),
    "QUEENS": (40.7282, -73.7949),
    "BRONX": (40.8448, -73.8648),
    "RICHMOND / STATEN ISLAND": (40.5795, -74.1502),
}

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "relative_humidity_2m_mean",
    "precipitation_sum",
]


def fetch(start_date: str, end_date: str) -> pd.DataFrame:
    """Дневная погода по 5 районам за период [start_date, end_date] включительно."""
    frames = []
    for borough, (lat, lon) in BOROUGHS.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(DAILY_VARS),
            "timezone": "America/New_York",  # чтобы «день» совпадал с местным днём вызовов скорой
        }
        # Бесплатный API ограничивает частоту запросов. Если сервер ответил 429
        # («слишком много запросов»), ждём и пробуем снова — так делают со всеми публичными API.
        for attempt in range(5):
            response = requests.get(URL, params=params, timeout=120)
            if response.status_code != 429:
                break
            wait = 30 * (attempt + 1)
            print(f"  лимит запросов, жду {wait} с...")
            time.sleep(wait)
        response.raise_for_status()
        time.sleep(5)  # небольшая пауза между районами
        df = pd.DataFrame(response.json()["daily"])
        df["borough"] = borough
        frames.append(df)
        print(f"{borough}: {len(df)} дней")

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.rename(columns={"time": "date"})
    weather["date"] = pd.to_datetime(weather["date"])
    return weather


def main() -> None:
    weather = fetch("2016-01-01", "2026-09-30")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(OUT_FILE, index=False)
    print(f"Сохранено: {OUT_FILE} ({len(weather):,} строк)")


if __name__ == "__main__":
    main()
