"""
Загрузка вызовов скорой помощи Нью-Йорка (NYC EMS Incident Dispatch Data).

Источник: NYC Open Data, датасет 76xm-jjuj
https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj

Как это работает:
- Портал работает на платформе Socrata. У неё есть REST API с языком запросов SoQL —
  он очень похож на SQL: $select (какие колонки), $where (фильтр), $limit (сколько строк).
- Всего в датасете ~30 млн строк. Скачивать всё одним запросом нельзя (упадёт по таймауту),
  поэтому качаем по одному месяцу и сохраняем каждый месяц в отдельный файл.
- Файлы сохраняем в формате Parquet: он в ~6 раз меньше CSV и читается намного быстрее.
- Если скрипт прервётся, при повторном запуске уже скачанные месяцы пропускаются.

Запуск:  uv run python src/download_ems.py
"""

from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.cityofnewyork.us/resource/76xm-jjuj.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ems"

# Берём только нужные колонки — так быстрее и файлы меньше.
COLUMNS = [
    "incident_datetime",               # когда поступил вызов
    "initial_call_type",               # тип вызова (HEAT = тепловой удар и т.п.)
    "initial_severity_level_code",     # срочность: 1 — самая высокая, 8 — самая низкая
    "first_assignment_datetime",       # когда назначили первую бригаду
    "incident_response_seconds_qy",    # секунд от вызова до приезда на место
    "valid_incident_rspns_time_indc",  # 'Y' — время доезда корректное, можно использовать
    "incident_close_datetime",         # когда вызов закрыт
    "borough",                         # район (боро) города
]

START = "2016-01"
END = "2026-09"


def download_month(month: pd.Period) -> str:
    out_file = OUT_DIR / f"{month}.parquet"
    if out_file.exists():
        return f"{month}: уже есть, пропускаю"

    start = month.start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end = (month + 1).start_time.strftime("%Y-%m-%dT%H:%M:%S")
    params = {
        "$select": ",".join(COLUMNS),
        "$where": f"incident_datetime >= '{start}' AND incident_datetime < '{end}'",
        "$limit": 1_000_000,  # в месяце ~130–150 тыс. вызовов, так что лимит с запасом
    }
    response = requests.get(BASE_URL, params=params, timeout=600)
    response.raise_for_status()  # если сервер вернул ошибку — сразу падаем с понятным сообщением

    df = pd.read_csv(
        StringIO(response.text),
        parse_dates=["incident_datetime", "first_assignment_datetime", "incident_close_datetime"],
        dtype={"initial_call_type": "string", "borough": "string",
               "valid_incident_rspns_time_indc": "string"},
    )
    df.to_parquet(out_file, index=False)
    return f"{month}: {len(df):,} вызовов"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(START, END, freq="M")
    # Качаем 4 месяца одновременно — это в ~4 раза быстрее, но не перегружает сервер.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for message in pool.map(download_month, months):
            print(message, flush=True)


if __name__ == "__main__":
    main()
