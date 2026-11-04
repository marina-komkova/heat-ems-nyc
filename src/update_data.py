"""
Ежемесячное обновление данных: докачивает новые месяцы вызовов и погоды
и дописывает их к агрегатам в data/processed/.

Сырые вызовы (16 млн строк) в репозитории не хранятся. Вместо них хранятся
три агрегата, которых достаточно для всех моделей и дашборда:
  - daily_borough.parquet — день × район: число вызовов, медиана доезда, погода, календарь;
  - daily_tail.parquet    — день × район: 90-й процентиль доезда, медианы для срочных и несрочных;
  - hourly_peak.parquet   — час × день × район (10:00–19:59): вызовы, занятость бригад, ожидание.

Запуск:
  uv run python src/update_data.py                 # докачать новые месяцы (так делает GitHub Actions)
  uv run python src/update_data.py --from-raw      # пересобрать агрегаты из локальных data/raw/ems
"""

import argparse
import sys
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

sys.path.append(str(Path(__file__).resolve().parent))
import download_weather  # noqa: E402
from download_ems import download_month  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW_EMS = ROOT / "data" / "raw" / "ems"
DAILY, TAIL, HOURLY = PROC / "daily_borough.parquet", PROC / "daily_tail.parquet", PROC / "hourly_peak.parquet"
API = "https://data.cityofnewyork.us/resource/76xm-jjuj.json"

CLEAN = "initial_call_type != 'STNDBY' AND borough != 'UNKNOWN'"   # как в ноутбуке 02
VALID = "valid_incident_rspns_time_indc = 'Y'"
WEATHER_COLS = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "apparent_temperature_max", "relative_humidity_2m_mean", "precipitation_sum"]


def aggregate(ems_glob: str):
    """Три агрегата из сырых месячных файлов (тот же SQL, что в ноутбуках 02, 04 и 07)."""
    con = duckdb.connect()
    daily = con.sql(f"""
        SELECT count(*) AS n_calls,
               count(*) FILTER (WHERE initial_severity_level_code <= 3) AS n_severe,
               median(incident_response_seconds_qy) FILTER (WHERE {VALID}) AS resp_median,
               CAST(incident_datetime AS DATE) AS date,
               borough
        FROM '{ems_glob}' WHERE {CLEAN}
        GROUP BY date, borough""").df()
    tail = con.sql(f"""
        SELECT CAST(incident_datetime AS DATE) AS date, borough,
               quantile_cont(incident_response_seconds_qy, 0.9) FILTER (WHERE {VALID}) AS p90,
               median(incident_response_seconds_qy)
                   FILTER (WHERE {VALID} AND initial_severity_level_code <= 3) AS med_sev,
               median(incident_response_seconds_qy)
                   FILTER (WHERE {VALID} AND initial_severity_level_code >= 4) AS med_lo
        FROM '{ems_glob}' WHERE {CLEAN}
        GROUP BY ALL""").df()
    hourly = con.sql(f"""
        SELECT CAST(incident_datetime AS DATE) AS date, hour(incident_datetime) AS hour, borough,
               count(*) AS n_calls,
               avg(epoch(incident_close_datetime - first_assignment_datetime) / 3600)
                   FILTER (WHERE epoch(incident_close_datetime - first_assignment_datetime)
                           BETWEEN 60 AND 6*3600) AS busy_h,
               avg((epoch(first_assignment_datetime - incident_datetime) > 60)::int) AS share_wait60
        FROM '{ems_glob}' WHERE {CLEAN} AND hour(incident_datetime) BETWEEN 10 AND 19
        GROUP BY ALL""").df()
    for df in (daily, tail, hourly):
        df["date"] = pd.to_datetime(df["date"])
        df["borough"] = df["borough"].astype("string")
    return daily, tail, hourly


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Календарные признаки, как в ноутбуке 02."""
    d = df["date"]
    holidays = USFederalHolidayCalendar().holidays(start="2016-01-01", end=f"{d.dt.year.max() + 1}-12-31")
    df["weekday"] = d.dt.dayofweek.astype("int32")
    df["month"] = d.dt.month.astype("int32")
    df["year"] = d.dt.year.astype("int32")
    df["is_holiday"] = d.isin(holidays)
    df["is_dst_change"] = (((d.dt.month == 3) & d.dt.day.between(8, 14) & (d.dt.dayofweek == 6))
                           | ((d.dt.month == 11) & d.dt.day.between(1, 7) & (d.dt.dayofweek == 6)))
    df["is_covid"] = d.between("2020-03-01", "2022-02-28")
    return df


def build_daily(calls: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    w = weather[["date", "borough"] + WEATHER_COLS].copy()
    w["borough"] = w["borough"].astype("string")
    df = calls.merge(w, on=["date", "borough"])
    df["relative_humidity_2m_mean"] = df["relative_humidity_2m_mean"].round().astype("int64")
    return add_calendar(df)


def append(path: Path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    old = pd.read_parquet(path) if path.exists() else new.iloc[0:0]
    out = (pd.concat([old, new], ignore_index=True)
           .drop_duplicates(subset=keys, keep="last")
           .sort_values(["borough"] + [k for k in keys if k != "borough"])
           .reset_index(drop=True))
    out.to_parquet(path, index=False)
    return out


def months_to_fetch() -> list[pd.Period]:
    """Полные месяцы после последней даты в daily_borough, которые уже есть на портале."""
    last_have = pd.read_parquet(DAILY, columns=["date"])["date"].max()
    r = requests.get(API, params={"$select": "max(incident_datetime) AS m"}, timeout=120)
    r.raise_for_status()
    last_portal = pd.Timestamp(r.json()[0]["m"])
    first = last_have.to_period("M") + 1
    # месяц считаем полным, если на портале есть его последний день
    last_full = last_portal.to_period("M") if last_portal.day >= 28 and \
        last_portal.normalize() == last_portal.to_period("M").end_time.normalize() else last_portal.to_period("M") - 1
    return list(pd.period_range(first, last_full, freq="M")) if first <= last_full else []


def update() -> bool:
    months = months_to_fetch()
    if not months:
        print("Новых полных месяцев нет — данные актуальны.")
        return False
    print("Докачиваю:", ", ".join(map(str, months)))
    with tempfile.TemporaryDirectory() as tmp:
        for m in months:
            print(" ", download_month(m, Path(tmp)), flush=True)
        calls, tail, hourly = aggregate(f"{tmp}/*.parquet")
    start, end = months[0].start_time.date(), months[-1].end_time.date()
    weather = download_weather.fetch(str(start), str(end))
    daily = build_daily(calls, weather)
    if daily["date"].nunique() != (end - start).days + 1:
        raise RuntimeError("Погода пришла не за все дни — попробуйте позже.")
    d = append(DAILY, daily, ["date", "borough"])
    append(TAIL, tail, ["date", "borough"])
    append(HOURLY, hourly, ["date", "hour", "borough"])
    print(f"Готово: данные по {d['date'].max().date()}, {len(d):,} строк день × район.")
    return True


def from_raw() -> None:
    """Однократно: агрегаты из локальных сырых файлов + сверка с ноутбуком 02."""
    calls, tail, hourly = aggregate(str(RAW_EMS / "*.parquet"))
    weather = pd.read_parquet(ROOT / "data" / "raw" / "weather_daily.parquet")
    daily = build_daily(calls, weather).sort_values(["borough", "date"]).reset_index(drop=True)
    ref = pd.read_parquet(DAILY).sort_values(["borough", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(daily[ref.columns], ref, check_dtype=False)
    print("Сверка с ноутбуком 02: совпадает полностью.")
    tail.sort_values(["borough", "date"]).to_parquet(TAIL, index=False)
    hourly.sort_values(["borough", "date", "hour"]).to_parquet(HOURLY, index=False)
    print(f"Сохранено: {TAIL.name} ({len(tail):,}), {HOURLY.name} ({len(hourly):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", action="store_true")
    args = ap.parse_args()
    from_raw() if args.from_raw else update()
