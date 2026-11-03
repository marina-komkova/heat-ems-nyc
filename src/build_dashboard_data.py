"""
Готовит небольшие таблицы для дашборда (data/processed/dashboard/*.parquet),
чтобы сам дашборд открывался мгновенно и не пересчитывал модели.

Запуск:  uv run python src/build_dashboard_data.py   (≈ 1 минута)
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from forecast import FEATS_FULL, FEATS_NO_WEATHER, load_features, split, train_lgbm

ROOT = Path(__file__).resolve().parent.parent
EMS = str(ROOT / "data" / "raw" / "ems" / "*.parquet")
OUT = ROOT / "data" / "processed" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)
CONTROLS = "C(month) + C(weekday) + C(year) + is_holiday + is_dst_change"
VALID = "valid_incident_rspns_time_indc = 'Y'"
CLEAN = "initial_call_type != 'STNDBY' AND borough != 'UNKNOWN'"

daily = pd.read_parquet(ROOT / "data" / "processed" / "daily_borough.parquet")
daily = daily[~daily["is_covid"]].copy()
con = duckdb.connect()


def save(df, name):
    df.to_parquet(OUT / f"{name}.parquet")
    print(f"  {name}: {len(df)} строк")


# ---------- H1: город × день, Пуассон ----------
city = (daily.groupby("date")
        .agg(n_calls=("n_calls", "sum"), tmax=("temperature_2m_max", "mean"),
             weekday=("weekday", "first"), month=("month", "first"), year=("year", "first"),
             is_holiday=("is_holiday", "first"), is_dst_change=("is_dst_change", "first"))
        .reset_index())
save(city[["date", "n_calls", "tmax"]], "city_daily")

bins = [-20, 0, 5, 10, 15, 20, 25, 30, 33, 45]
labels = ["< 0", "0–5", "5–10", "10–15", "15–20", "20–25", "25–30", "30–33", "> 33"]
city["tbin"] = pd.cut(city["tmax"], bins=bins, labels=labels).astype(str)
m3 = smf.glm(f"n_calls ~ C(tbin, Treatment(reference='15–20')) + {CONTROLS}",
             data=city, family=sm.families.Poisson()).fit(cov_type="HC0")
rows = []
for lab in labels:
    if lab == "15–20":
        rows.append({"bin": lab, "effect": 0.0, "lo": 0.0, "hi": 0.0})
        continue
    key = f"C(tbin, Treatment(reference='15–20'))[T.{lab}]"
    lo, hi = m3.conf_int().loc[key]
    rows.append({"bin": lab, "effect": (np.exp(m3.params[key]) - 1) * 100,
                 "lo": (np.exp(lo) - 1) * 100, "hi": (np.exp(hi) - 1) * 100})
tb = pd.DataFrame(rows)
tb["days"] = city["tbin"].value_counts().reindex(labels).to_numpy()
save(tb, "temp_bins")

thr = []
for T in np.arange(0, 30.5, 0.5):
    city["heat"] = (city["tmax"] - T).clip(lower=0)
    m = smf.glm(f"n_calls ~ heat + {CONTROLS}", data=city, family=sm.families.Poisson()).fit()
    thr.append({"T": T, "aic": m.aic, "pct_5C": (np.exp(5 * m.params["heat"]) - 1) * 100})
thr = pd.DataFrame(thr)
save(thr, "threshold_search")

# ---------- H2: нагрузка и время доезда ----------
tail = con.sql(f"""
SELECT CAST(incident_datetime AS DATE) AS date, borough,
       quantile_cont(incident_response_seconds_qy, 0.9) FILTER (WHERE {VALID}) AS p90,
       median(incident_response_seconds_qy) FILTER (WHERE {VALID} AND initial_severity_level_code <= 3) AS med_sev,
       median(incident_response_seconds_qy) FILTER (WHERE {VALID} AND initial_severity_level_code >= 4) AS med_lo
FROM '{EMS}' WHERE {CLEAN} GROUP BY ALL""").df()
tail["date"] = pd.to_datetime(tail["date"])
d = daily.merge(tail, on=["date", "borough"], how="left")
g = ["borough", "year", "month", "weekday"]
d["load"] = (d["n_calls"] / d.groupby(g)["n_calls"].transform("mean") - 1) * 100
for col in ["resp_median", "p90", "med_sev", "med_lo"]:
    d[col + "_pct"] = (d[col] / d.groupby(g)[col].transform("mean") - 1) * 100
d["load_bin"] = pd.cut(d["load"], [-50, -20, -10, -5, 0, 5, 10, 15, 20, 30, 100])
lr = (d.groupby("load_bin", observed=True)
      .agg(days=("load", "size"), load=("load", "mean"),
           median_all=("resp_median_pct", "mean"), p90=("p90_pct", "mean"),
           severe=("med_sev_pct", "mean"), non_severe=("med_lo_pct", "mean"))
      .reset_index(drop=True))
save(lr, "load_response")

# ---------- H3: прогноз ----------
feats = load_features()
train, valid, test = split(feats)
m_full = train_lgbm(train, valid, FEATS_FULL)
m_nw = train_lgbm(train, valid, FEATS_NO_WEATHER)
fc = test[["date", "borough", "n_calls", "lag364", "temperature_2m_max"]].copy()
fc["borough"] = fc["borough"].astype(str)
fc["pred"] = m_full.predict(test[FEATS_FULL])
fc["pred_nw"] = m_nw.predict(test[FEATS_NO_WEATHER])
save(fc, "forecast")

# ---------- H4: пиковые часы, лето 2025–2026 ----------
hourly = con.sql(f"""
SELECT CAST(incident_datetime AS DATE) AS date, hour(incident_datetime) AS hour, borough,
       count(*) AS n_calls,
       avg(epoch(incident_close_datetime - first_assignment_datetime) / 3600)
           FILTER (WHERE epoch(incident_close_datetime - first_assignment_datetime) BETWEEN 60 AND 6*3600) AS busy_h,
       avg((epoch(first_assignment_datetime - incident_datetime) > 60)::int) AS share_wait60
FROM '{ROOT}/data/raw/ems/202[5-6]-0[6-8].parquet' WHERE {CLEAN} GROUP BY ALL""").df()
hourly["date"] = pd.to_datetime(hourly["date"])
peak = hourly[hourly["hour"].between(10, 19)].merge(
    daily[["date", "borough", "temperature_2m_max", "weekday"]], on=["date", "borough"])
peak_day = (peak.groupby(["date", "borough"])
            .agg(calls_per_hour=("n_calls", "mean"), busy_hours=("busy_h", "mean"),
                 share_wait60=("share_wait60", "mean"), tmax=("temperature_2m_max", "first"),
                 weekday=("weekday", "first"))
            .reset_index())
save(peak_day, "peak_days")

# рост вызовов в пиковый час с температурой: «излом» на 27 °C (лучший порог по AIC среди 24–30 °C);
# ниже 27 °C летом вызовы почти не меняются
PEAK_T = 27
peak_day["heat"] = (peak_day["tmax"] - PEAK_T).clip(lower=0)
slopes = []
for b, x in peak_day.groupby("borough"):
    m = smf.ols("np.log(calls_per_hour) ~ heat + C(weekday)", data=x).fit(cov_type="HC0")
    slopes.append({"borough": b, "beta": m.params["heat"], "threshold": PEAK_T,
                   "lo": m.conf_int().loc["heat", 0], "hi": m.conf_int().loc["heat", 1]})
save(pd.DataFrame(slopes), "peak_slopes")
print("готово:", OUT)
