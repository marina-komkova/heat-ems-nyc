"""
Готовит небольшие таблицы для дашборда (data/processed/dashboard/*.parquet),
чтобы сам дашборд открывался мгновенно и не пересчитывал модели.

Работает только с агрегатами из data/processed (их обновляет src/update_data.py),
поэтому запускается и на GitHub Actions, где сырых данных нет.

Запуск:  uv run python src/build_dashboard_data.py   (≈ 1 минута)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from forecast import FEATS_FULL, FEATS_NO_WEATHER, load_features, split, train_lgbm
from queueing import units_needed

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)
CONTROLS = "C(month) + C(weekday) + C(year) + is_holiday + is_dst_change"

daily_all = pd.read_parquet(PROC / "daily_borough.parquet")
daily = daily_all[~daily_all["is_covid"]].copy()
kpi = {"date_min": str(daily_all["date"].min().date()), "date_max": str(daily_all["date"].max().date()),
       "n_calls_total": int(daily_all["n_calls"].sum()), "n_days_clean": int(daily["date"].nunique())}


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

best_T = float(thr.loc[thr["aic"].idxmin(), "T"])
city["heat"] = (city["tmax"] - best_T).clip(lower=0)
m_best = smf.glm(f"n_calls ~ heat + {CONTROLS}", data=city, family=sm.families.Poisson()).fit(cov_type="HC0")
lo, hi = m_best.conf_int().loc["heat"]
kpi.update(threshold=best_T, pct_5C=(np.exp(5 * m_best.params["heat"]) - 1) * 100,
           pct_5C_lo=(np.exp(5 * lo) - 1) * 100, pct_5C_hi=(np.exp(5 * hi) - 1) * 100,
           hot_bin_effect=float(tb.loc[tb["bin"] == "> 33", "effect"].iloc[0]))

# ---------- H2: нагрузка и время доезда ----------
tail = pd.read_parquet(PROC / "daily_tail.parquet")
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
top = lr[lr["days"] >= 30].iloc[-1]
kpi.update(tail_p90=float(top["p90"]), tail_median=float(top["median_all"]), tail_load=float(top["load"]))

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

# ---------- H4: пиковые часы двух последних полных лет (июнь–август) ----------
hourly = pd.read_parquet(PROC / "hourly_peak.parquet")
last = daily_all["date"].max()
last_summer = last.year if last >= pd.Timestamp(f"{last.year}-08-31") else last.year - 1
summers = [last_summer - 1, last_summer]
hourly = hourly[hourly["date"].dt.year.isin(summers) & hourly["date"].dt.month.isin([6, 7, 8])]
kpi["summers"] = summers
peak = hourly.merge(
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
slopes = pd.DataFrame(slopes)
save(slopes, "peak_slopes")

# сколько бригад нужно в дни 33 °C+ по городу (та же модель, что в калькуляторе дашборда)
n0 = n_hot = 0
for b, x in peak_day.groupby("borough"):
    beta = slopes.set_index("borough").loc[b, "beta"]
    normal = x[x["tmax"] <= PEAK_T]
    lam0, s0, target = normal["calls_per_hour"].mean(), normal["busy_hours"].mean(), normal["share_wait60"].mean()
    t_hot = x.loc[x["tmax"] >= 33, "tmax"].mean()
    n0 += units_needed(lam0 * s0, s0, target)
    n_hot += units_needed(lam0 * np.exp(beta * (t_hot - PEAK_T)) * s0, s0, target)
kpi["units_hot_pct"] = (n_hot / n0 - 1) * 100

(OUT / "kpis.json").write_text(json.dumps(kpi, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("KPI:", {k: round(v, 1) if isinstance(v, float) else v for k, v in kpi.items()})
print("готово:", OUT)
