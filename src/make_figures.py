"""
Графики для README (reports/figures/*.png) из таблиц дашборда.

Запуск:  uv run python src/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from queueing import units_needed

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "dashboard"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


def temp_bins():
    tb = pd.read_parquet(DATA / "temp_bins.parquet")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [RED if v > 0 else (AXIS if v == 0 else BLUE) for v in tb["effect"]]
    ax.bar(tb["bin"], tb["effect"], color=colors, width=0.75, zorder=2)
    ax.errorbar(tb["bin"], tb["effect"], yerr=[tb["effect"] - tb["lo"], tb["hi"] - tb["effect"]],
                fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, zorder=3)
    for x, (v, lo, hi) in enumerate(zip(tb["effect"], tb["lo"], tb["hi"])):
        label = "база" if v == 0 else f"{v:+.0f} %".replace("-", "−")
        ax.annotate(label, (x, hi if v >= 0 else lo), xytext=(0, 6 if v >= 0 else -14),
                    textcoords="offset points", ha="center", color=INK2, fontsize=10)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_title("В дни выше 33 °C вызовов скорой на 16 % больше, чем при 15–20 °C")
    ax.set_xlabel("Максимальная температура дня, °C")
    ax.set_ylabel("Вызовов относительно 15–20 °C, %")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "01_heat_calls.png")


def load_response():
    lr = pd.read_parquet(DATA / "load_response.parquet")
    lr = lr[lr["days"] >= 30]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for col, name, color in [("p90", "90-й процентиль", ORANGE), ("non_severe", "медиана, несрочные", YELLOW),
                             ("median_all", "медиана, все", BLUE), ("severe", "медиана, срочные", AQUA)]:
        ax.plot(lr["load"], lr[col], "-o", color=color, linewidth=2, markersize=5, label=name)
        ax.annotate(f"{name}: {lr[col].iloc[-1]:+.0f} %", (lr["load"].iloc[-1], lr[col].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", color=INK2, fontsize=10)
    ax.axhline(0, color=AXIS, linewidth=1); ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_title("Перегрузка бьёт по самым долгим вызовам, срочные защищены")
    ax.set_xlabel("Вызовов больше обычного, %")
    ax.set_ylabel("Время доезда больше обычного, %")
    ax.set_xlim(right=lr["load"].max() + 14)
    fig.savefig(FIG / "02_load_response.png")


def forecast():
    fc = pd.read_parquet(DATA / "forecast.parquet")
    city = fc.groupby("date")[["n_calls", "pred", "lag364"]].sum().loc["2025-06-01":"2025-08-31"]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.axvspan(pd.Timestamp("2025-06-22 12:00"), pd.Timestamp("2025-06-25 12:00"), color=RED, alpha=0.1, lw=0)
    ax.text(pd.Timestamp("2025-06-26"), city["n_calls"].max() * 0.995, "волна жары 23–25.06", color=INK2, fontsize=10, va="top")
    ax.plot(city.index, city["lag364"], ":", color=MUTED, linewidth=1.5, label="как год назад")
    ax.plot(city.index, city["pred"], color=BLUE, linewidth=2, label="прогноз LightGBM")
    ax.plot(city.index, city["n_calls"], color=INK, linewidth=2, label="факт")
    ax.set_title("Прогноз вызовов на завтра, весь город, лето 2025")
    ax.set_ylabel("Вызовов в день")
    ax.legend(loc="upper right", frameon=False)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.savefig(FIG / "03_forecast.png")


def units():
    peak = pd.read_parquet(DATA / "peak_days.parquet")
    slopes = pd.read_parquet(DATA / "peak_slopes.parquet").set_index("borough")
    names = {"BROOKLYN": "Бруклин", "QUEENS": "Квинс", "MANHATTAN": "Манхэттен", "BRONX": "Бронкс",
             "RICHMOND / STATEN ISLAND": "Статен-Айленд"}
    temps = np.arange(27, 41)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n0_city = np.zeros(1); n_city = np.zeros(len(temps))
    for b in names:
        x = peak[peak["borough"] == b]; thr = slopes.loc[b, "threshold"]; beta = slopes.loc[b, "beta"]
        normal = x[x["tmax"] <= thr]
        lam0, s0, target = normal["calls_per_hour"].mean(), normal["busy_hours"].mean(), normal["share_wait60"].mean()
        n0_city += units_needed(lam0 * s0, s0, target)
        n_city += [units_needed(lam0 * np.exp(beta * max(t - thr, 0)) * s0, s0, target) for t in temps]
    extra = (n_city / n0_city - 1) * 100
    ax.plot(temps, extra, "-o", color=BLUE, linewidth=2, markersize=5)
    for t in [33, 36, 39]:
        i = list(temps).index(t)
        ax.annotate(f"{t} °C: +{extra[i]:.0f} %", (t, extra[i]), xytext=(-8, 8), textcoords="offset points",
                    color=INK2, fontsize=10, ha="right")
    ax.set_title("Сколько бригад добавить в пиковый час, чтобы ожидание не выросло")
    ax.set_xlabel("Прогноз максимальной температуры, °C")
    ax.set_ylabel("Бригад сверх обычного летнего дня, %")
    fig.savefig(FIG / "04_units.png")


if __name__ == "__main__":
    temp_bins(); load_response(); forecast(); units()
    print("готово:", sorted(p.name for p in FIG.glob("*.png")))
