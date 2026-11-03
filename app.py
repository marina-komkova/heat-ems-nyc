"""
Дашборд «Жара и скорая помощь Нью-Йорка».

Запуск:  uv run streamlit run app.py
Данные для него готовит:  uv run python src/build_dashboard_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from queueing import p_wait_longer, units_needed  # noqa: E402

DATA = ROOT / "data" / "processed" / "dashboard"

st.set_page_config(page_title="Жара и скорая Нью-Йорка", page_icon="🚑", layout="wide")

# ---------------------------------------------------------------- оформление
THEMES = {
    "light": dict(surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", border="rgba(11,11,11,0.10)",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100",
                  cool="#2a78d6", warm="#e34948", neutral="#c3c2b7"),
    "dark": dict(surface="#1a1a19", page="#0d0d0d", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", border="rgba(255,255,255,0.10)",
                 s1="#3987e5", s2="#d95926", s3="#199e70", s4="#c98500",
                 cool="#3987e5", warm="#e66767", neutral="#383835"),
}
mode = getattr(st.context.theme, "type", None) or "light"
C = THEMES["dark" if mode == "dark" else "light"]

st.markdown(f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 8px 0 4px; }}
.kpi {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 12px; padding: 16px 18px; }}
.kpi .label {{ color: {C['ink2']}; font-size: 0.85rem; line-height: 1.3; min-height: 2.6em; }}
.kpi .value {{ color: {C['ink']}; font-size: 2rem; font-weight: 650; line-height: 1.15; margin: 6px 0 2px; }}
.kpi .note {{ color: {C['muted']}; font-size: 0.8rem; line-height: 1.3; }}
.lede {{ color: {C['ink2']}; font-size: 1.05rem; max-width: 760px; }}
.takeaway {{ border-left: 3px solid {C['s1']}; padding: 2px 0 2px 14px; color: {C['ink']}; margin: 4px 0 12px; }}
</style>
""", unsafe_allow_html=True)


def kpis(items):
    cells = "".join(
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="note">{note}</div></div>' for label, value, note in items)
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)


def style(fig, height=420, y_title="", x_title="", legend=True):
    fig.update_layout(
        height=height, margin=dict(l=72, r=56, t=40, b=64),
        paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=C["ink2"], size=13),
        hoverlabel=dict(bgcolor=C["surface"], bordercolor=C["axis"], font=dict(color=C["ink"])),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=C["ink2"])),
        barcornerradius=4,
    )
    fig.update_xaxes(title=dict(text=x_title, standoff=14), automargin=True, gridcolor=C["grid"], linecolor=C["axis"], zerolinecolor=C["axis"],
                     tickfont=dict(color=C["muted"]), title_font=dict(color=C["ink2"]))
    fig.update_yaxes(title=dict(text=y_title, standoff=14), automargin=True, gridcolor=C["grid"], linecolor=C["axis"], zerolinecolor=C["axis"],
                     tickfont=dict(color=C["muted"]), title_font=dict(color=C["ink2"]))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


BOROUGHS = {
    "BROOKLYN": "Бруклин", "QUEENS": "Квинс", "MANHATTAN": "Манхэттен",
    "BRONX": "Бронкс", "RICHMOND / STATEN ISLAND": "Статен-Айленд",
}


@st.cache_data
def load(name):
    return pd.read_parquet(DATA / f"{name}.parquet")


# ---------------------------------------------------------------- шапка
st.title("Жара и скорая помощь Нью-Йорка")
st.markdown(
    '<p class="lede">Как температура влияет на число вызовов скорой, время доезда и нужное число бригад. '
    "16 млн вызовов NYC EMS за 2016–2026 годы и погода по пяти районам; период COVID исключён.</p>",
    unsafe_allow_html=True)

kpis([
    ("Порог, с которого жара добавляет вызовы", "10 °C", "максимальная температура дня, выбран по AIC"),
    ("Вызовов на каждые +5 °C выше порога", "+4,2 %", "95 % ДИ 3,9–4,5 %, с поправкой на сезон"),
    ("Доезд самых долгих 10 % вызовов при перегрузке", "+45 %", "при +20–30 % вызовов; медиана — лишь +13 %"),
    ("Бригад нужно в дни 33 °C+ в пиковые часы", "+12 %", "чтобы ожидание не выросло (Erlang C)"),
])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Жара → вызовы", "Перегрузка → доезд", "Прогноз на завтра", "Калькулятор бригад", "О проекте"])

# ---------------------------------------------------------------- 1. жара и вызовы
with tab1:
    tb = load("temp_bins")
    st.markdown('<div class="takeaway">В дни выше 33 °C вызовов на <b>16 %</b> больше, чем при комфортных '
                "15–20 °C, — при сравнении с днями того же месяца, дня недели и года.</div>",
                unsafe_allow_html=True)
    colors = [C["warm"] if v > 0 else (C["neutral"] if v == 0 else C["cool"]) for v in tb["effect"]]
    fig = go.Figure(go.Bar(
        x=tb["bin"], y=tb["effect"], marker_color=colors,
        error_y=dict(type="data", symmetric=False, array=tb["hi"] - tb["effect"],
                     arrayminus=tb["effect"] - tb["lo"], color=C["muted"], thickness=1.2, width=4),
        customdata=np.c_[tb["lo"], tb["hi"], tb["days"]],
        hovertemplate="<b>%{x} °C</b><br>%{y:+.1f} % вызовов<br>95 % ДИ %{customdata[0]:+.1f}…%{customdata[1]:+.1f}"
                      "<br>дней: %{customdata[2]}<extra></extra>",
    ))
    for _, r in tb.iterrows():   # подпись за концом «уса», чтобы не налезала на него
        up = r["effect"] >= 0
        fig.add_annotation(x=r["bin"], y=r["hi"] if up else r["lo"], yshift=12 if up else -12,
                           text="база" if r["effect"] == 0 else f"{r['effect']:+.0f} %".replace("-", "−"),
                           showarrow=False, font=dict(color=C["ink2"], size=12))
    fig.add_hline(y=0, line_color=C["axis"], line_width=1)
    show(style(fig, y_title="Вызовов относительно дней 15–20 °C, %",
               x_title="Максимальная температура дня, °C", legend=False))
    st.caption("Регрессия Пуассона по городу целиком, 3 196 дней без COVID; контроль месяца, дня недели, "
               "года, праздников и перевода часов. Усы — 95 % доверительный интервал (устойчивые ошибки).")

    with st.expander("Как выбран порог 10 °C"):
        thr = load("threshold_search")
        best = thr.loc[thr["aic"].idxmin()]
        f2 = go.Figure(go.Scatter(x=thr["T"], y=thr["aic"], mode="lines", line=dict(color=C["s1"], width=2),
                                  hovertemplate="порог %{x} °C<br>AIC %{y:,.0f}<extra></extra>"))
        f2.add_trace(go.Scatter(x=[best["T"]], y=[best["aic"]], mode="markers+text",
                                marker=dict(size=10, color=C["s1"], line=dict(color=C["surface"], width=2)),
                                text=[f"  минимум: {best['T']:.0f} °C"], textposition="middle right",
                                textfont=dict(color=C["ink"]), hoverinfo="skip"))
        show(style(f2, height=300, y_title="AIC (меньше — лучше)", x_title="Порог T, °C", legend=False))
        st.caption("Перебрали пороги от 0 до 30 °C: модель «вызовы растут на каждый градус выше T». "
                   "Лучше всего данные описывает T = 10 °C.")

# ---------------------------------------------------------------- 2. перегрузка и доезд
with tab2:
    lr = load("load_response")
    lr = lr[lr["days"] >= 30]
    st.markdown('<div class="takeaway">Типичный вызов почти не страдает от перегрузки, а самые долгие — '
                "страдают сильно: при перегрузке на 20–30 % ожидание «хвоста» растёт примерно на <b>45 %</b>. "
                "Срочные вызовы защищены приоритетом диспетчеров.</div>", unsafe_allow_html=True)
    series = [("p90", "90-й процентиль (самые долгие 10 %)", C["s2"]),
              ("non_severe", "Медиана, несрочные вызовы", C["s4"]),
              ("median_all", "Медиана, все вызовы", C["s1"]),
              ("severe", "Медиана, срочные (угроза жизни)", C["s3"])]
    fig = go.Figure()
    for col, name, color in series:
        fig.add_trace(go.Scatter(
            x=lr["load"], y=lr[col], name=name, mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=C["surface"], width=2)),
            hovertemplate=f"{name}: %{{y:+.1f}} %<extra></extra>"))
        fig.add_annotation(x=lr["load"].iloc[-1], y=lr[col].iloc[-1], text=f"{lr[col].iloc[-1]:+.0f} %",
                           xanchor="left", xshift=8, showarrow=False, font=dict(color=C["ink2"]))
    fig.add_hline(y=0, line_color=C["axis"], line_width=1)
    fig.add_vline(x=0, line_color=C["axis"], line_width=1)
    fig.update_layout(hovermode="x unified")
    show(style(fig, height=460, y_title="Время доезда больше обычного, %",
               x_title="Вызовов больше обычного для этого района, месяца и дня недели, %"))
    st.caption("День × район, 2016–2026 без COVID. «Обычное» значение — среднее для того же района, года, "
               "месяца и дня недели. Показаны группы, где не меньше 30 дней.")

# ---------------------------------------------------------------- 3. прогноз
with tab3:
    fc = load("forecast")
    c1, c2 = st.columns([1, 2])
    with c1:
        area = st.selectbox("Район", ["Весь город"] + list(BOROUGHS.values()))
    with c2:
        dmin, dmax = fc["date"].min().date(), fc["date"].max().date()
        period = st.slider("Период", min_value=dmin, max_value=dmax,
                           value=(pd.Timestamp("2025-06-01").date(), pd.Timestamp("2025-08-31").date()),
                           format="DD.MM.YYYY")
    sel = fc if area == "Весь город" else fc[fc["borough"].map(BOROUGHS) == area]
    agg = sel.groupby("date")[["n_calls", "pred", "pred_nw", "lag364"]].sum()
    agg["tmax"] = sel.groupby("date")["temperature_2m_max"].mean()
    view = agg.loc[str(period[0]):str(period[1])]

    def mape(p, d=agg):
        return ((d["n_calls"] - d[p]).abs() / d["n_calls"]).mean() * 100

    def ru(v):
        return f"{v:.1f}".replace(".", ",")

    hot = agg[agg["tmax"] > 30]
    kpis([
        ("Ошибка LightGBM, весь тест 2025–2026", f"{ru(mape('pred'))} %", "средняя ошибка прогноза на завтра, MAPE"),
        ("Ошибка правила «как год назад»", f"{ru(mape('lag364'))} %", "тот же день недели год назад"),
        ("Ошибка LightGBM в дни выше 30 °C", f"{ru(mape('pred', hot))} %",
         f"без погоды: {ru(mape('pred_nw', hot))} %"),
    ])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.index, y=view["n_calls"], name="Факт", line=dict(color=C["ink"], width=2),
                             hovertemplate="факт: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=view.index, y=view["pred"], name="LightGBM с погодой",
                             line=dict(color=C["s1"], width=2), hovertemplate="LightGBM: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=view.index, y=view["lag364"], name="Как год назад",
                             line=dict(color=C["muted"], width=1.5, dash="dot"),
                             hovertemplate="год назад: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=view.index, y=view["tmax"], name="tmax", visible=False,
                             hovertemplate="tmax: %{y:.1f} °C<extra></extra>"))
    for d0, d1 in [("2025-06-22", "2025-06-25")]:
        if pd.Timestamp(d0) >= view.index.min() and pd.Timestamp(d1) <= view.index.max():
            fig.add_vrect(x0=pd.Timestamp(d0) + pd.Timedelta(hours=12), x1=pd.Timestamp(d1) + pd.Timedelta(hours=12),
                          fillcolor=C["warm"], opacity=0.10, line_width=0,
                          annotation_text="волна жары", annotation_position="top left",
                          annotation_font_color=C["ink2"])
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat="%d.%m.%y")
    show(style(fig, height=440, y_title="Вызовов в день"))
    st.caption("Прогноз делается вечером на следующий день. Обучение: 2016–2023, остановка: 2024, "
               "тест: 01.2025–09.2026. Погода — фактическая; с реальным прогнозом погоды ошибка будет немного выше.")

# ---------------------------------------------------------------- 4. калькулятор
with tab4:
    peak = load("peak_days")
    slopes = load("peak_slopes").set_index("borough")
    st.markdown('<div class="takeaway">Выберите район и прогноз температуры — модель очередей Erlang C '
                "покажет, сколько бригад нужно в пиковый час (10:00–20:00), чтобы доля вызовов, ждущих "
                "бригаду дольше минуты, не выросла по сравнению с обычным летним днём.</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        b_ru = st.selectbox("Район ", list(BOROUGHS.values()))
    b = {v: k for k, v in BOROUGHS.items()}[b_ru]
    x = peak[peak["borough"] == b]
    thr = slopes.loc[b, "threshold"]
    normal = x[x["tmax"] <= thr]
    lam0, s0 = normal["calls_per_hour"].mean(), normal["busy_hours"].mean()
    with c2:
        temp = st.slider("Прогноз максимальной температуры, °C", 24, 42, 35)
    with c3:
        target = st.slider("Допустимая доля ждущих дольше 1 мин, %", 5, 70,
                           int(round(normal["share_wait60"].mean() * 100)),
                           help="По умолчанию — как в обычный летний день (до 27 °C) в этом районе.") / 100

    beta = slopes.loc[b, "beta"]

    def calls_at(t):
        return lam0 * np.exp(beta * max(t - thr, 0))

    n0 = units_needed(lam0 * s0, s0, target)
    lam = calls_at(temp)
    n = units_needed(lam * s0, s0, target)
    kpis([
        ("Вызовов в пиковый час", f"{lam:.0f}", f"обычный летний день (до {thr:.0f} °C): {lam0:.0f}"),
        ("Бригад нужно (эффективных)", f"{n}", f"обычный летний день: {n0}"),
        ("Добавить к обычной смене", f"+{n - n0}", f"+{(n / n0 - 1) * 100:.0f} %"),
    ])

    temps = np.arange(24, 43)
    extra = [(units_needed(calls_at(t) * s0, s0, target) / n0 - 1) * 100 for t in temps]
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(x=temps, y=extra, mode="lines", line=dict(color=C["s1"], width=2),
                                   hovertemplate="%{x} °C: +%{y:.0f} % бригад<extra></extra>"))
        fig.add_trace(go.Scatter(x=[temp], y=[(n / n0 - 1) * 100], mode="markers",
                                 marker=dict(size=11, color=C["s1"], line=dict(color=C["surface"], width=2)),
                                 hoverinfo="skip"))
        st.markdown("**Сколько бригад добавить, % к обычной смене**")
        show(style(fig, height=340, x_title="Прогноз tmax, °C", legend=False))
    with right:
        ns = np.arange(int(lam0 * s0) - 1, int(lam * s0) + 18)
        fig = go.Figure()
        for name, lam_, color in [(f"обычный день (до {thr:.0f} °C)", lam0, C["s1"]), (f"{temp} °C", lam, C["s2"])]:
            fig.add_trace(go.Scatter(x=ns, y=[p_wait_longer(int(k), lam_ * s0, s0) * 100 for k in ns], name=name,
                                     line=dict(color=color, width=2),
                                     hovertemplate=f"{name}<br>%{{x}} бригад: %{{y:.0f}} % ждут<extra></extra>"))
        fig.add_hline(y=target * 100, line_dash="dot", line_color=C["muted"],
                      annotation_text="цель", annotation_font_color=C["ink2"])
        st.markdown("**Доля вызовов, ждущих бригаду дольше 1 мин, %**")
        style(fig, height=360, x_title="Бригад на смене")
        fig.update_layout(legend=dict(orientation="h", y=-0.28, yanchor="top", x=0), margin=dict(b=96))
        show(fig)
    st.caption(f"Данные: пиковые часы лета 2025–2026. Бригада занята одним вызовом в среднем {s0 * 60:.0f} мин "
               f"(от назначения до закрытия). До {thr:.0f} °C вызовы в пиковый час почти не меняются, выше — "
               f"+{(np.exp(beta) - 1) * 100:.1f} % на каждый градус в этом районе. «Эффективные» бригады — модельный эквивалент, а не "
               "реальные машины: сравнивать стоит разницу, а не абсолютные числа.")

# ---------------------------------------------------------------- 5. о проекте
with tab5:
    st.markdown(Path(ROOT / "reports" / "about.md").read_text(encoding="utf-8"))
