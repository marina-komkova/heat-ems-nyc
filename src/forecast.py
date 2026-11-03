"""
Общий код прогноза вызовов на завтра (этап 5), чтобы не копировать его между ноутбуками.

    from forecast import load_features, split, train_lgbm, FEATS_FULL
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "daily_borough.parquet"

CAL = ["borough", "weekday", "month", "doy", "is_holiday", "is_dst_change"]
LAGS = ["lag1", "lag7", "lag14", "roll7", "roll28"]
WEATHER = ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
           "relative_humidity_2m_mean", "precipitation_sum", "heat"]
FEATS_FULL = CAL + LAGS + WEATHER
FEATS_NO_WEATHER = CAL + LAGS

HEAT_THRESHOLD = 10  # порог из этапа 4


def load_features() -> pd.DataFrame:
    """Таблица «день × район» с лагами, календарём и погодой, без COVID."""
    df = pd.read_parquet(DATA).sort_values(["borough", "date"]).reset_index(drop=True)
    g = df.groupby("borough")["n_calls"]
    for lag in [1, 7, 14, 364]:
        df[f"lag{lag}"] = g.shift(lag)
    df["roll7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["roll28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    df["doy"] = df["date"].dt.dayofyear
    df["heat"] = (df["temperature_2m_max"] - HEAT_THRESHOLD).clip(lower=0)
    df["borough"] = df["borough"].astype("category")
    covid = (df["date"] >= "2020-03-01") & (df["date"] < "2022-04-01")
    return df[~covid & df["roll28"].notna()].copy()


def split(df: pd.DataFrame):
    """train: 2016–2023, valid: 2024, test: 2025-01 … 2026-09."""
    train = df[df["date"] < "2024-01-01"]
    valid = df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")]
    test = df[df["date"] >= "2025-01-01"].copy()
    return train, valid, test


def train_lgbm(train, valid, features):
    model = lgb.LGBMRegressor(objective="poisson", n_estimators=3000, learning_rate=0.03,
                              num_leaves=31, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=42, verbose=-1)
    model.fit(train[features], train["n_calls"],
              eval_X=(valid[features],), eval_y=(valid["n_calls"],),
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def shift_temperature(data: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Сценарий «на delta °C теплее»: сдвигаем все температурные признаки разом."""
    out = data.copy()
    for col in ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max"]:
        out[col] = out[col] + delta
    out["heat"] = (out["temperature_2m_max"] - HEAT_THRESHOLD).clip(lower=0)
    return out
