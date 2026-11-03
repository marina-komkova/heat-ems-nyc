"""Формулы теории очередей (Erlang C) для расчёта бригад. Подробности — notebooks/07_erlang_c.ipynb."""

import numpy as np


def erlang_c(n: int, a: float) -> float:
    """Вероятность, что вызов ждёт: n бригад, нагрузка a эрлангов."""
    if n <= a:
        return 1.0
    b = 1.0
    for k in range(1, n + 1):
        b = a * b / (k + a * b)          # Erlang B по шагам
    return n * b / (n - a * (1 - b))


def p_wait_longer(n: int, a: float, s_hours: float, t_hours: float = 60 / 3600) -> float:
    """Доля вызовов, ждущих бригаду дольше t."""
    if n <= a:
        return 1.0
    return erlang_c(n, a) * np.exp(-(n - a) * t_hours / s_hours)


def units_needed(a: float, s_hours: float, target: float, t_hours: float = 60 / 3600) -> int:
    """Минимум бригад, при котором доля ждущих дольше t не выше target."""
    n = int(a) + 1
    while p_wait_longer(n, a, s_hours, t_hours) > target:
        n += 1
    return n
