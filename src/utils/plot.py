# Generates plots for data visualization and performance analysis.

from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

# Use a non-interactive backend so the figure renders without a display (CI,
# headless servers) and can be embedded in Streamlit via st.pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _drawdown(curve: Sequence[float]) -> np.ndarray:
    """Drawdown series (fraction, <= 0) for an equity curve."""
    values = np.asarray(curve, dtype=float)
    if len(values) == 0:
        return values
    running_max = np.maximum.accumulate(values)
    # Guard against zero/negative peaks.
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (values - running_max) / running_max
    return np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)


def plot_equity_curves(
    curves: Dict[str, Sequence[float]],
    title: str = "Backtest: Agent vs. Baselines",
    dates: Optional[Sequence] = None,
) -> Figure:
    """Plot equity curves and their drawdowns for a set of strategies.

    Args:
        curves: Mapping of strategy name -> equity curve (list of portfolio values).
        title: Figure title.
        dates: Optional shared x-axis values; falls back to the time step index.

    Returns:
        A matplotlib ``Figure`` (so callers can both save it and render it via
        ``st.pyplot``).
    """
    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    for name, curve in curves.items():
        values = np.asarray(curve, dtype=float)
        if len(values) == 0:
            continue
        x = np.asarray(dates[: len(values)]) if dates is not None else np.arange(len(values))
        ax_eq.plot(x, values, label=name, linewidth=1.6)
        ax_dd.plot(x, _drawdown(values) * 100.0, label=name, linewidth=1.2)

    ax_eq.set_title(title)
    ax_eq.set_ylabel("Portfolio value ($)")
    ax_eq.legend(loc="best")
    ax_eq.grid(True, alpha=0.3)

    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Trading day" if dates is None else "Date")
    ax_dd.grid(True, alpha=0.3)
    ax_dd.axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    return fig
