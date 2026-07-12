"""
HYBRID BACKTEST — QQQ 50/200 regime-timing CORE + active long-only SLEEVE.
The goal: capture most of the market's growth (core) while keeping a small active
book for daily engagement (sleeve), at a lower drawdown than pure buy-hold.

Core   = hold QQQ when SMA50>SMA200 (yesterday), else CASH. (~1 switch/yr, robust.)
Sleeve = our existing long-only trend engine with the 7d cooldown (portfolio_backtest).

We blend the two equity curves at weight w (no rebalancing — honest, conservative).
Compared against: pure QQQ buy-hold, pure core-timing, pure active sleeve.

Run:  python hybrid_backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# NOTE: importing portfolio_backtest reconfigures sys.stdout to UTF-8 (once).
from portfolio_backtest import run_portfolio, UNIVERSES, START_EQUITY

SWITCH_COST = 0.0005
START = "2006-01-01"


def met(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / START_EQUITY) ** (1 / yrs) - 1) * 100
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    return cagr, dd


def qqq_timing_curve(index):
    px = yf.Ticker("QQQ").history(period="max", auto_adjust=True)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px = px[px.index >= pd.Timestamp(START)]
    ret = px.pct_change().fillna(0)
    sma50, sma200 = px.rolling(50).mean(), px.rolling(200).mean()
    sig = (sma50.shift(1) > sma200.shift(1)).astype(float)
    sw = sig.diff().abs().fillna(0)
    strat_ret = sig * ret - sw * SWITCH_COST
    core = (1 + strat_ret).cumprod() * START_EQUITY
    bh = (1 + ret).cumprod() * START_EQUITY
    return core.reindex(index, method="ffill"), bh.reindex(index, method="ffill")


def main():
    print("\n" + "=" * 96)
    print("  HYBRID BACKTEST — QQQ 50/200 timing core + active long-only sleeve")
    print("=" * 96)

    # active long-only sleeve equity curve
    active = run_portfolio(UNIVERSES["core"], start=START, long_only=True)
    sleeve = active["equity_curve"]
    core, bh = qqq_timing_curve(sleeve.index)

    c_cagr, c_dd = met(core)
    s_cagr, s_dd = met(sleeve)
    b_cagr, b_dd = met(bh)

    print(f"\n  Window {sleeve.index[0].date()} -> {sleeve.index[-1].date()}\n")
    print(f"  {'Pure QQQ buy-hold':<32} final ${bh.iloc[-1]:>10,.0f}  CAGR {b_cagr:>5.1f}%  MaxDD {b_dd:>5.0f}%")
    print(f"  {'Pure 50/200 timing (core)':<32} final ${core.iloc[-1]:>10,.0f}  CAGR {c_cagr:>5.1f}%  MaxDD {c_dd:>5.0f}%")
    print(f"  {'Pure active sleeve':<32} final ${sleeve.iloc[-1]:>10,.0f}  CAGR {s_cagr:>5.1f}%  MaxDD {s_dd:>5.0f}%")
    print(f"\n  HYBRID BLENDS (core + sleeve):")
    for w in (0.6, 0.7, 0.8):
        blend = w * core + (1 - w) * sleeve
        h_cagr, h_dd = met(blend)
        print(f"    {int(w*100)}% core / {int((1-w)*100)}% active   "
              f"final ${blend.iloc[-1]:>10,.0f}  CAGR {h_cagr:>5.1f}%  MaxDD {h_dd:>5.0f}%")
    print()


if __name__ == "__main__":
    main()
