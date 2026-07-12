"""
METHOD COMPARISON — is there a technically-better approach than our active swing engine?
Honest, evidence-first. Compares over the SAME long window:

  A) Buy & hold QQQ                      — the benchmark that's hard to beat
  B) QQQ 200-day MA trend-timing         — invested when QQQ > SMA200, else CASH.
                                            The documented "tactical" method (Meb Faber):
                                            ~buy-hold returns at ~half the drawdown.
  C) Dual-MA timing (SMA50>SMA200 gate)  — slower golden/death-cross switch.

Our active swing system is already known from portfolio_backtest.py:
  CAGR ~11.1%, MaxDD ~-13%, PF 2.05 (with the new cooldown).

Run:  python method_compare.py
"""

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

SWITCH_COST = 0.0005   # 5 bps per regime switch (generous for a monthly-ish signal)


def metrics(equity: pd.Series, start_val=10_000.0):
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = ((equity.iloc[-1] / start_val) ** (1 / years) - 1) * 100
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min() * 100
    # % of days invested (exposure)
    return cagr, max_dd, years


def run(ticker="QQQ", start="2006-01-01"):
    px = yf.Ticker(ticker).history(period="max", auto_adjust=True)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px = px[px.index >= pd.Timestamp(start)]
    ret = px.pct_change().fillna(0)

    sma200 = px.rolling(200).mean()
    sma50  = px.rolling(50).mean()

    # ---- A) Buy & hold ----
    bh = (1 + ret).cumprod() * 10_000

    # ---- B) 200-day MA timing: invested when yesterday's close > yesterday's SMA200 ----
    signal_b = (px.shift(1) > sma200.shift(1)).astype(float)   # 0/1 exposure, no look-ahead
    switches_b = signal_b.diff().abs().fillna(0)
    strat_b_ret = signal_b * ret - switches_b * SWITCH_COST
    b = (1 + strat_b_ret).cumprod() * 10_000
    exposure_b = signal_b.mean() * 100
    nsw_b = int(switches_b.sum())

    # ---- C) 50/200 golden-cross timing ----
    signal_c = (sma50.shift(1) > sma200.shift(1)).astype(float)
    switches_c = signal_c.diff().abs().fillna(0)
    strat_c_ret = signal_c * ret - switches_c * SWITCH_COST
    c = (1 + strat_c_ret).cumprod() * 10_000
    exposure_c = signal_c.mean() * 100
    nsw_c = int(switches_c.sum())

    print("\n" + "=" * 92)
    print(f"  METHOD COMPARISON on {ticker}   window {px.index[0].date()} -> {px.index[-1].date()}")
    print("=" * 92)
    for label, eq, extra in [
        ("A) Buy & hold QQQ",            bh, ""),
        (f"B) 200-MA timing",            b, f"  exposure {exposure_b:.0f}%  switches {nsw_b}"),
        (f"C) 50/200 cross timing",      c, f"  exposure {exposure_c:.0f}%  switches {nsw_c}"),
    ]:
        cagr, dd, yrs = metrics(eq)
        print(f"  {label:<26} final ${eq.iloc[-1]:>10,.0f}  CAGR {cagr:>5.1f}%  MaxDD {dd:>5.0f}%{extra}")
    print(f"\n  (our active swing system, same era: CAGR ~11.1%  MaxDD ~-13%  — from portfolio_backtest.py)")
    print(f"  Note: timing sits in CASH earning 0% here; real T-bill yield would add a bit to B/C.\n")


if __name__ == "__main__":
    run("QQQ", "2006-01-01")
    run("QQQ", "1999-03-10")   # full QQQ history incl. dot-com bust — where timing shines
