"""
EXPERIMENT — anti-whipsaw filters on the long-only trend engine.
Tests two additions against the current baseline, over the full 20y portfolio sim:
  1. ADX(14) trend-strength gate — only enter when a real trend exists (ADX >= min).
  2. Cooldown — after a ticker stops us out, don't re-enter it for N days
     (directly targets the live failure: AMZN bought+stopped TWICE in June 2026).

Decision rule (project ethos): keep a filter ONLY if it improves risk-adjusted
return (PF / MaxDD) without gutting CAGR. Curve-fitting to recent chop is rejected.

Run:  python test_filters.py
"""

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import (
    compute_indicators, generate_signal,
    ATR_STOP_MULT, TRAIL_STOP_PCT, MAX_HOLD_DAYS,
)

START_EQUITY   = 10_000.0
RISK_PER_TRADE = 0.02
COMMISSION_RT  = 5.0
TICKERS        = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN"]
START          = "2006-01-01"


def load_data():
    data = {}
    for tk in TICKERS:
        raw = yf.Ticker(tk).history(period="max", auto_adjust=True)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        if len(raw) < 250:
            continue
        data[tk] = compute_indicators(raw)
    qqq = yf.Ticker("QQQ").history(period="max", auto_adjust=True)
    qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
    ma50, ma200 = qqq["Close"].rolling(50).mean(), qqq["Close"].rolling(200).mean()
    regime = pd.DataFrame({
        "up":   (qqq["Close"] > ma50) & (ma50 > ma200),
        "down": (qqq["Close"] < ma50) & (ma50 < ma200),
    })
    return data, regime, qqq["Close"]


def run(data, regime, adx_min=0.0, cooldown_days=0):
    max_positions = len(data)
    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp(START)]

    cash = START_EQUITY
    positions, trades, equity_curve, dates_curve = {}, [], [], []
    last_stop = {}   # ticker -> date it last stopped us out

    def mkt(d):
        if d in regime.index:
            return {"market_up": bool(regime.loc[d, "up"]), "market_down": bool(regime.loc[d, "down"])}
        return {"market_up": False, "market_down": False}

    for d in all_dates:
        market = mkt(d)

        eq = cash
        for tk, p in positions.items():
            px = float(data[tk].loc[d, "Close"]) if d in data[tk].index else p["entry"]
            eq += p["shares"] * px

        # exits
        for tk in list(positions.keys()):
            if d not in data[tk].index:
                continue
            bar = data[tk].loc[d]
            p = positions[tk]
            price, low = float(bar["Close"]), float(bar["Low"])
            held = (d - p["entry_date"]).days
            exit_price, reason = None, None
            if low <= p["sl"]:
                exit_price, reason = p["sl"], "stop"
            else:
                if (price - p["entry"]) / p["entry"] > 0.03:
                    p["sl"] = max(p["sl"], price * (1 - TRAIL_STOP_PCT))
                if held >= MAX_HOLD_DAYS:
                    exit_price, reason = price, "time"
            if exit_price is not None:
                gross = (exit_price - p["entry"]) * p["shares"]
                net = gross - COMMISSION_RT
                cash += net + p["entry"] * p["shares"]
                trades.append({"ticker": tk, "pnl": net, "reason": reason, "days": held})
                if reason == "stop":
                    last_stop[tk] = d
                del positions[tk]

        # entries
        if len(positions) < max_positions:
            candidates = []
            for tk, df in data.items():
                if tk in positions or d not in df.index:
                    continue
                loc = df.index.get_loc(d)
                if loc < 200:
                    continue
                # cooldown gate
                if cooldown_days and tk in last_stop and (d - last_stop[tk]).days < cooldown_days:
                    continue
                # ADX trend-strength gate
                if adx_min:
                    adx = df.iloc[loc]["ADX14"]
                    if np.isnan(adx) or adx < adx_min:
                        continue
                sig = generate_signal(df.iloc[: loc + 1], tk, market=market)
                if sig["signal"] == "LONG":
                    candidates.append((abs(sig["score"]), tk, sig))

            candidates.sort(reverse=True)
            for _, tk, sig in candidates:
                if len(positions) >= max_positions:
                    break
                df = data[tk]
                bar = df.loc[d]
                price = float(bar["Close"])
                atr = float(bar["ATR14"]) if not np.isnan(bar["ATR14"]) else price * 0.02
                sl = price - ATR_STOP_MULT * atr
                risk_ps = price - sl
                if risk_ps < 0.01:
                    continue
                shares = (eq * RISK_PER_TRADE) / risk_ps
                shares = min(shares, (eq * 0.80 / max_positions) / price)
                cost = shares * price
                if shares <= 0 or cash < cost + COMMISSION_RT:
                    continue
                cash -= cost + COMMISSION_RT
                positions[tk] = {"entry": price, "shares": shares, "sl": sl, "entry_date": d}

        equity_curve.append(eq)
        dates_curve.append(d)

    ec = pd.Series(equity_curve, index=dates_curve)
    final = equity_curve[-1]
    years = (dates_curve[-1] - dates_curve[0]).days / 365.25
    cagr = ((final / START_EQUITY) ** (1 / max(years, 0.5)) - 1) * 100
    peak = ec.cummax()
    max_dd = ((ec - peak) / peak).min() * 100
    wins = [t for t in trades if t["pnl"] > 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    return {
        "final": final, "cagr": cagr, "max_dd": max_dd, "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "pf": gw / gl if gl > 0 else float("inf"),
        "calmar": cagr / abs(max_dd) if max_dd else 0,
    }


def fmt(label, r):
    pf = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "inf"
    print(f"  {label:<26} final ${r['final']:>9,.0f}  CAGR {r['cagr']:>5.1f}%  "
          f"MaxDD {r['max_dd']:>5.0f}%  Calmar {r['calmar']:>4.2f}  "
          f"trades {r['trades']:>4}  win {r['win_rate']:>4.1f}%  PF {pf}")


def main():
    print("\n" + "=" * 104)
    print(f"  ANTI-WHIPSAW FILTER EXPERIMENT — long-only, {TICKERS}, from {START}")
    print("=" * 104 + "\n")
    data, regime, _ = load_data()

    configs = [
        ("BASELINE (no filter)",        dict(adx_min=0, cooldown_days=0)),
        ("Cooldown 5d",                 dict(adx_min=0, cooldown_days=5)),
        ("Cooldown 7d",                 dict(adx_min=0, cooldown_days=7)),
        ("Cooldown 10d",                dict(adx_min=0, cooldown_days=10)),
        ("Cooldown 12d",                dict(adx_min=0, cooldown_days=12)),
        ("Cooldown 15d",                dict(adx_min=0, cooldown_days=15)),
    ]
    for label, kw in configs:
        fmt(label, run(data, regime, **kw))
    print()


if __name__ == "__main__":
    main()
