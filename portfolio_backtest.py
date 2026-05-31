"""
PORTFOLIO-LEVEL BACKTEST — the honest "what would we actually make" test.
Simulates the REAL system: all tickers share one $10k account, max-N concurrent
positions, QQQ regime gate, trend-following entries, ATR stops, ride-the-trend exits.

Unlike research_backtest.py (each ticker in isolation), this captures the true
diversification effect and the real equity curve through 2008 / 2020 / 2022.

Run:  python portfolio_backtest.py
      python portfolio_backtest.py --rs 3        # only trade the 3 strongest names
      python portfolio_backtest.py --universe big # expanded ticker universe
"""

import sys, io, argparse, warnings
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

UNIVERSES = {
    "core": ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN"],
    "big":  ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "AMD", "TSLA", "NFLX"],
}


def load_data(tickers):
    """Fetch + index all tickers and the QQQ regime series. Returns (data, regime)."""
    data = {}
    for tk in tickers:
        try:
            raw = yf.Ticker(tk).history(period="max", auto_adjust=True)
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            if len(raw) < 250:
                continue
            data[tk] = compute_indicators(raw)
        except Exception as e:
            print(f"  skip {tk}: {e}")
    qqq = yf.Ticker("QQQ").history(period="max", auto_adjust=True)
    qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
    ma50, ma200 = qqq["Close"].rolling(50).mean(), qqq["Close"].rolling(200).mean()
    regime = pd.DataFrame({
        "up":   (qqq["Close"] > ma50) & (ma50 > ma200),
        "down": (qqq["Close"] < ma50) & (ma50 < ma200),
    })
    return data, regime


def run_portfolio(tickers, rs_top=None, max_positions=None, start=None, long_only=False):
    data, regime = load_data(tickers)
    if not data:
        print("  no data"); return None
    max_positions = max_positions or len(data)

    # common trading calendar = union of all dates, sorted
    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    if start:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(start)]

    cash = START_EQUITY
    positions = {}                # ticker -> dict
    trades = []
    equity_curve = []
    dates_curve = []

    def mkt(d):
        if d in regime.index:
            return {"market_up": bool(regime.loc[d, "up"]), "market_down": bool(regime.loc[d, "down"])}
        return {"market_up": False, "market_down": False}

    for d in all_dates:
        market = mkt(d)

        # ---- mark-to-market equity ----
        # Long: full cost was deducted from cash at open → position worth shares*px.
        # Short: P&L-only model (only commission left cash at open) → worth (entry-px)*shares.
        eq = cash
        for tk, p in positions.items():
            px = float(data[tk].loc[d, "Close"]) if d in data[tk].index else p["entry"]
            eq += p["shares"] * px if p["side"] == "long" else (p["entry"] - px) * p["shares"]

        # ---- manage exits ----
        for tk in list(positions.keys()):
            if d not in data[tk].index:
                continue
            bar = data[tk].loc[d]
            p = positions[tk]
            price, high, low = float(bar["Close"]), float(bar["High"]), float(bar["Low"])
            held = (d - p["entry_date"]).days
            exit_price, reason = None, None
            if p["side"] == "long":
                if low <= p["sl"]:
                    exit_price, reason = p["sl"], "stop"
                else:
                    if (price - p["entry"]) / p["entry"] > 0.03:
                        p["sl"] = max(p["sl"], price * (1 - TRAIL_STOP_PCT))
                    if held >= MAX_HOLD_DAYS:
                        exit_price, reason = price, "time"
            else:
                if high >= p["sl"]:
                    exit_price, reason = p["sl"], "stop"
                else:
                    if (p["entry"] - price) / p["entry"] > 0.03:
                        p["sl"] = min(p["sl"], price * (1 + TRAIL_STOP_PCT))
                    if held >= MAX_HOLD_DAYS:
                        exit_price, reason = price, "time"
            if exit_price is not None:
                gross = (exit_price - p["entry"]) * p["shares"] if p["side"] == "long" else (p["entry"] - exit_price) * p["shares"]
                net = gross - COMMISSION_RT
                cash += net + (p["entry"] * p["shares"] if p["side"] == "long" else 0)
                trades.append({"ticker": tk, "side": p["side"], "pnl": net, "reason": reason, "days": held})
                del positions[tk]

        # ---- look for entries ----
        if len(positions) < max_positions:
            # rank candidates by trend strength (relative strength) if rs_top set
            candidates = []
            for tk, df in data.items():
                if tk in positions or d not in df.index:
                    continue
                loc = df.index.get_loc(d)
                if loc < 200:
                    continue
                sig = generate_signal(df.iloc[: loc + 1], tk, market=market)
                allowed = ("LONG",) if long_only else ("LONG", "SHORT")
                if sig["signal"] in allowed:
                    candidates.append((abs(sig["score"]), tk, sig))

            candidates.sort(reverse=True)   # strongest signal first
            if rs_top:
                candidates = candidates[:rs_top]

            for _, tk, sig in candidates:
                if len(positions) >= max_positions:
                    break
                df = data[tk]
                bar = df.loc[d]
                price = float(bar["Close"])
                atr = float(bar["ATR14"]) if not np.isnan(bar["ATR14"]) else price * 0.02
                side = "long" if sig["signal"] == "LONG" else "short"
                sl = price - ATR_STOP_MULT * atr if side == "long" else price + ATR_STOP_MULT * atr
                risk_ps = abs(price - sl)
                if risk_ps < 0.01:
                    continue
                shares = (eq * RISK_PER_TRADE) / risk_ps
                shares = min(shares, (eq * 0.80 / max_positions) / price)
                cost = shares * price if side == "long" else 0
                if shares <= 0 or cash < cost + COMMISSION_RT:
                    continue
                cash -= cost + COMMISSION_RT
                positions[tk] = {"side": side, "entry": price, "shares": shares,
                                 "sl": sl, "entry_date": d}

        equity_curve.append(eq)
        dates_curve.append(d)

    # ---- metrics ----
    ec = pd.Series(equity_curve, index=dates_curve)
    final = equity_curve[-1]
    years = (dates_curve[-1] - dates_curve[0]).days / 365.25
    cagr = ((final / START_EQUITY) ** (1 / max(years, 0.5)) - 1) * 100
    peak = ec.cummax()
    max_dd = ((ec - peak) / peak).min() * 100
    wins = [t for t in trades if t["pnl"] > 0]
    gw = sum(t["pnl"] for t in wins); gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))

    # buy & hold QQQ benchmark over same window
    qqq = yf.Ticker("QQQ").history(period="max", auto_adjust=True)
    qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
    qqq_win = qqq[(qqq.index >= dates_curve[0]) & (qqq.index <= dates_curve[-1])]["Close"]
    qqq_ret = (qqq_win.iloc[-1] / qqq_win.iloc[0] - 1) * 100
    qqq_cagr = ((qqq_win.iloc[-1] / qqq_win.iloc[0]) ** (1 / max(years, 0.5)) - 1) * 100
    qqq_peak = qqq_win.cummax(); qqq_dd = ((qqq_win - qqq_peak) / qqq_peak).min() * 100

    return {
        "final": final, "total_ret": (final / START_EQUITY - 1) * 100, "cagr": cagr,
        "max_dd": max_dd, "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "pf": gw / gl if gl > 0 else float("inf"),
        "years": years, "start": dates_curve[0].date(), "end": dates_curve[-1].date(),
        "qqq_ret": qqq_ret, "qqq_cagr": qqq_cagr, "qqq_dd": qqq_dd,
        "equity_curve": ec,
    }


def fmt(label, r):
    pf = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "inf"
    print(f"  {label:<28} final ${r['final']:>9,.0f}  ret {r['total_ret']:>7.0f}%  "
          f"CAGR {r['cagr']:>5.1f}%  MaxDD {r['max_dd']:>5.0f}%  "
          f"trades {r['trades']:>4}  win {r['win_rate']:>4.1f}%  PF {pf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=int, default=0, help="only trade top-N strongest names")
    ap.add_argument("--universe", default="core", choices=list(UNIVERSES))
    ap.add_argument("--maxpos", type=int, default=0)
    ap.add_argument("--start", default="2006-01-01", help="backtest start (when full universe exists)")
    ap.add_argument("--longonly", action="store_true", help="never short")
    ap.add_argument("--compare", action="store_true", help="run all variants + hybrid blend")
    args = ap.parse_args()

    tickers = UNIVERSES[args.universe]
    print("\n" + "=" * 100)
    print(f"  PORTFOLIO BACKTEST — universe={args.universe} {tickers}  from {args.start}")
    print("=" * 100)

    if not args.compare:
        base = run_portfolio(tickers, rs_top=(args.rs or None), max_positions=(args.maxpos or None),
                             start=args.start, long_only=args.longonly)
        if not base:
            return
        print(f"\n  Window: {base['start']} -> {base['end']}  ({base['years']:.0f} years)\n")
        fmt("Our system", base)
        print(f"  {'Buy & hold QQQ':<28} ret {base['qqq_ret']:>7.0f}%  CAGR {base['qqq_cagr']:>5.1f}%  "
              f"MaxDD {base['qqq_dd']:>5.0f}%   (benchmark)")
        print()
        return

    # ── COMPARE MODE: long/short vs long-only, + hybrid blends vs QQQ ──────────
    ls = run_portfolio(tickers, start=args.start, long_only=False)
    lo = run_portfolio(tickers, start=args.start, long_only=True)
    print(f"\n  Window: {ls['start']} -> {ls['end']}  ({ls['years']:.0f} years)\n")
    fmt("System (long+short)", ls)
    fmt("System (LONG-ONLY)", lo)
    print(f"  {'Buy & hold QQQ':<28} final ${10000*(1+ls['qqq_ret']/100):>9,.0f}  ret {ls['qqq_ret']:>7.0f}%  "
          f"CAGR {ls['qqq_cagr']:>5.1f}%  MaxDD {ls['qqq_dd']:>5.0f}%   (benchmark)")

    # hybrid: blend QQQ buy-hold with the LONG-ONLY system (best risk-adj engine)
    import yfinance as yf
    qqq = yf.Ticker("QQQ").history(period="max", auto_adjust=True)
    qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
    sys_ec = lo["equity_curve"]
    qqq = qqq["Close"].reindex(sys_ec.index, method="ffill")
    qnorm = qqq / qqq.iloc[0]
    print(f"\n  HYBRID BLENDS (QQQ buy-hold + long-only system, no rebalance):")
    for w in (0.5, 0.7, 0.8):
        blend = (1 - w) * sys_ec + w * START_EQUITY * qnorm
        yrs = (blend.index[-1] - blend.index[0]).days / 365.25
        cagr = ((blend.iloc[-1] / START_EQUITY) ** (1 / yrs) - 1) * 100
        peak = blend.cummax(); dd = ((blend - peak) / peak).min() * 100
        print(f"    {int(w*100)}% QQQ / {int((1-w)*100)}% system   "
              f"final ${blend.iloc[-1]:>9,.0f}  CAGR {cagr:>5.1f}%  MaxDD {dd:>5.0f}%")
    print()


if __name__ == "__main__":
    main()
