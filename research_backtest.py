"""
RESEARCH BACKTEST — Evidence before rewriting.
Compares strategy DIRECTIONS over full available history for all 5 tickers:

  A) CONTRARIAN  (current production signal — fades strength)  + current exits
  B) TREND       (new momentum signal — trades with the trend) + current exits
  C) TREND+      (new momentum signal)                         + trend exits (let winners run)

Same exit engine for A vs B isolates the *entry signal* quality.
C shows the full redesign (entry + exit).

Run:  python research_backtest.py

⚠️ NOTE (2026-05-30): strategy.generate_signal was REWRITTEN to trend-following
after this study. The evidence below was captured BEFORE that rewrite, when
generate_signal was the old contrarian model. If you re-run this now, variant "A"
no longer represents the old contrarian baseline (it runs the new trend signal).
Recorded verdict (pre-rewrite, 5 tickers, full history):
  A Contrarian        CAGR 1.4%  PF 1.07  MaxDD -39%
  C Trend(trend exit) CAGR 4.4%  PF 1.60  MaxDD -23%   ← winner, now in production
  D Trend+Regime      CAGR 3.7%  PF 1.62  MaxDD -19%   ← regime gate added live
"""

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import compute_indicators, generate_signal

TICKERS = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN"]
START_EQUITY = 10_000.0
RISK_PER_TRADE = 0.02
COMMISSION_RT = 5.0   # round-trip $ per trade


# ─────────────────────────────────────────────────────────────────────────────
#  TREND-FOLLOWING SIGNAL  (Soloway-flavored but ALIGNED with the trend)
# ─────────────────────────────────────────────────────────────────────────────
def trend_signal(df: pd.DataFrame, ticker: str = "") -> dict:
    """Momentum / trend-following entry. Buys strength on pullbacks; shorts only
    genuine downtrends. The opposite philosophy of the contrarian model."""
    if len(df) < 200:
        return {"signal": "HOLD", "score": 0.0}

    t = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(t["Close"])
    ma20, ma50, ma200 = float(t["MA20"]), float(t["MA50"]), float(t["MA200"])
    rsi = float(t["RSI"])
    macd, macd_sig = float(t["MACD"]), float(t["MACD_sig"])
    macd_prev, macd_prev_sig = float(prev["MACD"]), float(prev["MACD_sig"])
    score = 0.0

    uptrend   = price > ma50 > ma200
    downtrend = price < ma50 < ma200

    # ── LONG: trade WITH an established uptrend, enter on healthy pullbacks ──
    if uptrend:
        score += 2.0
        # pullback toward MA20/MA50 (buy the dip in an uptrend) — not extended
        dist_ma20 = (price - ma20) / ma20
        if -0.02 <= dist_ma20 <= 0.03:
            score += 1.5                      # near MA20 = prime pullback entry
        if price <= ma50 * 1.02:
            score += 1.0                      # deeper pullback to MA50
        if 40 <= rsi <= 62:
            score += 1.0                      # momentum healthy, not overbought
        elif rsi < 40:
            score += 1.5                      # oversold dip inside uptrend
        if macd > macd_sig:
            score += 1.0
        if macd_prev < macd_prev_sig and macd > macd_sig:
            score += 1.5                      # fresh bullish MACD cross
        if bool(t["RSI_div_bull"]):
            score += 1.0

    # ── SHORT: only genuine downtrend with weakness ──
    if downtrend:
        score -= 2.0
        dist_ma20 = (price - ma20) / ma20
        if -0.03 <= dist_ma20 <= 0.02:
            score -= 1.5                      # bounce to MA20 = short the rally
        if price >= ma50 * 0.98:
            score -= 1.0
        if 38 <= rsi <= 60:
            score -= 1.0
        elif rsi > 60:
            score -= 1.5                      # overbought bounce in downtrend
        if macd < macd_sig:
            score -= 1.0
        if macd_prev > macd_prev_sig and macd < macd_sig:
            score -= 1.5
        if bool(t["RSI_div_bear"]):
            score -= 1.0

    if score >= 4.0:
        sig = "LONG"
    elif score <= -4.0:
        sig = "SHORT"
    else:
        sig = "HOLD"
    return {"signal": sig, "score": round(score, 2)}


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATION ENGINE  (one ticker, one position at a time)
# ─────────────────────────────────────────────────────────────────────────────
def simulate(df, signal_fn, ticker, exit_mode="current", regime=None):
    """
    exit_mode:
      'current' = 3.5%/3% stop, 5.5% fixed TP, 2.5% trail after +3%, 8-day time stop
      'trend'   = ATR-based stop, NO fixed TP, ride trailing stop, 40-day max
    regime: optional pd.Series (bool, market-uptrend per date). If given, longs
            only allowed when market up, shorts only when market down.
    Returns metrics dict.
    """
    equity = START_EQUITY
    pos = None
    trades = []
    equity_curve = []

    idx = df.index
    for i in range(200, len(df)):
        bar = df.iloc[i]
        price = float(bar["Close"])
        high, low = float(bar["High"]), float(bar["Low"])
        atr = float(bar["ATR14"]) if not np.isnan(bar["ATR14"]) else price * 0.02

        # ---- manage open position ----
        if pos is not None:
            exit_price, reason = None, None
            held = (idx[i] - pos["entry_idx"]).days

            if pos["side"] == "long":
                if low <= pos["sl"]:
                    exit_price, reason = pos["sl"], "stop"
                elif pos["tp"] and high >= pos["tp"]:
                    exit_price, reason = pos["tp"], "target"
                else:
                    gain = (price - pos["entry"]) / pos["entry"]
                    if gain > 0.03:
                        trail = price * (1 - pos["trail_pct"])
                        pos["sl"] = max(pos["sl"], trail)
                    if held >= pos["max_days"]:
                        exit_price, reason = price, "time"
            else:  # short
                if high >= pos["sl"]:
                    exit_price, reason = pos["sl"], "stop"
                elif pos["tp"] and low <= pos["tp"]:
                    exit_price, reason = pos["tp"], "target"
                else:
                    gain = (pos["entry"] - price) / pos["entry"]
                    if gain > 0.03:
                        trail = price * (1 + pos["trail_pct"])
                        pos["sl"] = min(pos["sl"], trail)
                    if held >= pos["max_days"]:
                        exit_price, reason = price, "time"

            if exit_price is not None:
                if pos["side"] == "long":
                    gross = (exit_price - pos["entry"]) * pos["shares"]
                else:
                    gross = (pos["entry"] - exit_price) * pos["shares"]
                net = gross - COMMISSION_RT
                equity += net
                trades.append({"side": pos["side"], "pnl": net, "ret": net / START_EQUITY,
                               "reason": reason, "days": held})
                pos = None

        # ---- look for entry when flat ----
        if pos is None:
            sig = signal_fn(df.iloc[: i + 1], ticker)
            s = sig["signal"]
            # market regime gate
            if regime is not None and s in ("LONG", "SHORT"):
                mkt_up = bool(regime.get(idx[i], True))
                if s == "LONG" and not mkt_up:
                    s = "HOLD"
                elif s == "SHORT" and mkt_up:
                    s = "HOLD"
            if s in ("LONG", "SHORT"):
                if exit_mode == "current":
                    if s == "LONG":
                        sl = price * (1 - 0.035); tp = price * (1 + 0.055)
                    else:
                        sl = price * (1 + 0.030); tp = price * (1 - 0.055)
                    trail_pct, max_days = 0.025, 8
                else:  # trend exits: ATR stop, no fixed TP, ride trend
                    if s == "LONG":
                        sl = price - 2.5 * atr
                    else:
                        sl = price + 2.5 * atr
                    tp = None
                    trail_pct, max_days = 0.06, 40

                risk_per_share = abs(price - sl)
                if risk_per_share < 0.01:
                    continue
                shares = (equity * RISK_PER_TRADE) / risk_per_share
                shares = min(shares, (equity * 0.80) / price)  # budget cap
                if shares <= 0:
                    continue
                pos = {"side": "long" if s == "LONG" else "short", "entry": price,
                       "shares": shares, "sl": sl, "tp": tp, "trail_pct": trail_pct,
                       "max_days": max_days, "entry_idx": idx[i]}

        equity_curve.append(equity)

    # ---- metrics ----
    ec = pd.Series(equity_curve)
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    years = (idx[-1] - idx[200]).days / 365.25 if len(idx) > 200 else 1
    total_ret = (equity - START_EQUITY) / START_EQUITY * 100
    cagr = ((equity / START_EQUITY) ** (1 / max(years, 0.5)) - 1) * 100 if equity > 0 else -100
    peak = ec.cummax()
    max_dd = ((ec - peak) / peak).min() * 100 if len(ec) else 0
    return {
        "trades": n,
        "win_rate": (len(wins) / n * 100) if n else 0,
        "total_ret": total_ret,
        "cagr": cagr,
        "final": equity,
        "max_dd": max_dd,
        "pf": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "years": years,
    }


def main():
    print("\n" + "=" * 100)
    print("  EVIDENCE BACKTEST — Contrarian (current) vs Trend-Following — full history")
    print("=" * 100)

    # ── market regime from QQQ: uptrend = QQQ above its 50-day & 200-day MA ──
    qqq = yf.Ticker("QQQ").history(period="max", auto_adjust=True)
    qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
    qqq_ma50  = qqq["Close"].rolling(50).mean()
    qqq_ma200 = qqq["Close"].rolling(200).mean()
    regime = (qqq["Close"] > qqq_ma50) & (qqq_ma50 > qqq_ma200)

    rows = []
    for tk in TICKERS:
        try:
            raw = yf.Ticker(tk).history(period="max", auto_adjust=True)
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            df = compute_indicators(raw)
        except Exception as e:
            print(f"  {tk}: fetch error {e}")
            continue

        A = simulate(df, generate_signal, tk, "current")              # current production
        B = simulate(df, trend_signal,    tk, "current")              # trend entry, same exits
        C = simulate(df, trend_signal,    tk, "trend")                # full trend redesign
        D = simulate(df, trend_signal,    tk, "trend", regime=regime) # trend + market regime gate

        for name, r in (("A Contrarian", A), ("B Trend(cur exit)", B),
                        ("C Trend(trend exit)", C), ("D Trend+Regime", D)):
            rows.append((tk, name, r))

        print(f"\n  ── {tk}  ({A['years']:.0f}y history) ──")
        print(f"  {'Variant':<22}{'Trades':>7}{'Win%':>7}{'TotRet%':>10}{'CAGR%':>8}{'MaxDD%':>8}{'PF':>7}")
        for name, r in (("A Contrarian", A), ("B Trend(cur exit)", B),
                        ("C Trend(trend exit)", C), ("D Trend+Regime", D)):
            pf = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "inf"
            print(f"  {name:<22}{r['trades']:>7}{r['win_rate']:>6.1f}%{r['total_ret']:>9.0f}%"
                  f"{r['cagr']:>7.1f}%{r['max_dd']:>7.0f}%{pf:>7}")

    # ---- aggregate ----
    print("\n" + "=" * 100)
    print("  AGGREGATE (avg across 5 tickers)")
    print("=" * 100)
    for variant in ["A Contrarian", "B Trend(cur exit)", "C Trend(trend exit)", "D Trend+Regime"]:
        rs = [r for (_, n, r) in rows if n == variant]
        if not rs:
            continue
        print(f"  {variant:<22} avg CAGR {np.mean([r['cagr'] for r in rs]):>6.1f}%   "
              f"avg Win {np.mean([r['win_rate'] for r in rs]):>5.1f}%   "
              f"avg MaxDD {np.mean([r['max_dd'] for r in rs]):>6.0f}%   "
              f"avg Trades {np.mean([r['trades'] for r in rs]):>5.0f}   "
              f"avg PF {np.mean([r['pf'] for r in rs if r['pf']!=float('inf')]):>4.2f}")
    print()


if __name__ == "__main__":
    main()
