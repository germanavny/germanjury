"""
Soloway Paper Trader — Local Dev Dashboard
Run: python app.py  →  open http://localhost:5000
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from flask import Flask, render_template, jsonify

from portfolio import (
    load_portfolio, load_trade_log, get_equity,
    STARTING_BALANCE, TICKERS,
)
from strategy import fetch_data, compute_indicators, generate_signal, MASTER_LEVELS

app = Flask(__name__)

# ── Per-ticker cache ──────────────────────────────────────────────────────────
_cache = {t: {"signal": None, "price": None, "high": None, "low": None,
               "volume": None, "date": None, "fetched_at": None}
          for t in TICKERS}
CACHE_TTL = 300


def get_live_signals():
    now = datetime.now()
    for ticker in TICKERS:
        tc  = _cache[ticker]
        age = (now - tc["fetched_at"]).seconds if tc["fetched_at"] else 9999
        if age > CACHE_TTL:
            try:
                df  = fetch_data(ticker, period="1y")
                df  = compute_indicators(df)
                sig = generate_signal(df, ticker)
                tc.update({
                    "signal":     sig,
                    "price":      float(df.iloc[-1]["Close"]),
                    "high":       float(df.iloc[-1]["High"]),
                    "low":        float(df.iloc[-1]["Low"]),
                    "volume":     float(df.iloc[-1]["Volume"]),
                    "date":       df.index[-1].strftime("%Y-%m-%d"),
                    "fetched_at": now,
                })
            except Exception as e:
                tc["signal"] = {"signal": "ERROR", "label": str(e), "notes": [], "score": 0}
                tc["price"]  = 0
    return _cache


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/data")
def api_data():
    portfolio = load_portfolio()
    trades    = load_trade_log()
    signals   = get_live_signals()

    prices    = {t: signals[t].get("price") or 0 for t in TICKERS}
    equity    = get_equity(portfolio, prices)
    total_pnl = equity - STARTING_BALANCE
    pnl_pct   = (total_pnl / STARTING_BALANCE * 100) if STARTING_BALANCE else 0
    peak      = max(portfolio.get("peak_equity", STARTING_BALANCE), equity)
    drawdown  = ((equity - peak) / peak * 100) if peak else 0
    total_t   = portfolio.get("total_trades", 0)
    wins      = portfolio.get("winning_trades", 0)
    losses    = portfolio.get("losing_trades", 0)
    win_rate  = (wins / total_t * 100) if total_t else 0
    total_comm = portfolio.get("total_commissions", 0.0)

    tickers_data = {}
    for ticker in TICKERS:
        tc    = signals[ticker]
        sig   = tc.get("signal") or {}
        price = tc.get("price") or 0
        pos   = portfolio.get("positions", {}).get(ticker, {})

        unrl_pnl = unrl_pct = 0.0
        if pos.get("active") and price:
            entry  = pos["entry_price"]
            shares = pos["shares"]
            if pos["side"] == "long":
                unrl_pnl = (price - entry) * shares
                unrl_pct = (price - entry) / entry * 100 if entry else 0
            else:
                unrl_pnl = (entry - price) * shares
                unrl_pct = (entry - price) / entry * 100 if entry else 0

        tickers_data[ticker] = {
            "price": {
                "close":  round(price, 2),
                "high":   round(tc.get("high") or price, 2),
                "low":    round(tc.get("low")  or price, 2),
                "volume": int(tc.get("volume") or 0),
                "date":   tc.get("date", ""),
            },
            "signal": {
                "signal":     sig.get("signal", "N/A"),
                "label":      sig.get("label", ""),
                "score":      sig.get("score", 0),
                "rsi":        sig.get("rsi", 0),
                "vol_ratio":  sig.get("vol_ratio", 1),
                "ma20":       sig.get("ma20", 0),
                "ma50":       sig.get("ma50", 0),
                "ma200":      sig.get("ma200", 0),
                "trend_bear": sig.get("trend_bear", False),
                "fib618":     sig.get("fib618", 0),
                "fib50":      sig.get("fib50", 0),
                "fib382":     sig.get("fib382", 0),
                "notes":      sig.get("notes", []),
                "atr":        sig.get("atr", 0),
                "strength":   sig.get("strength", 0),
            },
            "position": {
                "active":         pos.get("active", False),
                "side":           pos.get("side"),
                "shares":         pos.get("shares", 0),
                "entry_price":    pos.get("entry_price", 0),
                "entry_date":     pos.get("entry_date"),
                "stop_loss":      pos.get("stop_loss", 0),
                "take_profit":    pos.get("take_profit", 0),
                "signal_label":   pos.get("signal_label", ""),
                "unrealized_pnl": round(unrl_pnl, 2),
                "unrealized_pct": round(unrl_pct, 2),
            },
        }

    daily_log   = portfolio.get("daily_log", [])
    eq_curve    = [{"date": d["date"], "equity": d["equity"]} for d in daily_log[-60:]]
    month       = datetime.now().strftime("%Y-%m")
    monthly_pnl = round(sum(d.get("pnl", 0) for d in daily_log if d.get("date", "").startswith(month)), 2)

    msft_price = prices.get("MSFT", 1)
    levels = []
    for name, (z_lo, z_hi) in MASTER_LEVELS.items():
        mid  = (z_lo + z_hi) / 2
        dist = (mid / msft_price - 1) * 100 if msft_price else 0
        levels.append({"name": name, "low": z_lo, "high": z_hi,
                        "dist": round(dist, 2),
                        "type": "SUPPORT" if name.startswith("S") else "RESISTANCE"})
    levels.sort(key=lambda x: abs(x["dist"]))

    active_positions = {t: p for t, p in portfolio.get("positions", {}).items() if p.get("active")}
    if active_positions:
        parts = []
        for ticker, pos in active_positions.items():
            price  = prices.get(ticker, pos["entry_price"])
            unrl   = ((price - pos["entry_price"]) if pos["side"] == "long" else (pos["entry_price"] - price)) * pos["shares"]
            status = "ברווח" if unrl >= 0 else "בהפסד"
            parts.append(f"{ticker} {status} ${unrl:+.2f}")
        explanation = f"{len(active_positions)} פוזיציות פתוחות: {' | '.join(parts)}."
    else:
        explanation = f"אין פוזיציות פתוחות. המערכת סורקת {len(TICKERS)} מניות כל לילה."

    today_log = daily_log[-1] if daily_log else {}

    return jsonify({
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tickers":      tickers_data,
        "account": {
            "starting":          STARTING_BALANCE,
            "cash":              round(portfolio.get("cash", 0), 2),
            "equity":            round(equity, 2),
            "total_pnl":         round(total_pnl, 2),
            "pnl_pct":           round(pnl_pct, 2),
            "peak":              round(peak, 2),
            "drawdown":          round(drawdown, 2),
            "total_commissions": round(total_comm, 2),
        },
        "stats": {
            "total_trades": total_t,
            "wins":         wins,
            "losses":       losses,
            "win_rate":     round(win_rate, 1),
        },
        "monthly_pnl":  monthly_pnl,
        "explanation":  explanation,
        "today_action": {
            "date":   today_log.get("date", ""),
            "action": today_log.get("action", ""),
            "pnl":    today_log.get("pnl", 0),
        },
        "trades":   list(reversed(trades[-20:])),
        "eq_curve": eq_curve,
        "levels":   levels,
    })


@app.route("/api/refresh")
def api_refresh():
    for tc in _cache.values():
        tc["fetched_at"] = None
    return jsonify({"status": "cache cleared"})


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Soloway Multi-Ticker Paper Trader — Dev Dashboard")
    print("  Tickers:", ", ".join(TICKERS))
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=True)
