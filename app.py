"""
MSFT Soloway Paper Trader — Flask Dashboard Server
Run: python app.py
Then open: http://localhost:5000
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime
from flask import Flask, render_template, jsonify

from portfolio import load_portfolio, load_trade_log, get_equity, STARTING_BALANCE
from strategy import fetch_data, compute_indicators, generate_signal, MASTER_LEVELS

app = Flask(__name__)
TICKER = "MSFT"

# ── Cache signal data so we don't hammer yfinance on every request ──────────
_cache = {"signal": None, "price": None, "fetched_at": None}
CACHE_TTL = 300   # 5 minutes


def get_live_signal():
    """Return cached signal or refresh if stale."""
    now = datetime.now()
    if (
        _cache["fetched_at"] is None
        or (now - _cache["fetched_at"]).seconds > CACHE_TTL
    ):
        try:
            df = fetch_data(TICKER, period="1y")
            df = compute_indicators(df)
            sig = generate_signal(df)
            price = float(df.iloc[-1]["Close"])
            high  = float(df.iloc[-1]["High"])
            low   = float(df.iloc[-1]["Low"])
            vol   = float(df.iloc[-1]["Volume"])
            date  = df.index[-1].strftime("%Y-%m-%d")

            _cache["signal"]     = sig
            _cache["price"]      = price
            _cache["high"]       = high
            _cache["low"]        = low
            _cache["volume"]     = vol
            _cache["date"]       = date
            _cache["fetched_at"] = now
        except Exception as e:
            _cache["signal"]     = {"signal": "ERROR", "label": str(e), "notes": [], "score": 0}
            _cache["price"]      = 0
    return _cache


# ── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/data")
def api_data():
    """Return all data needed by the dashboard in one call."""
    portfolio = load_portfolio()
    trades    = load_trade_log()
    cache     = get_live_signal()
    sig       = cache.get("signal") or {}
    price     = cache.get("price", 0)

    equity    = get_equity(portfolio, price)
    total_pnl = equity - STARTING_BALANCE
    pnl_pct   = (total_pnl / STARTING_BALANCE * 100) if STARTING_BALANCE else 0
    peak      = max(portfolio.get("peak_equity", STARTING_BALANCE), equity)
    drawdown  = ((equity - peak) / peak * 100) if peak else 0

    total_t   = portfolio.get("total_trades", 0)
    wins      = portfolio.get("winning_trades", 0)
    losses    = portfolio.get("losing_trades", 0)
    win_rate  = (wins / total_t * 100) if total_t else 0

    pos = portfolio.get("position", {})
    unrl_pnl  = 0.0
    unrl_pct  = 0.0
    if pos.get("active"):
        entry  = pos["entry_price"]
        shares = pos["shares"]
        if pos["side"] == "long":
            unrl_pnl = (price - entry) * shares
            unrl_pct = (price - entry) / entry * 100 if entry else 0
        else:
            unrl_pnl = (entry - price) * shares
            unrl_pct = (entry - price) / entry * 100 if entry else 0

    # Equity curve from daily_log
    daily_log = portfolio.get("daily_log", [])
    eq_curve  = [{"date": d["date"], "equity": d["equity"]} for d in daily_log[-60:]]

    # Master levels annotated vs current price
    levels = []
    for name, (z_lo, z_hi) in MASTER_LEVELS.items():
        mid  = (z_lo + z_hi) / 2
        dist = (mid / price - 1) * 100 if price else 0
        levels.append({
            "name":  name,
            "low":   z_lo,
            "high":  z_hi,
            "dist":  round(dist, 2),
            "type":  "SUPPORT" if name.startswith("S") else "RESISTANCE",
        })
    levels.sort(key=lambda x: abs(x["dist"]))

    return jsonify({
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_price_date": cache.get("date", "N/A"),
        "account": {
            "starting":  STARTING_BALANCE,
            "cash":      round(portfolio.get("cash", 0), 2),
            "equity":    round(equity, 2),
            "total_pnl": round(total_pnl, 2),
            "pnl_pct":   round(pnl_pct, 2),
            "peak":      round(peak, 2),
            "drawdown":  round(drawdown, 2),
        },
        "stats": {
            "total_trades": total_t,
            "wins":         wins,
            "losses":       losses,
            "win_rate":     round(win_rate, 1),
        },
        "price": {
            "close":  round(price, 2),
            "high":   round(cache.get("high", price), 2),
            "low":    round(cache.get("low",  price), 2),
            "volume": int(cache.get("volume", 0)),
        },
        "signal": {
            "signal":     sig.get("signal", "N/A"),
            "label":      sig.get("label", ""),
            "score":      sig.get("score", 0),
            "strength":   sig.get("strength", 0),
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
        },
        "position": {
            "active":      pos.get("active", False),
            "side":        pos.get("side"),
            "shares":      pos.get("shares", 0),
            "entry_price": pos.get("entry_price", 0),
            "entry_date":  pos.get("entry_date"),
            "stop_loss":   pos.get("stop_loss", 0),
            "take_profit": pos.get("take_profit", 0),
            "signal_label":pos.get("signal_label", ""),
            "unrealized_pnl": round(unrl_pnl, 2),
            "unrealized_pct": round(unrl_pct, 2),
        },
        "trades":    list(reversed(trades[-20:])),
        "eq_curve":  eq_curve,
        "levels":    levels,
    })


@app.route("/api/refresh")
def api_refresh():
    """Force refresh signal cache."""
    _cache["fetched_at"] = None
    return jsonify({"status": "cache cleared"})


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  MSFT Soloway Paper Trader — Web Dashboard")
    print("  Open: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=True)
