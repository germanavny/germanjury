"""
Soloway Paper Trader — Cloud Version (PythonAnywhere)
- Flask dashboard always live
- APScheduler runs the trader automatically at 20:30 UTC (23:30 Israel)
- Supports MSFT, AAPL, NVDA, GOOGL, AMZN
"""

import sys, os

DATA_DIR = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH",
           os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import portfolio as _port
_port.PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")
_port.TRADE_LOG_FILE = os.path.join(DATA_DIR, "trade_log.json")
_port.JOURNAL_FILE   = os.path.join(DATA_DIR, "journal.json")

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from portfolio import (
    load_portfolio, save_portfolio, load_trade_log, load_journal,
    get_equity, STARTING_BALANCE, TICKERS, CORE_TICKER,
)
from strategy import (
    fetch_data, compute_indicators, generate_signal,
    get_market_regime, get_core_regime, MASTER_LEVELS,
)
from run_trader import run_day, manage_core

app = Flask(__name__)

# ── Per-ticker signal cache ─────────────────────────────────────────────────
_cache = {t: {"signal": None, "price": None, "high": None, "low": None,
               "volume": None, "date": None, "fetched_at": None}
          for t in TICKERS}
CACHE_TTL = 300   # seconds


def get_live_signals():
    """Return cached signals for all tickers, refreshing stale ones."""
    now = datetime.now()
    market = get_market_regime()
    for ticker in TICKERS:
        tc = _cache[ticker]
        age = (now - tc["fetched_at"]).seconds if tc["fetched_at"] else 9999
        if age > CACHE_TTL:
            try:
                df  = fetch_data(ticker, period="1y")
                df  = compute_indicators(df)
                sig = generate_signal(df, ticker, market=market)
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


# ── Scheduled trading job ────────────────────────────────────────────────────
def run_scheduled_trade():
    """Runs every weekday at 20:30 UTC (23:30 Israel)."""
    from datetime import date
    today = date.today()
    if today.weekday() >= 5:
        return

    print(f"\n[SCHEDULER] Running trade job at {datetime.now(timezone.utc)} UTC")

    # Force fresh data
    for tc in _cache.values():
        tc["fetched_at"] = None

    portfolio = load_portfolio()
    run_date  = datetime.now().strftime("%Y-%m-%d")
    results   = {}
    prices    = {}
    market    = get_market_regime()
    print(f"[SCHEDULER] Market regime: {market['label']}")

    # ── HYBRID CORE (QQQ 50/200) — manage first so it gets its ~70% allocation ──
    core_regime = get_core_regime()
    prices[CORE_TICKER] = core_regime["qqq_price"]
    core_pnl = 0.0
    try:
        equity_now = get_equity(portfolio, prices)
        portfolio, core_pnl, core_action, core_detail = manage_core(
            portfolio, equity_now, core_regime, run_date, persist=True
        )
        print(f"[SCHEDULER] CORE: {core_action} | {core_regime['label']}")
    except Exception as e:
        print(f"[SCHEDULER] Error on CORE: {e}")

    for ticker in TICKERS:
        try:
            df = fetch_data(ticker, period="1y")
            df = compute_indicators(df)
            prices[ticker] = float(df.iloc[-1]["Close"])
            portfolio, day_pnl, action, detail, sig = run_day(
                portfolio, df, ticker, run_date, verbose=False, persist=True, market=market
            )
            results[ticker] = {"pnl": day_pnl, "action": action, "detail": detail, "signal": sig}
            if action != "HOLD":
                print(f"[SCHEDULER] {ticker}: {action} | P&L: ${day_pnl:+.2f}")
        except Exception as e:
            print(f"[SCHEDULER] Error on {ticker}: {e}")

    # Update portfolio daily log
    equity = get_equity(portfolio, prices)
    if equity > portfolio.get("peak_equity", STARTING_BALANCE):
        portfolio["peak_equity"] = equity

    total_day_pnl = core_pnl + sum(r.get("pnl", 0) for r in results.values())
    portfolio.setdefault("daily_log", []).append({
        "date":    run_date,
        "equity":  round(equity, 2),
        "pnl":     round(total_day_pnl, 2),
        "action":  " | ".join(f"{t}:{r['action'].split()[0]}" for t, r in results.items()),
        "prices":  {t: round(prices.get(t, 0), 2) for t in TICKERS},
        "signals": {t: r["signal"].get("signal", "HOLD") for t, r in results.items() if r.get("signal")},
    })
    portfolio["last_run_date"] = run_date
    save_portfolio(portfolio)

    # Append to auto log
    reports_dir = os.path.join(DATA_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    log_path = os.path.join(reports_dir, "auto_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{run_date}] equity=${equity:.2f} day_pnl=${total_day_pnl:+.2f}\n")
        for ticker, res in results.items():
            f.write(f"  {ticker}: {res['action']} | P&L: ${res.get('pnl', 0):+.2f}\n")

    print(f"[SCHEDULER] Done. Equity: ${equity:.2f} | Day P&L: ${total_day_pnl:+.2f}")


# ── Scheduler ───────────────────────────────────────────────────────────────
scheduler = BackgroundScheduler(timezone=pytz.utc)
scheduler.add_job(
    run_scheduled_trade,
    CronTrigger(day_of_week="mon-fri", hour=20, minute=30, timezone=pytz.utc),
    id="daily_trade",
    replace_existing=True,
)
scheduler.start()
print("Scheduler started — trade job runs Mon-Fri at 20:30 UTC (23:30 Israel)")


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/data")
def api_data():
    portfolio = load_portfolio()
    trades    = load_trade_log()
    signals   = get_live_signals()

    # Build prices dict
    prices = {t: signals[t].get("price", 0) or 0 for t in TICKERS}
    equity = get_equity(portfolio, prices)

    total_pnl = equity - STARTING_BALANCE
    pnl_pct   = (total_pnl / STARTING_BALANCE * 100) if STARTING_BALANCE else 0
    peak      = max(portfolio.get("peak_equity", STARTING_BALANCE), equity)
    drawdown  = ((equity - peak) / peak * 100) if peak else 0
    total_t   = portfolio.get("total_trades", 0)
    wins      = portfolio.get("winning_trades", 0)
    losses    = portfolio.get("losing_trades", 0)
    win_rate  = (wins / total_t * 100) if total_t else 0
    total_comm = portfolio.get("total_commissions", 0.0)

    # Per-ticker data
    tickers_data = {}
    for ticker in TICKERS:
        tc    = signals[ticker]
        sig   = tc.get("signal") or {}
        price = tc.get("price") or 0
        pos   = portfolio.get("positions", {}).get(ticker, {})

        unrl_pnl = 0.0
        unrl_pct = 0.0
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
                "uptrend":    sig.get("uptrend", False),
                "downtrend":  sig.get("downtrend", False),
                "gated":      sig.get("gated", False),
                "news":       sig.get("news", {}),
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
                "take_profit":    pos.get("take_profit") or 0,
                "signal_label":   pos.get("signal_label", ""),
                "unrealized_pnl": round(unrl_pnl, 2),
                "unrealized_pct": round(unrl_pct, 2),
            },
        }

    # Equity curve
    daily_log = portfolio.get("daily_log", [])
    eq_curve  = [{"date": d["date"], "equity": d["equity"]} for d in daily_log[-60:]]

    # Monthly P&L
    month       = datetime.now().strftime("%Y-%m")
    monthly_pnl = round(sum(d.get("pnl", 0) for d in daily_log if d.get("date", "").startswith(month)), 2)

    # MSFT master levels annotated vs current price
    msft_price = prices.get("MSFT", 1)
    levels = []
    for name, (z_lo, z_hi) in MASTER_LEVELS.items():
        mid  = (z_lo + z_hi) / 2
        dist = (mid / msft_price - 1) * 100 if msft_price else 0
        levels.append({
            "name": name, "low": z_lo, "high": z_hi,
            "dist": round(dist, 2),
            "type": "SUPPORT" if name.startswith("S") else "RESISTANCE",
        })
    levels.sort(key=lambda x: abs(x["dist"]))

    today_log   = daily_log[-1] if daily_log else {}
    explanation = _build_explanation(portfolio, prices, daily_log)

    return jsonify({
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market":       get_market_regime(),
        "tickers":      tickers_data,
        "account": {
            "starting":         STARTING_BALANCE,
            "cash":             round(portfolio.get("cash", 0), 2),
            "equity":           round(equity, 2),
            "total_pnl":        round(total_pnl, 2),
            "pnl_pct":          round(pnl_pct, 2),
            "peak":             round(peak, 2),
            "drawdown":         round(drawdown, 2),
            "total_commissions":round(total_comm, 2),
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
        "trades":    list(reversed(trades[-20:])),
        "eq_curve":  eq_curve,
        "levels":    levels,
    })


def _build_explanation(portfolio, prices, daily_log):
    """Generate plain-language Hebrew summary of portfolio state."""
    positions = portfolio.get("positions", {})
    active    = {t: p for t, p in positions.items() if p.get("active")}

    if not active:
        last = next((d for d in reversed(daily_log) if "CLOSED" in d.get("action", "")), None)
        if last:
            pnl    = last.get("pnl", 0)
            result = "ברווח" if pnl >= 0 else "בהפסד"
            return (f"אין פוזיציות פתוחות. "
                    f"אתמול נסגרה עסקה {result} של ${abs(pnl):.2f}. "
                    f"המערכת סורקת {len(TICKERS)} מניות בחיפוש הזדמנויות.")
        return f"אין פוזיציות פתוחות. המערכת סורקת {len(TICKERS)} מניות כל לילה."

    parts = []
    for ticker, pos in active.items():
        side  = pos["side"]
        entry = pos["entry_price"]
        price = prices.get(ticker, entry)
        if side == "long":
            unrl = (price - entry) * pos["shares"]
        else:
            unrl = (entry - price) * pos["shares"]
        direction = "עלייה" if side == "long" else "ירידה"
        status    = "ברווח" if unrl >= 0 else "בהפסד"
        parts.append(f"{ticker} ({direction}) {status} ${unrl:+.2f}")

    return f"{len(active)} פוזיציות פתוחות: {' | '.join(parts)}."


@app.route("/api/journal")
def api_journal():
    return jsonify(load_journal())


@app.route("/api/refresh")
def api_refresh():
    for tc in _cache.values():
        tc["fetched_at"] = None
    return jsonify({"status": "cache cleared"})


CRON_SECRET = os.environ.get("CRON_SECRET", "soloway2026")

@app.route("/run-trade/<secret>")
def run_trade_endpoint(secret):
    if secret != CRON_SECRET:
        return jsonify({"error": "unauthorized"}), 403

    from datetime import date
    today = date.today()
    if today.weekday() >= 5:
        return jsonify({"status": "skipped", "reason": "weekend"})

    try:
        run_scheduled_trade()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)), host="0.0.0.0")
