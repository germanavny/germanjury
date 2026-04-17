"""
MSFT Soloway Paper Trader — Cloud Version (Railway)
- Flask dashboard always live
- APScheduler runs the trader automatically at 20:30 UTC (23:30 Israel)
- Data persists in /data volume on Railway
"""

import sys, os

# ── Data directory: local fallback or Railway volume ──────────────────────
DATA_DIR = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH",
           os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

# Override portfolio/trade log paths to use DATA_DIR
import portfolio as _port
_port.PORTFOLIO_FILE = os.path.join(DATA_DIR, "portfolio.json")
_port.TRADE_LOG_FILE = os.path.join(DATA_DIR, "trade_log.json")
_port.JOURNAL_FILE   = os.path.join(DATA_DIR, "journal.json")

import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime
from flask import Flask, render_template, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from portfolio import load_portfolio, load_trade_log, load_journal, get_equity, STARTING_BALANCE
from strategy import fetch_data, compute_indicators, generate_signal, MASTER_LEVELS
from run_trader import run_day

app = Flask(__name__)
TICKER = "MSFT"

# ── Signal cache ────────────────────────────────────────────────────────────
_cache = {"signal": None, "price": None, "fetched_at": None,
          "high": None, "low": None, "volume": None, "date": None}
CACHE_TTL = 300


def get_live_signal():
    now = datetime.now()
    if (_cache["fetched_at"] is None or
            (now - _cache["fetched_at"]).seconds > CACHE_TTL):
        try:
            df  = fetch_data(TICKER, period="1y")
            df  = compute_indicators(df)
            sig = generate_signal(df)
            _cache.update({
                "signal":     sig,
                "price":      float(df.iloc[-1]["Close"]),
                "high":       float(df.iloc[-1]["High"]),
                "low":        float(df.iloc[-1]["Low"]),
                "volume":     float(df.iloc[-1]["Volume"]),
                "date":       df.index[-1].strftime("%Y-%m-%d"),
                "fetched_at": now,
            })
        except Exception as e:
            _cache["signal"] = {"signal": "ERROR", "label": str(e), "notes": [], "score": 0}
    return _cache


# ── Scheduled trading job ────────────────────────────────────────────────────
def run_scheduled_trade():
    """Runs automatically every weekday at 20:30 UTC (23:30 Israel)."""
    from datetime import date
    today = date.today()
    if today.weekday() >= 5:   # skip weekend
        return

    print(f"\n[SCHEDULER] Running trade job at {datetime.utcnow()} UTC")
    _cache["fetched_at"] = None   # force fresh data

    portfolio = load_portfolio()
    cache     = get_live_signal()
    df        = fetch_data(TICKER, period="1y")
    df        = compute_indicators(df)

    run_date  = datetime.now().strftime("%Y-%m-%d")
    portfolio, day_pnl, action, detail, sig = run_day(
        portfolio, df, run_date, verbose=True, persist=True
    )

    from portfolio import save_portfolio
    save_portfolio(portfolio)

    # Append to reports log
    reports_dir = os.path.join(DATA_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    log_path = os.path.join(reports_dir, "auto_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{run_date}] {action} | {detail} | P&L: ${day_pnl:+.2f}\n")

    print(f"[SCHEDULER] Done. Action: {action} | P&L: ${day_pnl:+.2f}")


# ── Start scheduler ──────────────────────────────────────────────────────────
scheduler = BackgroundScheduler(timezone=pytz.utc)
scheduler.add_job(
    run_scheduled_trade,
    CronTrigger(day_of_week="mon-fri", hour=20, minute=30, timezone=pytz.utc),
    id="daily_trade",
    replace_existing=True,
)
scheduler.start()
print("Scheduler started — trade job runs Mon-Fri at 20:30 UTC (23:30 Israel)")


# ── Routes (identical to app.py) ────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/data")
def api_data():
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

    pos      = portfolio.get("position", {})
    unrl_pnl = 0.0
    unrl_pct = 0.0
    if pos.get("active"):
        entry  = pos["entry_price"]
        shares = pos["shares"]
        if pos["side"] == "long":
            unrl_pnl = (price - entry) * shares
            unrl_pct = (price - entry) / entry * 100 if entry else 0
        else:
            unrl_pnl = (entry - price) * shares
            unrl_pct = (entry - price) / entry * 100 if entry else 0

    daily_log = portfolio.get("daily_log", [])
    eq_curve  = [{"date": d["date"], "equity": d["equity"]} for d in daily_log[-60:]]

    levels = []
    for name, (z_lo, z_hi) in MASTER_LEVELS.items():
        mid  = (z_lo + z_hi) / 2
        dist = (mid / price - 1) * 100 if price else 0
        levels.append({
            "name": name, "low": z_lo, "high": z_hi,
            "dist": round(dist, 2),
            "type": "SUPPORT" if name.startswith("S") else "RESISTANCE",
        })
    levels.sort(key=lambda x: abs(x["dist"]))

    today_log = daily_log[-1] if daily_log else {}
    explanation = _build_explanation(portfolio, pos, price, sig, daily_log)
    monthly_pnl = _calc_monthly_pnl(daily_log)

    return jsonify({
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_price_date": cache.get("date", "N/A"),
        "explanation":     explanation,
        "today_action": {
            "date":   today_log.get("date", ""),
            "action": today_log.get("action", ""),
            "price":  today_log.get("price", 0),
            "pnl":    today_log.get("pnl", 0),
        },
        "monthly_pnl": monthly_pnl,
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
            "active":          pos.get("active", False),
            "side":            pos.get("side"),
            "shares":          pos.get("shares", 0),
            "entry_price":     pos.get("entry_price", 0),
            "entry_date":      pos.get("entry_date"),
            "stop_loss":       pos.get("stop_loss", 0),
            "take_profit":     pos.get("take_profit", 0),
            "signal_label":    pos.get("signal_label", ""),
            "unrealized_pnl":  round(unrl_pnl, 2),
            "unrealized_pct":  round(unrl_pct, 2),
        },
        "trades":   list(reversed(trades[-20:])),
        "eq_curve": eq_curve,
        "levels":   levels,
    })


def _build_explanation(portfolio, pos, price, sig, daily_log):
    """Generate plain-language explanation of current status."""
    if not pos.get("active"):
        last = next((d for d in reversed(daily_log) if d.get("action", "").startswith("CLOSED")), None)
        if last:
            pnl = last.get("pnl", 0)
            result = "ברווח" if pnl >= 0 else "בהפסד"
            return (f"אין פוזיציה פתוחה כרגע. "
                    f"העסקה האחרונה נסגרה {result} של ${pnl:+.2f}. "
                    f"המערכת מחכה להזדמנות הבאה.")
        return "אין פוזיציה פתוחה. המערכת סורקת הזדמנויות כל יום."

    side     = pos["side"]
    entry    = pos["entry_price"]
    sl       = pos["stop_loss"]
    tp       = pos["take_profit"]
    days_in  = 0
    if pos.get("entry_date"):
        from datetime import date
        try:
            days_in = (date.today() - date.fromisoformat(pos["entry_date"])).days
        except Exception:
            pass
    unrl = (entry - price) * pos["shares"] if side == "short" else (price - entry) * pos["shares"]
    direction = "ירידה" if side == "short" else "עלייה"
    action    = "SHORT" if side == "short" else "LONG"

    if abs(unrl) < 0.01:
        pnl_str = "פוזיציה חדשה, ממתים לתנועה"
    else:
        status  = "ברווח" if unrl >= 0 else "בהפסד"
        pnl_str = f"כרגע {status} של ${unrl:+.2f}"

    return (
        f"{action} ב-${entry:.2f} לפני {days_in} ימים — מהמרים על {direction}. "
        f"{pnl_str}. "
        f"יציאה אוטומטית: רווח ב-${tp:.2f} | הגנה ב-${sl:.2f}."
    )


def _calc_monthly_pnl(daily_log):
    """Sum P&L for the current calendar month."""
    month = datetime.now().strftime("%Y-%m")
    return round(sum(d.get("pnl", 0) for d in daily_log if d.get("date", "").startswith(month)), 2)


@app.route("/api/journal")
def api_journal():
    return jsonify(load_journal())


@app.route("/api/refresh")
def api_refresh():
    _cache["fetched_at"] = None
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
        _cache["fetched_at"] = None
        portfolio = load_portfolio()
        df = fetch_data(TICKER, period="1y")
        from strategy import compute_indicators
        df = compute_indicators(df)
        run_date = today.strftime("%Y-%m-%d")
        portfolio, day_pnl, action, detail, sig = run_day(
            portfolio, df, run_date, verbose=False, persist=True
        )
        from portfolio import save_portfolio
        save_portfolio(portfolio)
        return jsonify({"status": "ok", "action": action, "pnl": round(day_pnl, 2), "detail": detail})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)), host="0.0.0.0")
