"""
Portfolio State Manager
Persists all account state to JSON so data survives between daily runs.
"""

import json
import os
from datetime import datetime
from typing import Optional

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")
TRADE_LOG_FILE = os.path.join(os.path.dirname(__file__), "trade_log.json")

STARTING_BALANCE = 1000.00


def load_portfolio() -> dict:
    """Load portfolio from disk. If first run, create fresh $1,000 account."""
    if not os.path.exists(PORTFOLIO_FILE):
        return _new_portfolio()
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)


def save_portfolio(p: dict) -> None:
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)


def load_trade_log() -> list:
    if not os.path.exists(TRADE_LOG_FILE):
        return []
    with open(TRADE_LOG_FILE, "r") as f:
        return json.load(f)


def append_trade(trade: dict) -> None:
    log = load_trade_log()
    log.append(trade)
    with open(TRADE_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def _new_portfolio() -> dict:
    return {
        "created":          datetime.now().strftime("%Y-%m-%d"),
        "starting_balance": STARTING_BALANCE,
        "cash":             STARTING_BALANCE,
        "position": {
            "active":       False,
            "side":         None,        # "long" or "short"
            "ticker":       None,
            "shares":       0.0,
            "entry_price":  0.0,
            "entry_date":   None,
            "stop_loss":    0.0,
            "take_profit":  0.0,
            "signal_label": None,
        },
        "total_trades":     0,
        "winning_trades":   0,
        "losing_trades":    0,
        "total_pnl":        0.0,
        "peak_equity":      STARTING_BALANCE,
        "last_run_date":    None,
        "daily_log":        [],          # list of {date, equity, action, pnl}
    }


def get_equity(p: dict, current_price: float) -> float:
    """Calculate total equity (cash + open position mark-to-market).

    LONG  equity = cash + shares * current_price
    SHORT equity = cash + (entry - current) * shares  [only P&L, no notional]
    """
    if not p["position"]["active"]:
        return p["cash"]
    pos    = p["position"]
    shares = pos["shares"]
    entry  = pos["entry_price"]
    if pos["side"] == "long":
        return p["cash"] + shares * current_price
    else:  # short: only unrealized P&L flows; no notional added
        unrealized = (entry - current_price) * shares
        return p["cash"] + unrealized


def open_position(
    p: dict,
    side: str,
    ticker: str,
    shares: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    signal_label: str,
    date: str,
) -> dict:
    """Open a new position.
    LONG:  debit cash immediately.
    SHORT: cash unchanged (notional is NOT credited — avoids runaway compounding).
           Only realized P&L will be credited/debited on close.
    """
    if side == "long":
        p["cash"] -= shares * entry_price
    # short: no cash change on open
    p["position"] = {
        "active":       True,
        "side":         side,
        "ticker":       ticker,
        "shares":       shares,
        "entry_price":  entry_price,
        "entry_date":   date,
        "stop_loss":    stop_loss,
        "take_profit":  take_profit,
        "signal_label": signal_label,
    }
    return p


def close_position(
    p: dict,
    exit_price: float,
    exit_reason: str,
    date: str,
    persist: bool = True,   # set False in backtest to avoid writing to trade log
) -> tuple[dict, float]:
    """Close open position. Returns updated portfolio and realized P&L."""
    pos = p["position"]
    shares = pos["shares"]
    entry  = pos["entry_price"]
    side   = pos["side"]

    if side == "long":
        pnl = (exit_price - entry) * shares
        p["cash"] += shares * exit_price   # get back sale proceeds
    else:  # short: only P&L hits cash (no notional was credited on open)
        pnl = (entry - exit_price) * shares
        p["cash"] += pnl

    p["total_pnl"]   += pnl
    p["total_trades"] += 1
    if pnl >= 0:
        p["winning_trades"] += 1
    else:
        p["losing_trades"] += 1

    trade_record = {
        "date":         date,
        "ticker":       pos["ticker"],
        "side":         side,
        "shares":       round(shares, 4),
        "entry_price":  round(entry, 4),
        "exit_price":   round(exit_price, 4),
        "exit_reason":  exit_reason,
        "pnl":          round(pnl, 4),
        "signal":       pos["signal_label"],
        "holding_days": _holding_days(pos["entry_date"], date),
    }
    if persist:
        append_trade(trade_record)

    p["position"] = _flat_position()
    return p, pnl


def _flat_position() -> dict:
    return {
        "active":       False,
        "side":         None,
        "ticker":       None,
        "shares":       0.0,
        "entry_price":  0.0,
        "entry_date":   None,
        "stop_loss":    0.0,
        "take_profit":  0.0,
        "signal_label": None,
    }


def _holding_days(entry_date: Optional[str], exit_date: str) -> int:
    if entry_date is None:
        return 0
    fmt = "%Y-%m-%d"
    return (datetime.strptime(exit_date, fmt) - datetime.strptime(entry_date, fmt)).days
