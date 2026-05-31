"""
Portfolio State Manager — Multi-Ticker Version
Persists all account state to JSON so data survives between daily runs.
"""

import json
import os
from datetime import datetime
from typing import Optional

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")
TRADE_LOG_FILE = os.path.join(os.path.dirname(__file__), "trade_log.json")
JOURNAL_FILE   = os.path.join(os.path.dirname(__file__), "journal.json")

STARTING_BALANCE = 10_000.00
COMMISSION       = 2.50    # USD per side (open or close)
TICKERS          = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN"]
MAX_POSITIONS    = len(TICKERS)


def load_portfolio() -> dict:
    """Load portfolio from disk. Migrates from old single-position format automatically."""
    if not os.path.exists(PORTFOLIO_FILE):
        return _new_portfolio()
    with open(PORTFOLIO_FILE, "r") as f:
        p = json.load(f)

    # Migrate from old single-position format
    if "position" in p and "positions" not in p:
        old_pos = p.pop("position")
        p["positions"] = {t: _flat_position() for t in TICKERS}
        if old_pos.get("active"):
            ticker = old_pos.get("ticker", "MSFT")
            if ticker in p["positions"]:
                p["positions"][ticker] = old_pos
        p.setdefault("total_commissions", 0.0)

    # Ensure all current tickers are present
    for t in TICKERS:
        p["positions"].setdefault(t, _flat_position())

    return p


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
        "created":           datetime.now().strftime("%Y-%m-%d"),
        "starting_balance":  STARTING_BALANCE,
        "cash":              STARTING_BALANCE,
        "positions":         {t: _flat_position() for t in TICKERS},
        "total_trades":      0,
        "winning_trades":    0,
        "losing_trades":     0,
        "total_pnl":         0.0,
        "total_commissions": 0.0,
        "peak_equity":       STARTING_BALANCE,
        "last_run_date":     None,
        "daily_log":         [],
    }


def get_equity(p: dict, prices) -> float:
    """
    Calculate total equity across all positions.
    prices: dict {ticker: price}  OR  float (backward compat = MSFT price).
    """
    if isinstance(prices, (int, float)):
        prices = {"MSFT": float(prices)}

    total = p["cash"]
    for ticker, pos in p.get("positions", {}).items():
        if not pos or not pos.get("active"):
            continue
        price  = prices.get(ticker, pos.get("entry_price", 0))
        shares = pos["shares"]
        entry  = pos["entry_price"]
        if pos["side"] == "long":
            total += shares * price
        else:
            total += (entry - price) * shares
    return total


def open_position(
    p: dict,
    ticker: str,
    side: str,
    shares: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    signal_label: str,
    date: str,
) -> dict:
    """Open a new position. Deducts entry commission from cash."""
    if side == "long":
        p["cash"] -= shares * entry_price
    p["cash"]              -= COMMISSION
    p["total_commissions"]  = p.get("total_commissions", 0.0) + COMMISSION

    p["positions"][ticker] = {
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
    ticker: str,
    exit_price: float,
    exit_reason: str,
    date: str,
    persist: bool = True,
) -> tuple[dict, float]:
    """Close open position. Returns (portfolio, net_pnl_after_all_commissions)."""
    pos    = p["positions"][ticker]
    shares = pos["shares"]
    entry  = pos["entry_price"]
    side   = pos["side"]

    if side == "long":
        gross_pnl = (exit_price - entry) * shares
        p["cash"] += shares * exit_price
    else:
        gross_pnl = (entry - exit_price) * shares
        p["cash"] += gross_pnl

    p["cash"]              -= COMMISSION
    p["total_commissions"]  = p.get("total_commissions", 0.0) + COMMISSION

    # Net P&L = gross movement minus BOTH commissions (entry was cash-deducted on open)
    net_pnl = gross_pnl - (COMMISSION * 2)

    p["total_pnl"]   += net_pnl
    p["total_trades"] += 1
    if net_pnl >= 0:
        p["winning_trades"] += 1
    else:
        p["losing_trades"] += 1

    if persist:
        append_trade({
            "date":         date,
            "ticker":       ticker,
            "side":         side,
            "shares":       round(shares, 4),
            "entry_price":  round(entry, 4),
            "exit_price":   round(exit_price, 4),
            "exit_reason":  exit_reason,
            "gross_pnl":    round(gross_pnl, 4),
            "commission":   round(COMMISSION * 2, 4),
            "pnl":          round(net_pnl, 4),
            "signal":       pos["signal_label"],
            "holding_days": _holding_days(pos["entry_date"], date),
        })

    p["positions"][ticker] = _flat_position()
    return p, net_pnl


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


# ── LEARNING JOURNAL ──────────────────────────────────────────────────────────

def load_journal() -> list:
    if not os.path.exists(JOURNAL_FILE):
        return []
    with open(JOURNAL_FILE, "r") as f:
        return json.load(f)


def append_journal_entry(entry: dict) -> None:
    journal = load_journal()
    journal.append(entry)
    with open(JOURNAL_FILE, "w") as f:
        json.dump(journal, f, indent=2)


def update_journal_outcome(ticker: str, entry_date: str, exit_price: float, exit_reason: str, pnl: float) -> None:
    journal = load_journal()
    for entry in journal:
        if (entry.get("entry_date") == entry_date
                and entry.get("ticker") == ticker
                and entry.get("outcome") is None):
            entry["exit_price"]  = round(exit_price, 4)
            entry["exit_reason"] = exit_reason
            entry["pnl"]         = round(pnl, 4)
            entry["outcome"]     = "WIN" if pnl >= 0 else "LOSS"
            break
    with open(JOURNAL_FILE, "w") as f:
        json.dump(journal, f, indent=2)
