"""
MSFT Soloway Paper Trader
=========================
Run this script once per day AFTER market close (after 4 PM Eastern).
It will:
  1. Fetch today's MSFT data
  2. Apply the Soloway strategy
  3. Execute paper trades (open/close positions)
  4. Print your daily report

Usage:
    python run_trader.py
    python run_trader.py --reset        # Reset account to $1,000 fresh start
    python run_trader.py --backtest 90  # Simulate last 90 days
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import sys
import json
import argparse
from datetime import datetime, date

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np

from portfolio import (
    load_portfolio, save_portfolio, get_equity,
    open_position, close_position, STARTING_BALANCE,
    append_journal_entry, update_journal_outcome,
)
from strategy import (
    fetch_data, compute_indicators, generate_signal,
    size_position, check_exit,
    STOP_LOSS_PCT_LONG, STOP_LOSS_PCT_SHORT,
    TP_LONG_1, TP_SHORT_1,
)
from reporter import generate_daily_report

TICKER = "MSFT"


def is_market_day(check_date: date) -> bool:
    """Return True if the given date is a weekday (Mon-Fri)."""
    return check_date.weekday() < 5


def run_day(portfolio: dict, df: pd.DataFrame, run_date: str, verbose: bool = True, persist: bool = True) -> tuple[dict, float, str, str, dict]:
    """
    Core daily logic. Processes one trading day.
    Returns: (updated_portfolio, day_pnl, action_taken, action_detail, signal_data)
    """
    if len(df) < 5:
        return portfolio, 0.0, "NO DATA", "Insufficient price data", {}

    today_bar = df.iloc[-1]
    current_price = float(today_bar["Close"])
    day_pnl       = 0.0
    action_taken  = "HOLD"
    action_detail = "No trade action today"

    # Compute indicators and generate signal
    signal_data = generate_signal(df)

    # ── STEP 1: CHECK IF EXISTING POSITION NEEDS EXIT ─────────────────────
    pos = portfolio["position"]
    if pos["active"]:
        today_bar_dict = {
            "high": float(today_bar["High"]),
            "low":  float(today_bar["Low"]),
            "close": current_price,
        }
        should_exit, exit_reason = check_exit(pos, today_bar_dict, current_price)

        # Also exit if signal flips against us
        if not should_exit:
            if pos["side"] == "long" and signal_data["signal"] == "SHORT":
                should_exit = True
                exit_reason = "Signal reversed to SHORT"
            elif pos["side"] == "short" and signal_data["signal"] == "LONG":
                should_exit = True
                exit_reason = "Signal reversed to LONG"

        # Max holding period: 8 trading days (swing trading focus)
        if not should_exit and pos["entry_date"]:
            entry_dt  = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
            run_dt    = datetime.strptime(run_date, "%Y-%m-%d")
            held_days = (run_dt - entry_dt).days
            if held_days >= 8:
                should_exit = True
                exit_reason = f"Max holding period (8d) reached"

        if should_exit:
            # Determine exit price (use close, or SL/TP if triggered)
            if "STOP LOSS" in exit_reason:
                exit_price = pos["stop_loss"]
            elif "TAKE PROFIT" in exit_reason:
                exit_price = pos["take_profit"]
            elif "TRAILING" in exit_reason:
                if pos["side"] == "long":
                    exit_price = current_price * (1 - 0.025)
                else:
                    exit_price = current_price * (1 + 0.025)
            else:
                exit_price = current_price

            entry_date_for_journal = pos.get("entry_date")
            portfolio, day_pnl = close_position(portfolio, exit_price, exit_reason, run_date, persist=persist)
            action_taken  = f"CLOSED {pos['side'].upper()}"
            action_detail = f"Exit at ${exit_price:.2f} -- {exit_reason}  |  P&L: ${day_pnl:+.2f}"
            if persist:
                update_journal_outcome(entry_date_for_journal, exit_price, exit_reason, day_pnl)

    # ── STEP 2: LOOK FOR NEW ENTRY (only if flat) ─────────────────────────
    if not portfolio["position"]["active"] and signal_data["signal"] != "HOLD":
        equity = get_equity(portfolio, current_price)

        if signal_data["signal"] == "LONG":
            entry_price = current_price
            stop_loss   = round(entry_price * (1 - STOP_LOSS_PCT_LONG), 2)
            take_profit = round(entry_price * (1 + TP_LONG_1), 2)
            shares      = size_position(equity, entry_price, stop_loss)

            if shares > 0 and portfolio["cash"] >= shares * entry_price:
                portfolio = open_position(
                    portfolio,
                    side        = "long",
                    ticker      = TICKER,
                    shares      = shares,
                    entry_price = entry_price,
                    stop_loss   = stop_loss,
                    take_profit = take_profit,
                    signal_label= signal_data["label"],
                    date        = run_date,
                )
                portfolio["cash"] -= shares * entry_price
                action_taken  = "OPENED LONG"
                action_detail = (
                    f"Bought {shares:.4f} shares @ ${entry_price:.2f} | "
                    f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | "
                    f"Risk: ${shares*(entry_price-stop_loss):.2f}"
                )
                if persist:
                    append_journal_entry({
                        "entry_date": run_date, "side": "long", "ticker": TICKER,
                        "entry_price": round(entry_price, 4), "stop_loss": round(stop_loss, 2),
                        "take_profit": round(take_profit, 2),
                        "risk_reward": round((take_profit - entry_price) / (entry_price - stop_loss), 2),
                        "indicators": {k: signal_data.get(k) for k in ["rsi","ma20","ma50","ma200","atr","vol_ratio","score","trend_bear"]},
                        "label": signal_data.get("label", ""),
                        "exit_price": None, "exit_reason": None, "pnl": None, "outcome": None,
                    })

        elif signal_data["signal"] == "SHORT":
            entry_price = current_price
            stop_loss   = round(entry_price * (1 + STOP_LOSS_PCT_SHORT), 2)
            take_profit = round(entry_price * (1 - TP_SHORT_1), 2)
            # For shorts, size against notional (simplified demo short)
            shares      = size_position(equity, entry_price, stop_loss)
            # For paper short: treat as if we borrow shares, credit cash at entry
            notional    = shares * entry_price
            if shares > 0 and equity >= notional * 0.20:   # 20% margin requirement
                portfolio = open_position(
                    portfolio,
                    side        = "short",
                    ticker      = TICKER,
                    shares      = shares,
                    entry_price = entry_price,
                    stop_loss   = stop_loss,
                    take_profit = take_profit,
                    signal_label= signal_data["label"],
                    date        = run_date,
                )
                # NOTE: No cash credit on short open (prevents compounding exploit)
                action_taken  = "OPENED SHORT"
                action_detail = (
                    f"Shorted {shares:.4f} shares @ ${entry_price:.2f} | "
                    f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | "
                    f"Risk: ${shares*(stop_loss-entry_price):.2f}"
                )
                if persist:
                    append_journal_entry({
                        "entry_date": run_date, "side": "short", "ticker": TICKER,
                        "entry_price": round(entry_price, 4), "stop_loss": round(stop_loss, 2),
                        "take_profit": round(take_profit, 2),
                        "risk_reward": round((entry_price - take_profit) / (stop_loss - entry_price), 2),
                        "indicators": {k: signal_data.get(k) for k in ["rsi","ma20","ma50","ma200","atr","vol_ratio","score","trend_bear"]},
                        "label": signal_data.get("label", ""),
                        "exit_price": None, "exit_reason": None, "pnl": None, "outcome": None,
                    })
            else:
                action_taken  = "SIGNAL: SHORT (no trade)"
                action_detail = "Insufficient equity for short margin requirement"

    # ── STEP 3: UPDATE PEAK EQUITY & DAILY LOG ────────────────────────────
    equity = get_equity(portfolio, current_price)
    if equity > portfolio.get("peak_equity", STARTING_BALANCE):
        portfolio["peak_equity"] = equity

    portfolio.setdefault("daily_log", []).append({
        "date":    run_date,
        "equity":  round(equity, 2),
        "pnl":     round(day_pnl, 2),
        "action":  action_taken,
        "price":   round(current_price, 2),
        "signal":  signal_data.get("signal", "HOLD"),
        "score":   signal_data.get("score", 0),
    })

    portfolio["last_run_date"] = run_date

    return portfolio, day_pnl, action_taken, action_detail, signal_data


def run_backtest(days: int) -> None:
    """Simulate the last N trading days and show cumulative results."""
    print(f"\n{'='*65}")
    print(f"  BACKTEST MODE: Last {days} trading days")
    print(f"{'='*65}\n")

    # Fresh portfolio for backtest
    from portfolio import _new_portfolio
    bt_portfolio = _new_portfolio()

    # Fetch data
    df_full = fetch_data(TICKER, period="2y")
    df_full = compute_indicators(df_full)

    # Get last N trading days
    trading_days = df_full.index[-days:]

    print(f"  {'Date':<12} {'Price':>8} {'Signal':>8} {'Action':<22} {'Day P&L':>10} {'Equity':>10}")
    print(f"  {'-'*73}")

    for i, run_dt in enumerate(trading_days):
        run_date   = run_dt.strftime("%Y-%m-%d")
        # Feed data up to and including this day
        df_slice   = df_full[df_full.index <= run_dt].copy()
        if len(df_slice) < 30: continue

        bt_portfolio, day_pnl, action, detail, sig = run_day(
            bt_portfolio, df_slice, run_date, verbose=False, persist=False
        )
        equity = get_equity(bt_portfolio, float(df_slice.iloc[-1]["Close"]))
        signal = sig.get("signal", "HOLD")
        price  = float(df_slice.iloc[-1]["Close"])
        action_short = action[:20]
        print(f"  {run_date:<12} ${price:>7.2f}  {signal:>7}  {action_short:<22} ${day_pnl:>+8.2f}  ${equity:>8.2f}")

    # Summary
    final_equity  = get_equity(bt_portfolio, float(df_full.iloc[-1]["Close"]))
    total_pnl     = final_equity - STARTING_BALANCE
    pnl_pct       = total_pnl / STARTING_BALANCE * 100
    total_trades  = bt_portfolio["total_trades"]
    win_rate      = bt_portfolio["winning_trades"] / total_trades * 100 if total_trades else 0

    print(f"\n  {'='*40}")
    print(f"  BACKTEST SUMMARY ({days}-day)")
    print(f"  {'='*40}")
    print(f"  Starting Capital : ${STARTING_BALANCE:,.2f}")
    print(f"  Final Equity     : ${final_equity:,.2f}")
    print(f"  Total P&L        : ${total_pnl:>+,.2f} ({pnl_pct:>+.1f}%)")
    print(f"  Total Trades     : {total_trades}")
    print(f"  Win Rate         : {win_rate:.1f}%")
    print(f"  Wins / Losses    : {bt_portfolio['winning_trades']} / {bt_portfolio['losing_trades']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="MSFT Soloway Paper Trader")
    parser.add_argument("--reset",     action="store_true", help="Reset account to fresh $1,000")
    parser.add_argument("--backtest",  type=int, default=0,  help="Run backtest for N days")
    parser.add_argument("--status",    action="store_true", help="Show current account status only")
    args = parser.parse_args()

    # ── RESET ─────────────────────────────────────────────────────────────
    if args.reset:
        confirm = input("Are you sure you want to RESET the account to $1,000? (yes/no): ")
        if confirm.lower() == "yes":
            from portfolio import _new_portfolio, PORTFOLIO_FILE, TRADE_LOG_FILE
            import json
            with open(PORTFOLIO_FILE, "w") as f:
                json.dump(_new_portfolio(), f, indent=2)
            if os.path.exists(TRADE_LOG_FILE):
                os.remove(TRADE_LOG_FILE)
            print("Account reset to $1,000. Run without --reset to start trading.")
        else:
            print("Reset cancelled.")
        return

    # ── BACKTEST MODE ─────────────────────────────────────────────────────
    if args.backtest > 0:
        run_backtest(args.backtest)
        return

    # ── NORMAL DAILY RUN ──────────────────────────────────────────────────
    today_date = datetime.now().strftime("%Y-%m-%d")

    # Load portfolio
    portfolio = load_portfolio()

    # Check if already ran today
    last_run = portfolio.get("last_run_date")
    if last_run == today_date and not args.status:
        print(f"\n  Already ran today ({today_date}). Use --status to view current state.")
        print("  If you need to re-run, delete today's entry from portfolio.json.")
        print()

    # Check market day (skip check for --status flag)
    today_obj = date.today()
    if not is_market_day(today_obj) and not args.status:
        print(f"\n  {today_date} is a weekend/holiday. No trading today.")
        print("  Next trading day: Monday (or use --status to view dashboard).")
        print()
        # Still show full status dashboard on weekends
        df = fetch_data(TICKER, period="1y")
        df = compute_indicators(df)
        signal_data = generate_signal(df)
        current_price = float(df.iloc[-1]["Close"])
        generate_daily_report(
            portfolio, signal_data,
            "WEEKEND - No Trade", f"Market closed ({today_date} is {today_obj.strftime('%A')})",
            0.0, current_price
        )
        return

    # Fetch data
    print(f"\n  Fetching {TICKER} data...")
    df = fetch_data(TICKER, period="1y")
    df = compute_indicators(df)

    if len(df) == 0:
        print("  ERROR: Could not fetch market data.")
        return

    current_price = float(df.iloc[-1]["Close"])
    last_data_date = df.index[-1].strftime("%Y-%m-%d")

    print(f"  {TICKER} last close: ${current_price:.2f} ({last_data_date})")

    if args.status:
        signal_data = generate_signal(df)
        generate_daily_report(
            portfolio, signal_data,
            "STATUS CHECK", "No trade action (status check only)",
            0.0, current_price
        )
        return

    # Run the trading day
    portfolio, day_pnl, action_taken, action_detail, signal_data = run_day(
        portfolio, df, today_date
    )

    # Save updated portfolio
    save_portfolio(portfolio)

    # Generate and print report
    generate_daily_report(
        portfolio,
        signal_data,
        action_taken,
        action_detail,
        day_pnl,
        current_price,
    )


if __name__ == "__main__":
    main()
