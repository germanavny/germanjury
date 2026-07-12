"""
MSFT Soloway Paper Trader — Multi-Ticker Daily Runner
=======================================================
Run this script once per day AFTER market close (after 4 PM Eastern).
Scans MSFT, AAPL, NVDA, GOOGL, AMZN and trades each independently.

Usage:
    python run_trader.py
    python run_trader.py --reset
    python run_trader.py --backtest 90
    python run_trader.py --status
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import argparse
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd

from portfolio import (
    load_portfolio, save_portfolio, get_equity,
    open_position, close_position, STARTING_BALANCE, COMMISSION,
    TICKERS, MAX_POSITIONS, CORE_TICKER, CORE_WEIGHT, SLEEVE_WEIGHT,
    append_journal_entry, update_journal_outcome,
)
from strategy import (
    fetch_data, compute_indicators, generate_signal,
    size_position, check_exit, get_market_regime, get_core_regime,
    ATR_STOP_MULT, TRAIL_STOP_PCT, MAX_HOLD_DAYS, COOLDOWN_DAYS,
    STOP_LOSS_PCT_LONG, STOP_LOSS_PCT_SHORT,
    TP_LONG_1, TP_SHORT_1,
)
from reporter import generate_daily_report


def is_market_day(check_date: date) -> bool:
    return check_date.weekday() < 5


def get_next_earnings(ticker: str):
    """Return next earnings date for this ticker, or None if unavailable."""
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is not None and not cal.empty:
            dates = cal.get("Earnings Date", [])
            if len(dates) > 0:
                return pd.Timestamp(dates[0]).date()
    except Exception:
        pass
    return None


def manage_core(portfolio: dict, equity: float, core_regime: dict,
                run_date: str, persist: bool = True) -> tuple[dict, float, str, str]:
    """
    HYBRID CORE: hold QQQ to ~CORE_WEIGHT of equity while the 50/200 regime is ON;
    move fully to cash when it flips OFF. All-in / all-out, no ATR stop, ~1 switch/yr.
    Uses the same open/close primitives as the sleeve so cash accounting stays exact.
    Returns (portfolio, day_pnl, action, detail).
    """
    pos       = portfolio["positions"].get(CORE_TICKER, {})
    qqq_price = float(core_regime.get("qqq_price") or 0.0)
    golden    = bool(core_regime.get("golden"))

    if qqq_price <= 0:
        return portfolio, 0.0, "CORE HOLD", f"{CORE_TICKER}: price unavailable — no core action"

    # regime OFF and we hold the core → sell to cash
    if not golden and pos.get("active"):
        portfolio, pnl = close_position(portfolio, CORE_TICKER, qqq_price,
                                        "Core regime OFF (SMA50<SMA200)", run_date, persist=persist)
        return portfolio, pnl, "CLOSED CORE", (
            f"{CORE_TICKER}: SOLD core @ ${qqq_price:.2f} — regime OFF | P&L: ${pnl:+.2f}")

    # regime ON and we're flat → buy core up to target weight (bounded by available cash)
    if golden and not pos.get("active"):
        target_dollars = min(CORE_WEIGHT * equity, portfolio["cash"] - COMMISSION)
        shares = round(target_dollars / qqq_price, 4)
        if shares > 0 and portfolio["cash"] >= shares * qqq_price + COMMISSION:
            portfolio = open_position(portfolio, CORE_TICKER, "long", shares, qqq_price,
                                      0.0, None, "QQQ core (50/200 regime)", run_date)
            return portfolio, 0.0, "OPENED CORE", (
                f"{CORE_TICKER}: BOUGHT {shares:.4f} core @ ${qqq_price:.2f} "
                f"(~{CORE_WEIGHT*100:.0f}% of equity) — regime ON")
        return portfolio, 0.0, "CORE CASH", f"{CORE_TICKER}: regime ON but insufficient cash"

    # otherwise: hold as-is
    if pos.get("active"):
        return portfolio, 0.0, "CORE HOLD", f"{CORE_TICKER}: holding core (regime ON)"
    return portfolio, 0.0, "CORE CASH", f"{CORE_TICKER}: flat (regime OFF)"


def run_day(
    portfolio: dict,
    df: pd.DataFrame,
    ticker: str,
    run_date: str,
    verbose: bool = True,
    persist: bool = True,
    market: dict | None = None,
) -> tuple[dict, float, str, str, dict]:
    """
    Core daily logic for ONE ticker.
    Returns: (updated_portfolio, day_pnl, action_taken, action_detail, signal_data)
    """
    if len(df) < 5:
        return portfolio, 0.0, "NO DATA", "Insufficient price data", {}

    today_bar     = df.iloc[-1]
    current_price = float(today_bar["Close"])
    atr           = float(today_bar["ATR14"]) if "ATR14" in today_bar and not pd.isna(today_bar["ATR14"]) else current_price * 0.02
    day_pnl       = 0.0
    action_taken  = "HOLD"
    action_detail = f"No trade action for {ticker} today"

    signal_data = generate_signal(df, ticker, market=market)

    # Approximate equity using entry prices for other open positions
    approx_prices = {}
    for t, pos in portfolio["positions"].items():
        approx_prices[t] = pos["entry_price"] if pos.get("active") else 0.0
    approx_prices[ticker] = current_price
    equity = get_equity(portfolio, approx_prices)

    # ── STEP 1: CHECK IF EXISTING POSITION NEEDS EXIT ─────────────────────────
    pos = portfolio["positions"][ticker]
    if pos.get("active"):
        today_bar_dict = {
            "high":  float(today_bar["High"]),
            "low":   float(today_bar["Low"]),
            "close": current_price,
        }
        should_exit, exit_reason = check_exit(pos, today_bar_dict, current_price)

        # Exit if signal flips against position
        if not should_exit:
            if pos["side"] == "long" and signal_data["signal"] == "SHORT":
                should_exit = True
                exit_reason = "Signal reversed to SHORT"
            elif pos["side"] == "short" and signal_data["signal"] == "LONG":
                should_exit = True
                exit_reason = "Signal reversed to LONG"

        # Max holding period (trend swing horizon)
        if not should_exit and pos.get("entry_date"):
            entry_dt  = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
            run_dt    = datetime.strptime(run_date, "%Y-%m-%d")
            held_days = (run_dt - entry_dt).days
            if held_days >= MAX_HOLD_DAYS:
                should_exit = True
                exit_reason = f"Max holding period ({MAX_HOLD_DAYS}d) reached"

        if should_exit:
            if "STOP LOSS" in exit_reason:
                exit_price = pos["stop_loss"]
            elif "TAKE PROFIT" in exit_reason:
                exit_price = pos["take_profit"]
            elif "TRAILING" in exit_reason:
                exit_price = current_price * ((1 - TRAIL_STOP_PCT) if pos["side"] == "long" else (1 + TRAIL_STOP_PCT))
            else:
                exit_price = current_price

            entry_date_for_journal = pos.get("entry_date")
            portfolio, day_pnl = close_position(portfolio, ticker, exit_price, exit_reason, run_date, persist=persist)
            # Record stop-outs so we can enforce the re-entry cooldown (anti-whipsaw)
            if "STOP LOSS" in exit_reason or "TRAILING" in exit_reason:
                portfolio.setdefault("last_stop", {})[ticker] = run_date
            action_taken  = f"CLOSED {pos['side'].upper()} {ticker}"
            action_detail = f"{ticker}: Exit at ${exit_price:.2f} — {exit_reason} | P&L: ${day_pnl:+.2f} (after $5 commission)"
            if persist:
                update_journal_outcome(ticker, entry_date_for_journal, exit_price, exit_reason, day_pnl)

    # ── STEP 2: LOOK FOR NEW ENTRY (only if flat for this ticker) ─────────────
    # Guard: skip new entries within 5 days of earnings (gap risk)
    earnings_date = get_next_earnings(ticker) if not portfolio["positions"][ticker].get("active") else None
    if earnings_date:
        run_date_obj = datetime.strptime(run_date, "%Y-%m-%d").date()
        days_to_earnings = (earnings_date - run_date_obj).days
        if 0 <= days_to_earnings <= 5:
            if action_taken == "HOLD":
                action_detail = f"{ticker}: {signal_data['signal']} signal — BLOCKED, earnings in {days_to_earnings}d ({earnings_date})"
                action_taken = "EARNINGS SKIP"
            return portfolio, day_pnl, action_taken, action_detail, signal_data

    if not portfolio["positions"][ticker].get("active") and signal_data["signal"] != "HOLD":
        # ── COOLDOWN GATE — don't re-enter a name that just stopped us out ─────
        last_stop = portfolio.get("last_stop", {}).get(ticker)
        if last_stop:
            days_since_stop = (datetime.strptime(run_date, "%Y-%m-%d")
                               - datetime.strptime(last_stop, "%Y-%m-%d")).days
            if days_since_stop < COOLDOWN_DAYS:
                signal_data["signal"] = "HOLD"
                action_taken  = "COOLDOWN"
                action_detail = (f"{ticker}: entry blocked — cooldown "
                                 f"{days_since_stop}/{COOLDOWN_DAYS}d since last stop-out")
                return portfolio, day_pnl, action_taken, action_detail, signal_data

        # ── NEWS / CATALYST GATE (advisory) — block entries against hard catalysts ──
        try:
            from news import get_news_sentiment, news_gate
            news = get_news_sentiment(ticker)
            signal_data["news"] = news
            gated_sig, news_reason = news_gate(signal_data["signal"], news)
            if gated_sig == "HOLD" and signal_data["signal"] != "HOLD":
                signal_data["signal"] = "HOLD"
                action_taken = "NEWS SKIP"
                action_detail = f"{ticker}: {news_reason} (flags={news.get('flags')})"
                return portfolio, day_pnl, action_taken, action_detail, signal_data
        except Exception as _e:
            signal_data["news"] = {"sentiment": 0.0, "label": "news error", "headlines": [], "flags": []}

        # Count current open SLEEVE positions (exclude the QQQ core) — cap at MAX_POSITIONS
        open_count = sum(1 for t in TICKERS if portfolio["positions"].get(t, {}).get("active"))
        if open_count >= MAX_POSITIONS:
            action_detail = f"{ticker}: Signal {signal_data['signal']} — portfolio full ({open_count}/{MAX_POSITIONS} positions)"
        else:
            if signal_data["signal"] == "LONG":
                entry_price = current_price
                # ATR-based stop (adapts to volatility); ride the trend → no fixed TP
                stop_loss   = round(entry_price - ATR_STOP_MULT * atr, 2)
                take_profit = None
                shares      = size_position(equity, entry_price, stop_loss, MAX_POSITIONS,
                                            budget_frac=SLEEVE_WEIGHT)

                min_cash_needed = shares * entry_price + COMMISSION
                if shares > 0 and portfolio["cash"] >= min_cash_needed:
                    portfolio = open_position(
                        portfolio, ticker, "long", shares, entry_price,
                        stop_loss, take_profit, signal_data["label"], run_date,
                    )
                    action_taken  = f"OPENED LONG {ticker}"
                    action_detail = (
                        f"{ticker}: Bought {shares:.4f} shares @ ${entry_price:.2f} | "
                        f"SL: ${stop_loss:.2f} (2.5xATR) | ride trend (trail {TRAIL_STOP_PCT*100:.0f}%, max {MAX_HOLD_DAYS}d) | "
                        f"Commission: ${COMMISSION:.2f}"
                    )
                    if persist:
                        append_journal_entry({
                            "entry_date": run_date, "ticker": ticker, "side": "long",
                            "entry_price": round(entry_price, 4),
                            "stop_loss": round(stop_loss, 2),
                            "take_profit": None,
                            "risk_reward": None,
                            "commission": COMMISSION,
                            "indicators": {k: signal_data.get(k) for k in ["rsi","ma20","ma50","ma200","atr","vol_ratio","score","uptrend","downtrend"]},
                            "label": signal_data.get("label", ""),
                            "exit_price": None, "exit_reason": None, "pnl": None, "outcome": None,
                        })
                else:
                    action_detail = f"{ticker}: LONG signal — insufficient cash (${portfolio['cash']:.2f} < ${min_cash_needed:.2f})"

            elif signal_data["signal"] == "SHORT":
                entry_price = current_price
                stop_loss   = round(entry_price + ATR_STOP_MULT * atr, 2)
                take_profit = None
                shares      = size_position(equity, entry_price, stop_loss, MAX_POSITIONS)
                notional    = shares * entry_price

                if shares > 0 and equity >= notional * 0.20 and portfolio["cash"] >= COMMISSION:
                    portfolio = open_position(
                        portfolio, ticker, "short", shares, entry_price,
                        stop_loss, take_profit, signal_data["label"], run_date,
                    )
                    action_taken  = f"OPENED SHORT {ticker}"
                    action_detail = (
                        f"{ticker}: Shorted {shares:.4f} shares @ ${entry_price:.2f} | "
                        f"SL: ${stop_loss:.2f} (2.5xATR) | ride trend (trail {TRAIL_STOP_PCT*100:.0f}%, max {MAX_HOLD_DAYS}d) | "
                        f"Commission: ${COMMISSION:.2f}"
                    )
                    if persist:
                        append_journal_entry({
                            "entry_date": run_date, "ticker": ticker, "side": "short",
                            "entry_price": round(entry_price, 4),
                            "stop_loss": round(stop_loss, 2),
                            "take_profit": None,
                            "risk_reward": None,
                            "commission": COMMISSION,
                            "indicators": {k: signal_data.get(k) for k in ["rsi","ma20","ma50","ma200","atr","vol_ratio","score","uptrend","downtrend"]},
                            "label": signal_data.get("label", ""),
                            "exit_price": None, "exit_reason": None, "pnl": None, "outcome": None,
                        })
                else:
                    action_detail = f"{ticker}: SHORT signal — insufficient funds"

    return portfolio, day_pnl, action_taken, action_detail, signal_data


def run_backtest(days: int, ticker: str = "MSFT") -> None:
    """Simulate the last N trading days for one ticker."""
    from portfolio import _new_portfolio, _flat_position
    print(f"\n{'='*65}")
    print(f"  BACKTEST MODE: {ticker} — Last {days} trading days")
    print(f"{'='*65}\n")

    bt_portfolio = _new_portfolio()
    df_full = fetch_data(ticker, period="2y")
    df_full = compute_indicators(df_full)
    trading_days = df_full.index[-days:]

    print(f"  {'Date':<12} {'Price':>8} {'Signal':>8} {'Action':<24} {'Day P&L':>10} {'Equity':>10}")
    print(f"  {'-'*75}")

    for run_dt in trading_days:
        run_date = run_dt.strftime("%Y-%m-%d")
        df_slice = df_full[df_full.index <= run_dt].copy()
        if len(df_slice) < 30:
            continue

        bt_portfolio, day_pnl, action, detail, sig = run_day(
            bt_portfolio, df_slice, ticker, run_date, verbose=False, persist=False
        )
        price  = float(df_slice.iloc[-1]["Close"])
        equity = get_equity(bt_portfolio, {ticker: price})
        signal = sig.get("signal", "HOLD")
        print(f"  {run_date:<12} ${price:>7.2f}  {signal:>7}  {action[:22]:<24} ${day_pnl:>+8.2f}  ${equity:>8.2f}")

    final_equity = get_equity(bt_portfolio, {ticker: float(df_full.iloc[-1]["Close"])})
    total_pnl    = final_equity - STARTING_BALANCE
    total_trades = bt_portfolio["total_trades"]
    win_rate     = bt_portfolio["winning_trades"] / total_trades * 100 if total_trades else 0
    total_comm   = bt_portfolio.get("total_commissions", 0)

    print(f"\n  {'='*40}")
    print(f"  BACKTEST SUMMARY ({days}-day, {ticker})")
    print(f"  {'='*40}")
    print(f"  Starting Capital  : ${STARTING_BALANCE:,.2f}")
    print(f"  Final Equity      : ${final_equity:,.2f}")
    print(f"  Total Net P&L     : ${total_pnl:>+,.2f}")
    print(f"  Total Commissions : ${total_comm:,.2f}")
    print(f"  Total Trades      : {total_trades}")
    print(f"  Win Rate          : {win_rate:.1f}%")
    print(f"  Wins / Losses     : {bt_portfolio['winning_trades']} / {bt_portfolio['losing_trades']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Soloway Multi-Ticker Paper Trader")
    parser.add_argument("--reset",    action="store_true", help="Reset account to fresh $10,000")
    parser.add_argument("--backtest", type=int, default=0, help="Run backtest for N days (MSFT only)")
    parser.add_argument("--status",   action="store_true", help="Show current account status only")
    parser.add_argument("--ticker",   type=str, default="MSFT", help="Ticker for backtest (default: MSFT)")
    args = parser.parse_args()

    # ── RESET ─────────────────────────────────────────────────────────────────
    if args.reset:
        confirm = input("Reset account to $10,000? (yes/no): ")
        if confirm.lower() == "yes":
            from portfolio import _new_portfolio, PORTFOLIO_FILE, TRADE_LOG_FILE, JOURNAL_FILE
            with open(PORTFOLIO_FILE, "w") as f:
                json.dump(_new_portfolio(), f, indent=2)
            if os.path.exists(TRADE_LOG_FILE):
                os.remove(TRADE_LOG_FILE)
            if os.path.exists(JOURNAL_FILE):
                os.remove(JOURNAL_FILE)
            print("Account reset to $10,000.")
        else:
            print("Reset cancelled.")
        return

    # ── BACKTEST ───────────────────────────────────────────────────────────────
    if args.backtest > 0:
        run_backtest(args.backtest, ticker=args.ticker.upper())
        return

    # ── NORMAL DAILY RUN ───────────────────────────────────────────────────────
    today_date = datetime.now().strftime("%Y-%m-%d")
    portfolio  = load_portfolio()

    last_run = portfolio.get("last_run_date")
    if last_run == today_date and not args.status:
        print(f"\n  Already ran today ({today_date}). Use --status to view.")
        return

    today_obj = date.today()
    if not is_market_day(today_obj) and not args.status:
        print(f"\n  {today_date} is a weekend. No trading today.\n")

    # ── MARKET REGIME (QQQ) — gates direction for every ticker ─────────────────
    market_regime = get_market_regime()
    print(f"\n  MARKET REGIME: {market_regime['label']}")

    print(f"\n  Fetching data for {len(TICKERS)} tickers: {', '.join(TICKERS)}")

    # Fetch all tickers
    dfs    = {}
    prices = {}
    for ticker in TICKERS:
        try:
            df = fetch_data(ticker, period="1y")
            df = compute_indicators(df)
            dfs[ticker]    = df
            prices[ticker] = float(df.iloc[-1]["Close"])
            print(f"  {ticker:<6} ${prices[ticker]:.2f}")
        except Exception as e:
            print(f"  {ticker:<6} ERROR: {e}")

    # ── HYBRID CORE regime (QQQ 50/200) — fetch price so equity includes the core ─
    core_regime = get_core_regime()
    prices[CORE_TICKER] = core_regime["qqq_price"]
    print(f"\n  {core_regime['label']}")

    if args.status:
        results = {}
        for ticker in TICKERS:
            if ticker in dfs:
                sig = generate_signal(dfs[ticker], ticker, market=market_regime)
                pos = portfolio["positions"].get(ticker, {})
                results[ticker] = {"pnl": 0.0, "action": "STATUS CHECK", "detail": "", "signal": sig}
        generate_daily_report(portfolio, results, prices, market=market_regime)
        return

    # ── CORE first: allocate ~70% to the QQQ regime sleeve before the swing book ──
    equity_now = get_equity(portfolio, prices)
    portfolio, core_pnl, core_action, core_detail = manage_core(
        portfolio, equity_now, core_regime, today_date
    )
    if core_action not in ("CORE HOLD", "CORE CASH"):
        print(f"  [CORE] {core_action}: {core_detail}")

    # Run each active sleeve ticker
    results = {}
    for ticker in TICKERS:
        if ticker not in dfs:
            continue
        portfolio, day_pnl, action, detail, sig = run_day(
            portfolio, dfs[ticker], ticker, today_date, market=market_regime
        )
        results[ticker] = {"pnl": day_pnl, "action": action, "detail": detail, "signal": sig}
        if action != "HOLD":
            print(f"  [{ticker}] {action}: {detail}")

    # Update daily log
    equity = get_equity(portfolio, prices)
    if equity > portfolio.get("peak_equity", STARTING_BALANCE):
        portfolio["peak_equity"] = equity

    total_day_pnl = core_pnl + sum(r["pnl"] for r in results.values())
    actions_str   = f"CORE:{core_action.split()[-1]} | " + " | ".join(
        f"{t}:{r['action'].split()[0]}" for t, r in results.items())
    portfolio.setdefault("daily_log", []).append({
        "date":    today_date,
        "equity":  round(equity, 2),
        "pnl":     round(total_day_pnl, 2),
        "action":  actions_str,
        "prices":  {t: round(prices.get(t, 0), 2) for t in TICKERS},
        "signals": {t: r["signal"].get("signal", "HOLD") for t, r in results.items()},
    })
    portfolio["last_run_date"] = today_date

    save_portfolio(portfolio)
    generate_daily_report(portfolio, results, prices, market=market_regime)


if __name__ == "__main__":
    main()
