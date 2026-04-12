"""
Daily Report Generator
Produces the end-of-day P&L report that the user sees each trading day.
"""

import os
from datetime import datetime
from portfolio import get_equity, load_trade_log, STARTING_BALANCE

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def _bar(value: float, max_val: float = 10.0, width: int = 30, positive_char="=", negative_char="-") -> str:
    if max_val == 0:
        return ""
    ratio = min(abs(value) / max_val, 1.0)
    filled = int(ratio * width)
    char   = positive_char if value >= 0 else negative_char
    return char * filled + " " * (width - filled)


def _pnl_color(pnl: float) -> str:
    """Return +/- emoji based on P&L."""
    if pnl > 0:   return "  [PROFIT]"
    elif pnl < 0: return "  [LOSS]"
    else:          return "  [FLAT]"


def generate_daily_report(
    portfolio: dict,
    signal_data: dict,
    action_taken: str,
    action_detail: str,
    day_pnl: float,
    current_price: float,
    ticker: str = "MSFT",
) -> str:
    """Build and print the full daily report. Also saves to file."""

    today_str  = datetime.now().strftime("%Y-%m-%d %H:%M")
    today_date = datetime.now().strftime("%Y-%m-%d")

    equity     = get_equity(portfolio, current_price)
    total_pnl  = equity - STARTING_BALANCE
    pnl_pct    = (total_pnl / STARTING_BALANCE) * 100
    peak_eq    = max(portfolio.get("peak_equity", STARTING_BALANCE), equity)
    drawdown   = ((equity - peak_eq) / peak_eq) * 100 if peak_eq > 0 else 0

    total_trades  = portfolio.get("total_trades", 0)
    winning       = portfolio.get("winning_trades", 0)
    losing        = portfolio.get("losing_trades", 0)
    win_rate      = (winning / total_trades * 100) if total_trades > 0 else 0

    pos           = portfolio.get("position", {})
    has_position  = pos.get("active", False)

    lines = []
    SEP  = "=" * 65
    sep2 = "-" * 65

    # ── HEADER ────────────────────────────────────────────────────────────
    lines += [
        "",
        SEP,
        f"  MSFT PAPER TRADER  //  Daily Report",
        f"  {today_str}",
        SEP,
        "",
    ]

    # ── ACCOUNT SUMMARY ───────────────────────────────────────────────────
    lines += [
        "  ACCOUNT SNAPSHOT",
        sep2,
        f"  Starting Capital : ${STARTING_BALANCE:>10,.2f}",
        f"  Current Equity   : ${equity:>10,.2f}  {_pnl_color(total_pnl)}",
        f"  Total P&L        : ${total_pnl:>+10,.2f}  ({pnl_pct:>+.2f}%)",
        f"  Cash Available   : ${portfolio['cash']:>10,.2f}",
        f"  Peak Equity      : ${peak_eq:>10,.2f}",
        f"  Drawdown         : {drawdown:>+.2f}%",
        "",
    ]

    # ── TODAY'S ACTIVITY ─────────────────────────────────────────────────
    lines += [
        "  TODAY'S ACTIVITY",
        sep2,
        f"  {ticker} Close Price : ${current_price:.2f}",
        f"  Action Taken     : {action_taken}",
        f"  Detail           : {action_detail}",
        f"  Day P&L          : ${day_pnl:>+.2f}{_pnl_color(day_pnl)}",
        "",
    ]

    # ── OPEN POSITION ─────────────────────────────────────────────────────
    if has_position:
        side     = pos["side"].upper()
        shares   = pos["shares"]
        entry    = pos["entry_price"]
        sl       = pos["stop_loss"]
        tp       = pos["take_profit"]
        entry_dt = pos["entry_date"]
        signal   = pos["signal_label"]

        if pos["side"] == "long":
            unrl_pnl = (current_price - entry) * shares
            unrl_pct = (current_price - entry) / entry * 100
        else:
            unrl_pnl = (entry - current_price) * shares
            unrl_pct = (entry - current_price) / entry * 100

        lines += [
            f"  OPEN POSITION  [{side}]",
            sep2,
            f"  Signal         : {signal}",
            f"  Entry Date     : {entry_dt}",
            f"  Entry Price    : ${entry:.2f}",
            f"  Shares         : {shares:.4f}",
            f"  Stop Loss      : ${sl:.2f}  ({(sl/entry-1)*100:>+.1f}%)",
            f"  Take Profit    : ${tp:.2f}  ({(tp/entry-1)*100:>+.1f}%)",
            f"  Unrealized P&L : ${unrl_pnl:>+.2f}  ({unrl_pct:>+.2f}%)",
            "",
        ]
    else:
        lines += [
            "  OPEN POSITION  [NONE - Cash]",
            sep2,
            "  No active trade. Waiting for setup.",
            "",
        ]

    # ── MARKET SIGNALS ────────────────────────────────────────────────────
    sig = signal_data
    lines += [
        "  SOLOWAY SIGNAL DASHBOARD",
        sep2,
        f"  Signal     : {sig.get('signal', 'N/A')}  (Score: {sig.get('score', 0):+.1f})",
        f"  Label      : {sig.get('label', '')}",
        f"  RSI(14)    : {sig.get('rsi', 0):.1f}{'  OVERSOLD' if sig.get('rsi',50) < 35 else '  OVERBOUGHT' if sig.get('rsi',50) > 65 else ''}",
        f"  Vol Ratio  : {sig.get('vol_ratio', 1):.2f}x  {'** HIGH VOL **' if sig.get('vol_ratio',1) > 2 else ''}",
        f"  MA20       : ${sig.get('ma20', 0):.2f}  ({'ABOVE' if current_price > sig.get('ma20',0) else 'BELOW'})",
        f"  MA50       : ${sig.get('ma50', 0):.2f}  ({'ABOVE' if current_price > sig.get('ma50',0) else 'BELOW'})",
        f"  MA200      : ${sig.get('ma200', 0):.2f}  ({'ABOVE' if current_price > sig.get('ma200',0) else 'BELOW'})",
        f"  Regime     : {'DEATH CROSS (Bearish)' if sig.get('trend_bear') else 'GOLDEN CROSS (Bullish)'}",
        f"  Fib 61.8%  : ${sig.get('fib618', 0):.2f}",
        f"  Fib 50.0%  : ${sig.get('fib50', 0):.2f}",
        f"  Fib 38.2%  : ${sig.get('fib382', 0):.2f}",
        "",
        "  Signal Notes:",
    ]
    for note in sig.get("notes", []):
        lines.append(f"    * {note}")

    lines.append("")

    # ── TRADE STATISTICS ─────────────────────────────────────────────────
    lines += [
        "  TRADE STATISTICS",
        sep2,
        f"  Total Trades : {total_trades}",
        f"  Wins         : {winning}",
        f"  Losses       : {losing}",
        f"  Win Rate     : {win_rate:.1f}%",
        "",
    ]

    # ── RECENT TRADE HISTORY ─────────────────────────────────────────────
    trade_log = load_trade_log()
    if trade_log:
        lines += ["  RECENT CLOSED TRADES (last 10)", sep2]
        header = f"  {'Date':<12} {'Side':<6} {'Shares':>8} {'Entry':>8} {'Exit':>8} {'P&L':>9} {'Reason'}"
        lines.append(header)
        lines.append(f"  {'-'*63}")
        for t in trade_log[-10:]:
            pnl_str  = f"${t['pnl']:>+.2f}"
            flag     = "[W]" if t["pnl"] >= 0 else "[L]"
            line = (
                f"  {t['date']:<12} {t['side']:<6} {t['shares']:>8.3f} "
                f"${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
                f"{pnl_str:>9}  {flag} {t['exit_reason'][:25]}"
            )
            lines.append(line)
        lines.append("")

    # ── EQUITY CURVE (ASCII) ─────────────────────────────────────────────
    daily_log = portfolio.get("daily_log", [])
    if len(daily_log) >= 3:
        lines += ["  EQUITY CURVE (last 20 sessions)", sep2]
        recent_log = daily_log[-20:]
        equities   = [d["equity"] for d in recent_log]
        dates      = [d["date"][-5:] for d in recent_log]   # MM-DD
        min_eq     = min(equities)
        max_eq     = max(equities)
        eq_range   = max_eq - min_eq if max_eq > min_eq else 1

        height = 6
        chart  = []
        for row in range(height, 0, -1):
            threshold = min_eq + (row / height) * eq_range
            row_line  = "  "
            for eq in equities:
                row_line += "#" if eq >= threshold else " "
            label_val = min_eq + (row / height) * eq_range
            row_line  += f"  ${label_val:.0f}"
            chart.append(row_line)

        chart.append("  " + "".join(d[-2:] for d in dates))
        lines += chart
        lines.append("")

    # ── MASTER LEVELS CHEAT SHEET ────────────────────────────────────────
    lines += [
        "  MASTER LEVELS CHEAT SHEET (vs current ${:.2f})".format(current_price),
        sep2,
        "  SUPPORT:",
        "    $339-$345  [MASTER] Double Fib Confluence -- PRIMARY TARGET",
        "    $365-$376  Current area (50% Fib from ATH)",
        "    $309-$320  78.6% Fib from ATH (capitulation)",
        "  RESISTANCE:",
        "    $390-$405  MA50 + 38.2% Fib from ATH",
        "    $428-$442  2024 resistance cluster",
        "    $458-$480  Pre-ATH distribution zone",
        "",
    ]

    # ── FOOTER ────────────────────────────────────────────────────────────
    lines += [
        SEP,
        "  Soloway Rule: 'Patience. Let price come to your level.'",
        "  This is a PAPER TRADING simulation. Not financial advice.",
        SEP,
        "",
    ]

    report_text = "\n".join(lines)

    # Print to console
    print(report_text)

    # Save to file
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"report_{today_date}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  [Report saved to: {report_path}]")

    return report_text
