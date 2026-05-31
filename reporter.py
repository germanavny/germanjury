"""
Daily Report Generator — Multi-Ticker Version
"""

import os
from datetime import datetime
from portfolio import get_equity, load_trade_log, STARTING_BALANCE, TICKERS

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def _pnl_tag(pnl: float) -> str:
    if pnl > 0:   return "  [PROFIT]"
    elif pnl < 0: return "  [LOSS]"
    else:          return "  [FLAT]"


def generate_daily_report(
    portfolio: dict,
    results: dict,
    prices: dict,
    market: dict | None = None,
) -> str:
    """
    Build and print the full daily report. Also saves to file.
    results = {ticker: {"pnl": ..., "action": ..., "detail": ..., "signal": sig_dict}}
    prices  = {ticker: current_price}
    market  = optional regime dict from strategy.get_market_regime()
    """
    today_str  = datetime.now().strftime("%Y-%m-%d %H:%M")
    today_date = datetime.now().strftime("%Y-%m-%d")

    equity     = get_equity(portfolio, prices)
    total_pnl  = equity - STARTING_BALANCE
    pnl_pct    = (total_pnl / STARTING_BALANCE) * 100
    peak_eq    = max(portfolio.get("peak_equity", STARTING_BALANCE), equity)
    drawdown   = ((equity - peak_eq) / peak_eq) * 100 if peak_eq > 0 else 0
    total_comm = portfolio.get("total_commissions", 0.0)

    total_trades = portfolio.get("total_trades", 0)
    winning      = portfolio.get("winning_trades", 0)
    losing       = portfolio.get("losing_trades", 0)
    win_rate     = (winning / total_trades * 100) if total_trades > 0 else 0

    lines = []
    SEP   = "=" * 70
    sep2  = "-" * 70

    # ── HEADER ────────────────────────────────────────────────────────────────
    lines += [
        "",
        SEP,
        f"  SOLOWAY MULTI-TICKER PAPER TRADER  //  Daily Report",
        f"  {today_str}",
        SEP,
        "",
    ]

    # ── ACCOUNT SUMMARY ───────────────────────────────────────────────────────
    lines += [
        "  ACCOUNT SNAPSHOT",
        sep2,
        f"  Starting Capital  : ${STARTING_BALANCE:>10,.2f}",
        f"  Current Equity    : ${equity:>10,.2f}  {_pnl_tag(total_pnl)}",
        f"  Total Net P&L     : ${total_pnl:>+10,.2f}  ({pnl_pct:>+.2f}%)",
        f"  Cash Available    : ${portfolio['cash']:>10,.2f}",
        f"  Peak Equity       : ${peak_eq:>10,.2f}",
        f"  Drawdown          : {drawdown:>+.2f}%",
        f"  Total Commissions : ${total_comm:>10,.2f}  (${total_comm/max(total_trades,1):.2f}/trade avg)",
        "",
    ]

    # ── MARKET REGIME ─────────────────────────────────────────────────────────
    if market:
        lines += [
            "  MARKET REGIME (QQQ-gated)",
            sep2,
            f"  {market.get('label', 'n/a')}",
            f"  QQQ: ${market.get('qqq_price', 0):.2f}",
            "",
        ]

    # ── PER-TICKER SUMMARY ────────────────────────────────────────────────────
    lines += ["  TODAY'S TICKER ACTIVITY", sep2]
    for ticker in TICKERS:
        res   = results.get(ticker, {})
        price = prices.get(ticker, 0)
        sig   = res.get("signal", {})
        pos   = portfolio.get("positions", {}).get(ticker, {})

        signal_str = sig.get("signal", "—") if sig else "—"
        score_str  = f"score {sig.get('score', 0):+.1f}" if sig else ""
        action     = res.get("action", "HOLD")

        # trend + news context
        trend_str = ""
        if sig:
            if sig.get("uptrend"):     trend_str = " UP↑"
            elif sig.get("downtrend"): trend_str = " DN↓"
        news = sig.get("news") if sig else None
        news_str = ""
        if news and news.get("flags"):
            news_str = f"  news:{news.get('sentiment',0):+.2f}{'/' + ','.join(news['flags'][:2]) if news.get('flags') else ''}"

        pos_str = ""
        if pos.get("active"):
            side   = pos["side"].upper()
            entry  = pos["entry_price"]
            shares = pos["shares"]
            if pos["side"] == "long":
                unrl = (price - entry) * shares
            else:
                unrl = (entry - price) * shares
            pos_str = f"  [{side} @ ${entry:.2f}, unrl: ${unrl:+.2f}]"

        lines.append(f"  {ticker:<6} ${price:>8.2f}  {signal_str:>6}{trend_str:<4} ({score_str:<12})  {action}{pos_str}{news_str}")

    lines.append("")

    # ── OPEN POSITIONS DETAIL ─────────────────────────────────────────────────
    active_positions = {t: p for t, p in portfolio.get("positions", {}).items() if p.get("active")}
    if active_positions:
        lines += ["  OPEN POSITIONS DETAIL", sep2]
        for ticker, pos in active_positions.items():
            price  = prices.get(ticker, pos["entry_price"])
            side   = pos["side"]
            entry  = pos["entry_price"]
            shares = pos["shares"]
            sl     = pos["stop_loss"]
            tp     = pos["take_profit"]
            if side == "long":
                unrl = (price - entry) * shares
                unrl_pct = (price - entry) / entry * 100
            else:
                unrl = (entry - price) * shares
                unrl_pct = (entry - price) / entry * 100

            tp_str = f"${tp:.2f}" if tp else "ride trend (trail)"
            lines += [
                f"  {ticker} [{side.upper()}]  Entry: ${entry:.2f}  Shares: {shares:.4f}",
                f"    Current: ${price:.2f}  Unrealized: ${unrl:+.2f} ({unrl_pct:+.2f}%)",
                f"    Stop Loss: ${sl:.2f}  Take Profit: {tp_str}  Entry Date: {pos['entry_date']}",
                "",
            ]
    else:
        lines += ["  OPEN POSITIONS: NONE — Cash only", sep2, ""]

    # ── TRADE STATISTICS ──────────────────────────────────────────────────────
    lines += [
        "  TRADE STATISTICS",
        sep2,
        f"  Total Trades  : {total_trades}",
        f"  Wins          : {winning}",
        f"  Losses        : {losing}",
        f"  Win Rate      : {win_rate:.1f}%",
        f"  Commissions   : ${total_comm:.2f} paid",
        "",
    ]

    # ── RECENT CLOSED TRADES ──────────────────────────────────────────────────
    trade_log = load_trade_log()
    if trade_log:
        lines += ["  RECENT CLOSED TRADES (last 10)", sep2]
        header = f"  {'Date':<12} {'Ticker':<6} {'Side':<6} {'Shares':>6} {'Entry':>8} {'Exit':>8} {'Gross':>8} {'Comm':>6} {'Net P&L':>8} {'Reason'}"
        lines.append(header)
        lines.append(f"  {'-'*80}")
        for t in trade_log[-10:]:
            flag     = "[W]" if t["pnl"] >= 0 else "[L]"
            gross    = t.get("gross_pnl", t["pnl"])
            comm_str = f"${t.get('commission', 5.0):.2f}"
            line = (
                f"  {t['date']:<12} {t.get('ticker','MSFT'):<6} {t['side']:<6} {t['shares']:>6.3f} "
                f"${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
                f"${gross:>+7.2f} {comm_str:>6} ${t['pnl']:>+7.2f}  {flag} {t['exit_reason'][:20]}"
            )
            lines.append(line)
        lines.append("")

    # ── EQUITY CURVE (ASCII) ──────────────────────────────────────────────────
    daily_log = portfolio.get("daily_log", [])
    if len(daily_log) >= 3:
        lines += ["  EQUITY CURVE (last 20 sessions)", sep2]
        recent_log = daily_log[-20:]
        equities   = [d["equity"] for d in recent_log]
        dates      = [d["date"][-5:] for d in recent_log]
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
            label_val  = min_eq + (row / height) * eq_range
            row_line  += f"  ${label_val:.0f}"
            chart.append(row_line)

        chart.append("  " + "".join(d[-2:] for d in dates))
        lines += chart
        lines.append("")

    # ── FOOTER ────────────────────────────────────────────────────────────────
    lines += [
        SEP,
        "  Soloway Rule: 'Patience. Let price come to your level.'",
        f"  Commission impact: ${5.0:.2f}/trade round-trip. Quality entries only.",
        "  This is a PAPER TRADING simulation. Not financial advice.",
        SEP,
        "",
    ]

    report_text = "\n".join(lines)
    print(report_text)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"report_{today_date}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  [Report saved to: {report_path}]")

    return report_text
