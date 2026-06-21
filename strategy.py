"""
Soloway Strategy Engine — Multi-Ticker
Signals based on:
  - Fibonacci retracements from swing high/low
  - Moving averages (MA20 / MA50 / MA200)
  - RSI (overbought/oversold)
  - Volume analysis (climactic spikes)
  - Trend bias (death cross / golden cross)
  - Master Levels (MSFT only — hand-researched S/R zones)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ── MSFT MASTER LEVELS (hand-researched key S/R zones) ────────────────────────
MASTER_LEVELS = {
    "S5_nuclear":        (280.0,  295.0),
    "S4_78pct_fib":      (309.0,  320.0),
    "S3_double_fib":     (339.0,  345.0),   # MASTER — double fib confluence
    "S2_50pct_fib_ath":  (368.0,  376.0),
    "S1_38pct_fib_ath":  (392.0,  400.0),
    "R1_ma50_area":      (390.0,  405.0),
    "R2_mid":            (428.0,  442.0),
    "R3_pre_ath":        (458.0,  480.0),
    "R4_ath":            (545.0,  560.0),
}

# Per-ticker master levels — other tickers rely on dynamic Fibonacci + MA only
MASTER_LEVELS_BY_TICKER = {
    "MSFT": MASTER_LEVELS,
    "AAPL": {},
    "NVDA": {},
    "GOOGL": {},
    "AMZN": {},
}

# Risk parameters — TREND-FOLLOWING engine (validated on 40y, 5 tickers)
# Backtest verdict: trend + ATR stops + ride winners = 3x return, HALF the drawdown
# vs the old contrarian model (PF 1.6 vs 1.0). See research_backtest.py.
RISK_PER_TRADE_PCT  = 0.02    # Risk 2% of total equity per trade
ATR_STOP_MULT       = 2.5     # Stop distance = 2.5 x ATR(14) — adapts to volatility
TRAIL_STOP_PCT      = 0.06    # Wide 6% trailing stop — let winners run, don't choke them
MAX_HOLD_DAYS       = 40      # Swing horizon: ride the trend up to 40 days
ENTRY_THRESHOLD     = 4.0     # |score| needed to trigger an entry

# ── ANTI-WHIPSAW: cooldown after a stop-out (validated 2026-06, test_filters.py) ──
# After a ticker stops us out, don't re-enter it for COOLDOWN_DAYS. A name just
# chopped out in a sideways tape tends to keep chopping. 20y long-only portfolio sim:
# baseline PF 1.80 / CAGR 10.8% / MaxDD -15%  →  cd=7  PF 2.05 / CAGR 11.1% / MaxDD -13%.
# Strict improvement on return AND risk AND commissions. (ADX gate was TESTED & REJECTED:
# it halved CAGR — pullback entries inherently have low ADX, so it killed good trades.)
COOLDOWN_DAYS       = 7

# LONG-ONLY: portfolio backtest (2006-2026) proved shorts are a net DRAG on these
# mega-cap tech names — even with the regime gate. Removing shorts: CAGR 7.7%→11.0%,
# MaxDD -26%→-15%, PF 1.45→1.86. Shorts disabled by default. See portfolio_backtest.py.
ALLOW_SHORTS        = False

# Legacy fixed-percent params (kept for backward compatibility / fallback only)
STOP_LOSS_PCT_LONG  = 0.035
STOP_LOSS_PCT_SHORT = 0.030
TP_LONG_1           = 0.055
TP_SHORT_1          = 0.055


def fetch_data(ticker: str = "MSFT", period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV daily data."""
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    df = df.copy()

    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VolMA20"]  = df["Volume"].rolling(20).mean()
    df["VolRatio"] = df["Volume"] / df["VolMA20"]

    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # ── ADX(14) — trend-strength filter (kills choppy-market whipsaw entries) ──
    up_move   = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_w     = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_w
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_w
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX14"] = dx.ewm(alpha=1/14, adjust=False).mean()

    df["BB_mid"]   = df["MA20"]
    df["BB_std"]   = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    df["DailyRet"]  = df["Close"].pct_change()
    df["RealVol20"] = df["DailyRet"].rolling(20).std() * np.sqrt(252) * 100

    df["Body"]      = df["Close"] - df["Open"]
    df["UpperWick"] = df["High"]  - df[["Close","Open"]].max(axis=1)
    df["LowerWick"] = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["IsBullBar"] = df["Body"] > 0
    df["IsHammer"]  = (
        (df["LowerWick"] > 2 * df["Body"].abs()) &
        (df["UpperWick"] < df["Body"].abs() * 0.5) &
        (df["Body"] > 0)
    )

    df["SwingHigh20"] = df["High"].rolling(20).max()
    df["SwingLow20"]  = df["Low"].rolling(20).min()

    df["Swing60H"] = df["High"].rolling(60).max()
    df["Swing60L"] = df["Low"].rolling(60).min()
    diff60 = df["Swing60H"] - df["Swing60L"]
    df["Fib61_8"] = df["Swing60H"] - 0.618 * diff60
    df["Fib50"]   = df["Swing60H"] - 0.500 * diff60
    df["Fib38_2"] = df["Swing60H"] - 0.382 * diff60

    # ── MACD (12/26/9) ───────────────────────────────────────────────────────
    exp12          = df["Close"].ewm(span=12, adjust=False).mean()
    exp26          = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]     = exp12 - exp26
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]= df["MACD"] - df["MACD_sig"]

    # ── RSI DIVERGENCE (10-bar lookback) ─────────────────────────────────────
    df["Price_ret_10"] = df["Close"].pct_change(10)
    df["RSI_chg_10"]   = df["RSI"].diff(10)
    df["RSI_div_bull"] = (df["Price_ret_10"] < -0.02) & (df["RSI_chg_10"] > 5)
    df["RSI_div_bear"] = (df["Price_ret_10"] >  0.02) & (df["RSI_chg_10"] < -5)

    df["TrendBull"]  = df["MA50"] > df["MA200"]
    df["TrendBear"]  = df["MA50"] < df["MA200"]
    df["AboveMa200"] = df["Close"] > df["MA200"]
    df["AboveMa50"]  = df["Close"] > df["MA50"]
    df["AboveMa20"]  = df["Close"] > df["MA20"]

    df["PosInRange"] = (df["Close"] - df["Swing60L"]) / (df["Swing60H"] - df["Swing60L"] + 0.01)

    return df


def _near_level(price: float, zone_low: float, zone_high: float, buffer: float = 0.01) -> bool:
    return zone_low * (1 - buffer) <= price <= zone_high * (1 + buffer)


def get_market_regime(period: str = "1y") -> dict:
    """
    Market regime from QQQ (Nasdaq-100 proxy) — gates direction for ALL tickers.
    The #1 lesson from the backtest: NEVER short a confirmed bull market, and be
    cautious going long in a confirmed bear. Returns dict with market_up + context.
    """
    try:
        qqq = yf.Ticker("QQQ").history(period=period, auto_adjust=True)
        qqq.index = pd.to_datetime(qqq.index).tz_localize(None)
        close = qqq["Close"]
        ma50  = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        px    = float(close.iloc[-1])
        market_up   = bool(px > ma50 and ma50 > ma200)
        market_down = bool(px < ma50 and ma50 < ma200)
        if market_up:
            label = "RISK-ON (QQQ above MA50 > MA200) — favor LONGS, block SHORTS"
        elif market_down:
            label = "RISK-OFF (QQQ below MA50 < MA200) — favor SHORTS, block LONGS"
        else:
            label = "MIXED / choppy market — both directions allowed, smaller size"
        return {"market_up": market_up, "market_down": market_down,
                "qqq_price": round(px, 2), "label": label}
    except Exception:
        # If QQQ unavailable, don't gate — allow both directions
        return {"market_up": False, "market_down": False, "qqq_price": 0.0,
                "label": "Market regime unavailable — no gate applied"}


def generate_signal(df: pd.DataFrame, ticker: str = "MSFT", market: dict | None = None) -> dict:
    """
    TREND-FOLLOWING signal generator — trades WITH the trend (Soloway-aligned).
    Buys strength on healthy pullbacks; shorts only genuine downtrends.
    `market` = optional regime dict from get_market_regime() to gate direction.
    Returns a dict with: signal, label, strength, score, notes, and all indicators.
    """
    if len(df) < 60:
        return {"signal": "HOLD", "label": "Insufficient data", "strength": 0, "score": 0, "notes": []}

    today  = df.iloc[-1]
    prev   = df.iloc[-2]
    price  = float(today["Close"])
    ma20   = float(today["MA20"])
    ma50   = float(today["MA50"])
    ma200  = float(today["MA200"])
    rsi    = float(today["RSI"])
    notes  = []
    score  = 0.0

    macd_now      = float(today["MACD"])
    macd_sig_val  = float(today["MACD_sig"])
    macd_prev     = float(prev["MACD"])
    macd_prev_sig = float(prev["MACD_sig"])
    vol_ratio     = float(today["VolRatio"])

    uptrend   = price > ma50 > ma200
    downtrend = price < ma50 < ma200

    # ── LONG SETUP: ride an established uptrend, enter on healthy pullbacks ────
    if uptrend:
        notes.append(f"{ticker}: UPTREND — price > MA50 (${ma50:.2f}) > MA200 (${ma200:.2f})")
        score += 2.0
        dist_ma20 = (price - ma20) / ma20
        if -0.02 <= dist_ma20 <= 0.03:
            notes.append("Pullback to MA20 — prime trend-continuation entry")
            score += 1.5
        if price <= ma50 * 1.02:
            notes.append("Deeper pullback near MA50 — strong support in uptrend")
            score += 1.0
        if 40 <= rsi <= 62:
            notes.append(f"RSI {rsi:.0f} — healthy momentum, not overbought")
            score += 1.0
        elif rsi < 40:
            notes.append(f"RSI {rsi:.0f} — oversold dip INSIDE uptrend (buy the dip)")
            score += 1.5
        if macd_now > macd_sig_val:
            score += 1.0
        if macd_prev < macd_prev_sig and macd_now > macd_sig_val:
            notes.append("Fresh BULLISH MACD cross — momentum turning up")
            score += 1.5
        if bool(today["RSI_div_bull"]):
            notes.append("Bullish RSI divergence — adds conviction")
            score += 1.0
        if bool(today["IsHammer"]):
            notes.append("Hammer candle at support")
            score += 0.5

    # ── SHORT SETUP: only genuine downtrends with weakness ────────────────────
    elif downtrend:
        notes.append(f"{ticker}: DOWNTREND — price < MA50 (${ma50:.2f}) < MA200 (${ma200:.2f})")
        score -= 2.0
        dist_ma20 = (price - ma20) / ma20
        if -0.03 <= dist_ma20 <= 0.02:
            notes.append("Bounce to MA20 — short the rally in a downtrend")
            score -= 1.5
        if price >= ma50 * 0.98:
            notes.append("Bounce near MA50 resistance")
            score -= 1.0
        if 38 <= rsi <= 60:
            notes.append(f"RSI {rsi:.0f} — weak momentum")
            score -= 1.0
        elif rsi > 60:
            notes.append(f"RSI {rsi:.0f} — overbought bounce in downtrend (short it)")
            score -= 1.5
        if macd_now < macd_sig_val:
            score -= 1.0
        if macd_prev > macd_prev_sig and macd_now < macd_sig_val:
            notes.append("Fresh BEARISH MACD cross — momentum rolling over")
            score -= 1.5
        if bool(today["RSI_div_bear"]):
            notes.append("Bearish RSI divergence — adds conviction")
            score -= 1.0
    else:
        notes.append(f"{ticker}: No clean trend (price between MA50/MA200) — stand aside")

    # ── VOLUME CONFIRMATION (amplifies an existing directional bias) ──────────
    if vol_ratio > 1.8 and bool(today["IsBullBar"]) and score > 0:
        notes.append(f"Strong volume ({vol_ratio:.1f}x) on up day — confirms long")
        score += 1.0
    elif vol_ratio > 1.8 and (not bool(today["IsBullBar"])) and score < 0:
        notes.append(f"Strong volume ({vol_ratio:.1f}x) on down day — confirms short")
        score -= 1.0

    # ── MASTER LEVELS (MSFT): in a TREND model these CONFIRM, not fade ─────────
    master_levels = MASTER_LEVELS_BY_TICKER.get(ticker, {})
    for level_name, (z_low, z_high) in master_levels.items():
        if _near_level(price, z_low, z_high):
            side = "SUPPORT" if level_name.startswith("S") else "RESISTANCE"
            notes.append(f"*** NEAR MASTER LEVEL: {level_name} ({side}) [{z_low:.0f}-{z_high:.0f}] ***")
            # support in an uptrend confirms long; resistance in a downtrend confirms short
            if side == "SUPPORT" and uptrend:
                score += 0.8
            elif side == "RESISTANCE" and downtrend:
                score -= 0.8

    raw_score = score

    # ── FINAL SIGNAL DECISION ─────────────────────────────────────────────────
    if score >= ENTRY_THRESHOLD:
        signal = "LONG"
    elif score <= -ENTRY_THRESHOLD and ALLOW_SHORTS:
        signal = "SHORT"
    else:
        signal = "HOLD"

    # ── MARKET REGIME GATE (the rule that would have saved us on MSFT) ─────────
    gated = False
    if market is not None and signal != "HOLD":
        if signal == "SHORT" and market.get("market_up"):
            notes.append(f"GATE: SHORT blocked — {market.get('label','')}")
            signal, gated = "HOLD", True
        elif signal == "LONG" and market.get("market_down"):
            notes.append(f"GATE: LONG blocked — {market.get('label','')}")
            signal, gated = "HOLD", True

    if signal == "LONG":
        label    = f"{ticker} Long: trend-continuation (score {raw_score:+.1f})"
        strength = min(raw_score / 8.0, 1.0)
    elif signal == "SHORT":
        label    = f"{ticker} Short: downtrend weakness (score {raw_score:+.1f})"
        strength = min(abs(raw_score) / 8.0, 1.0)
    else:
        label    = f"{ticker}: No trade — {'gated by market regime' if gated else 'no trend setup'}"
        strength = 0.0

    notes.append(f"Total score: {raw_score:+.1f} (threshold ±{ENTRY_THRESHOLD})")

    return {
        "signal":     signal,
        "label":      label,
        "strength":   round(strength, 3),
        "score":      round(raw_score, 2),
        "notes":      notes,
        "price":      round(price, 2),
        "rsi":        round(rsi, 1),
        "vol_ratio":  round(vol_ratio, 2),
        "ma20":       round(ma20, 2),
        "ma50":       round(ma50, 2),
        "ma200":      round(ma200, 2),
        "atr":        round(float(today["ATR14"]), 2),
        "trend_bear": bool(today["TrendBear"]),
        "uptrend":    uptrend,
        "downtrend":  downtrend,
        "fib618":     round(float(today["Fib61_8"]), 2),
        "fib50":      round(float(today["Fib50"]), 2),
        "fib382":     round(float(today["Fib38_2"]), 2),
        "macd":       round(macd_now, 4),
        "macd_sig":   round(macd_sig_val, 4),
        "rsi_div_bull": bool(today["RSI_div_bull"]),
        "rsi_div_bear": bool(today["RSI_div_bear"]),
        "gated":      gated,
    }


def size_position(equity: float, entry_price: float, stop_loss: float,
                  max_positions: int = 1) -> float:
    """
    Position sizing:
    - Risk 2% of equity per trade
    - Cap: never more than (80% / max_positions) of equity in one trade
    Returns shares (fractional allowed).
    """
    risk_amount    = equity * RISK_PER_TRADE_PCT
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share < 0.01:
        return 0.0
    shares_by_risk = risk_amount / risk_per_share

    # Budget cap per position
    budget_per_pos   = (equity * 0.80) / max_positions
    max_shares_by_budget = budget_per_pos / entry_price

    return round(min(shares_by_risk, max_shares_by_budget), 4)


def check_exit(position: dict, today_bar: dict, current_price: float) -> tuple[bool, str]:
    """
    Check if open position should be exited today.
    Returns (should_exit, reason).
    """
    if not position["active"]:
        return False, ""

    side  = position["side"]
    entry = position["entry_price"]
    sl    = position["stop_loss"]
    tp    = position.get("take_profit")   # may be None in trend mode (ride winners)
    low   = today_bar.get("low", current_price)
    high  = today_bar.get("high", current_price)

    if side == "long":
        if low <= sl:
            return True, f"STOP LOSS hit at ${sl:.2f}"
        if tp and high >= tp:
            return True, f"TAKE PROFIT hit at ${tp:.2f}"
        gain_pct = (current_price - entry) / entry
        if gain_pct > 0.03:
            trail_stop = current_price * (1 - TRAIL_STOP_PCT)
            if low <= trail_stop:
                return True, f"TRAILING STOP hit at ${trail_stop:.2f}"

    elif side == "short":
        if high >= sl:
            return True, f"STOP LOSS hit at ${sl:.2f}"
        if tp and low <= tp:
            return True, f"TAKE PROFIT hit at ${tp:.2f}"
        gain_pct = (entry - current_price) / entry
        if gain_pct > 0.03:
            trail_stop = current_price * (1 + TRAIL_STOP_PCT)
            if high >= trail_stop:
                return True, f"TRAILING STOP hit at ${trail_stop:.2f}"

    return False, ""
