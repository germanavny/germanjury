"""
Soloway Strategy Engine for MSFT
Signals based on:
  - Fibonacci retracements from swing high/low
  - Moving averages (MA20 / MA50 / MA200)
  - RSI (overbought/oversold)
  - Volume analysis (climactic spikes)
  - Trend bias (death cross / golden cross)
  - Master Levels from the historical analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


# ── MASTER LEVELS (from Agent 4 analysis) ────────────────────────────────
# Key S/R levels MSFT has historically respected most
MASTER_LEVELS = {
    # SUPPORT ZONES
    "S5_nuclear":        (280.0,  295.0),   # Nuclear capitulation
    "S4_78pct_fib":      (309.0,  320.0),   # 78.6% retrace from ATH
    "S3_double_fib":     (339.0,  345.0),   # 61.8% ATH + 50% COVID bull (MASTER)
    "S2_50pct_fib_ath":  (368.0,  376.0),   # 50% retrace from ATH (current)
    "S1_38pct_fib_ath":  (392.0,  400.0),   # 38.2% retrace from ATH
    # RESISTANCE ZONES
    "R1_ma50_area":      (390.0,  405.0),   # MA50 area + old support
    "R2_mid":            (428.0,  442.0),   # 2024 resistance cluster
    "R3_pre_ath":        (458.0,  480.0),   # Pre-ATH distribution zone
    "R4_ath":            (545.0,  560.0),   # ATH zone
}

# Risk parameters
RISK_PER_TRADE_PCT  = 0.02    # Risk 2% of equity per trade
STOP_LOSS_PCT_LONG  = 0.035   # 3.5% stop for longs
STOP_LOSS_PCT_SHORT = 0.030   # 3.0% stop for shorts
TP_LONG_1           = 0.055   # First target: +5.5%
TP_LONG_2           = 0.10    # Second target: +10%
TP_SHORT_1          = 0.055   # Short first target: -5.5%
TRAIL_STOP_PCT      = 0.025   # Trailing stop once in profit


def fetch_data(ticker: str = "MSFT", period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV daily data."""
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    df = df.copy()

    # Moving Averages
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # RSI (14)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Volume SMA
    df["VolMA20"]   = df["Volume"].rolling(20).mean()
    df["VolRatio"]  = df["Volume"] / df["VolMA20"]

    # ATR (14)
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # Bollinger Bands (20, 2)
    df["BB_mid"]   = df["MA20"]
    df["BB_std"]   = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    # Daily return & realized vol
    df["DailyRet"]   = df["Close"].pct_change()
    df["RealVol20"]  = df["DailyRet"].rolling(20).std() * np.sqrt(252) * 100

    # Candle features
    df["Body"]        = df["Close"] - df["Open"]
    df["UpperWick"]   = df["High"]  - df[["Close","Open"]].max(axis=1)
    df["LowerWick"]   = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["IsBullBar"]   = df["Body"] > 0
    df["IsHammer"]    = (
        (df["LowerWick"] > 2 * df["Body"].abs()) &
        (df["UpperWick"] < df["Body"].abs() * 0.5) &
        (df["Body"] > 0)
    )

    # Swing high/low (20-day)
    df["SwingHigh20"] = df["High"].rolling(20).max()
    df["SwingLow20"]  = df["Low"].rolling(20).min()

    # Fibonacci from 60-day swing
    df["Swing60H"] = df["High"].rolling(60).max()
    df["Swing60L"] = df["Low"].rolling(60).min()
    diff60 = df["Swing60H"] - df["Swing60L"]
    df["Fib61_8"]  = df["Swing60H"] - 0.618 * diff60   # Golden ratio support
    df["Fib50"]    = df["Swing60H"] - 0.500 * diff60
    df["Fib38_2"]  = df["Swing60H"] - 0.382 * diff60

    # Trend
    df["TrendBull"]  = df["MA50"] > df["MA200"]          # Golden cross regime
    df["TrendBear"]  = df["MA50"] < df["MA200"]          # Death cross regime
    df["AboveMa200"] = df["Close"] > df["MA200"]
    df["AboveMa50"]  = df["Close"] > df["MA50"]
    df["AboveMa20"]  = df["Close"] > df["MA20"]

    # Price position relative to range
    df["PosInRange"] = (df["Close"] - df["Swing60L"]) / (df["Swing60H"] - df["Swing60L"] + 0.01)

    return df


def _near_level(price: float, zone_low: float, zone_high: float, buffer: float = 0.01) -> bool:
    """Check if price is within or near a master level zone."""
    low  = zone_low  * (1 - buffer)
    high = zone_high * (1 + buffer)
    return low <= price <= high


def generate_signal(df: pd.DataFrame) -> dict:
    """
    Main Soloway signal generator.
    Returns a dict with: signal, label, strength, notes
    """
    if len(df) < 60:
        return {"signal": "HOLD", "label": "Insufficient data", "strength": 0, "notes": []}

    today  = df.iloc[-1]
    prev   = df.iloc[-2]
    prev2  = df.iloc[-3]
    price  = float(today["Close"])
    notes  = []
    score  = 0   # positive = long bias, negative = short bias

    # ── TREND REGIME ────────────────────────────────────────────────────
    trend_bear = bool(today["TrendBear"])
    above_ma200 = bool(today["AboveMa200"])
    above_ma50  = bool(today["AboveMa50"])
    above_ma20  = bool(today["AboveMa20"])

    if trend_bear:
        notes.append("BEAR REGIME: Death cross active (MA50 < MA200)")
        score -= 2
    else:
        notes.append("BULL REGIME: Golden cross active (MA50 > MA200)")
        score += 2

    if not above_ma200:
        notes.append(f"Price BELOW MA200 (${today['MA200']:.2f}) -- bearish")
        score -= 1
    if not above_ma50:
        notes.append(f"Price BELOW MA50 (${today['MA50']:.2f}) -- bearish")
        score -= 1
    if not above_ma20:
        notes.append(f"Price BELOW MA20 (${today['MA20']:.2f}) -- bearish")
        score -= 0.5

    # ── RSI ──────────────────────────────────────────────────────────────
    rsi = float(today["RSI"])
    notes.append(f"RSI(14): {rsi:.1f}")
    if rsi < 28:
        notes.append("RSI OVERSOLD (<28) -- extreme fear, buy signal")
        score += 3
    elif rsi < 35:
        notes.append("RSI oversold zone (28-35)")
        score += 1.5
    elif rsi > 72:
        notes.append("RSI OVERBOUGHT (>72) -- sell signal")
        score -= 3
    elif rsi > 65:
        notes.append("RSI overbought zone (65-72)")
        score -= 1.5

    # ── VOLUME ───────────────────────────────────────────────────────────
    vol_ratio = float(today["VolRatio"])
    notes.append(f"Volume ratio vs 20d avg: {vol_ratio:.2f}x")
    if vol_ratio > 2.5:
        if bool(today["IsBullBar"]):
            notes.append("CLIMACTIC VOLUME + bullish candle -- accumulation signal")
            score += 2
        else:
            notes.append("CLIMACTIC VOLUME + bearish candle -- distribution / capitulation")
            score += 1.5   # capitulation can mark bottom
    elif vol_ratio < 0.6:
        notes.append("Very low volume -- indecision")

    # ── MASTER LEVELS ────────────────────────────────────────────────────
    for level_name, (z_low, z_high) in MASTER_LEVELS.items():
        if _near_level(price, z_low, z_high):
            side = "SUPPORT" if level_name.startswith("S") else "RESISTANCE"
            notes.append(f"*** NEAR MASTER LEVEL: {level_name} ({side}) [{z_low:.0f}-{z_high:.0f}] ***")
            if side == "SUPPORT":
                score += 1.5
            else:
                score -= 1.5

    # ── FIBONACCI ────────────────────────────────────────────────────────
    fib618 = float(today["Fib61_8"])
    fib50  = float(today["Fib50"])
    fib382 = float(today["Fib38_2"])

    if abs(price - fib618) / price < 0.015:
        notes.append(f"Near 61.8% Fibonacci level (${fib618:.2f}) -- key support/resistance")
        score += 1.5 if price > float(today["Swing60L"]) + (float(today["Swing60H"]) - float(today["Swing60L"])) * 0.3 else -1.5
    if abs(price - fib50) / price < 0.015:
        notes.append(f"Near 50% Fibonacci level (${fib50:.2f})")
        score += 1.0
    if abs(price - fib382) / price < 0.015:
        notes.append(f"Near 38.2% Fibonacci level (${fib382:.2f})")
        score += 0.8

    # ── CANDLE PATTERNS ──────────────────────────────────────────────────
    if bool(today["IsHammer"]) and score > 0:
        notes.append("Hammer candle -- reversal signal")
        score += 1
    # Bearish engulfing
    if (not bool(today["IsBullBar"])) and bool(prev["IsBullBar"]) \
            and abs(float(today["Body"])) > abs(float(prev["Body"])):
        notes.append("Bearish engulfing pattern")
        score -= 1

    # ── PRICE POSITION IN RANGE ───────────────────────────────────────────
    pos_in_range = float(today["PosInRange"])
    notes.append(f"Position in 60d range: {pos_in_range*100:.1f}%")
    if pos_in_range < 0.15:
        notes.append("Price at BOTTOM of 60d range -- contrarian buy zone")
        score += 1.5
    elif pos_in_range > 0.85:
        notes.append("Price at TOP of 60d range -- contrarian sell zone")
        score -= 1.5

    # ── MOMENTUM (2-day) ─────────────────────────────────────────────────
    two_day_ret = (float(today["Close"]) / float(prev2["Close"]) - 1) * 100
    if two_day_ret < -4:
        notes.append(f"2-day drop: {two_day_ret:.1f}% -- oversold short-term")
        score += 0.5
    elif two_day_ret > 4:
        notes.append(f"2-day gain: {two_day_ret:.1f}% -- overbought short-term")
        score -= 0.5

    # ── BOLLINGER BANDS ───────────────────────────────────────────────────
    bb_pos = (price - float(today["BB_lower"])) / (float(today["BB_upper"]) - float(today["BB_lower"]) + 0.01)
    if bb_pos < 0.05:
        notes.append("Price at/below lower Bollinger Band -- oversold")
        score += 1
    elif bb_pos > 0.95:
        notes.append("Price at/above upper Bollinger Band -- overbought")
        score -= 1

    # ── FINAL SIGNAL DECISION ────────────────────────────────────────────
    notes.append(f"Total score: {score:.1f}")

    # Soloway discipline: In a death cross regime, require HIGHER threshold to go long
    if trend_bear:
        long_threshold  = 5.0   # Need very strong signal to go long in bear market
        short_threshold = -3.0  # Moderate signal enough to go short
    else:
        long_threshold  = 3.0
        short_threshold = -5.0   # Need strong signal to go short in bull market

    if score >= long_threshold:
        signal  = "LONG"
        label   = "Soloway Long Setup: Strong support signals"
        strength = min(score / 8.0, 1.0)
    elif score <= short_threshold:
        signal  = "SHORT"
        label   = "Soloway Short Setup: Resistance / trend continuation"
        strength = min(abs(score) / 8.0, 1.0)
    else:
        signal   = "HOLD"
        label    = "No clear setup -- stay flat or hold position"
        strength = 0.0

    return {
        "signal":   signal,
        "label":    label,
        "strength": round(strength, 3),
        "score":    round(score, 2),
        "notes":    notes,
        "price":    round(price, 2),
        "rsi":      round(rsi, 1),
        "vol_ratio": round(vol_ratio, 2),
        "ma20":     round(float(today["MA20"]), 2),
        "ma50":     round(float(today["MA50"]), 2),
        "ma200":    round(float(today["MA200"]), 2),
        "atr":      round(float(today["ATR14"]), 2),
        "trend_bear": trend_bear,
        "fib618":   round(fib618, 2),
        "fib50":    round(fib50, 2),
        "fib382":   round(fib382, 2),
    }


def size_position(equity: float, entry_price: float, stop_loss: float) -> float:
    """
    Kelly-lite position sizing.
    Risk 2% of equity. Position size = risk_amount / (entry - stop_loss).
    Returns number of shares (allows fractional).
    """
    risk_amount   = equity * RISK_PER_TRADE_PCT
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share < 0.01:
        return 0.0
    shares = risk_amount / risk_per_share
    # Cap: never risk more than 80% of equity in any single trade
    max_shares = (equity * 0.80) / entry_price
    return round(min(shares, max_shares), 4)


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
    tp    = position["take_profit"]
    low   = today_bar.get("low", current_price)
    high  = today_bar.get("high", current_price)

    if side == "long":
        # Stop loss hit?
        if low <= sl:
            return True, f"STOP LOSS hit at ${sl:.2f}"
        # Take profit hit?
        if high >= tp:
            return True, f"TAKE PROFIT hit at ${tp:.2f}"
        # Trailing stop (if up >3%, trail 2.5% below current)
        gain_pct = (current_price - entry) / entry
        if gain_pct > 0.03:
            trail_stop = current_price * (1 - TRAIL_STOP_PCT)
            if low <= trail_stop:
                return True, f"TRAILING STOP hit at ${trail_stop:.2f}"

    elif side == "short":
        # Stop loss hit?
        if high >= sl:
            return True, f"STOP LOSS hit at ${sl:.2f}"
        # Take profit hit?
        if low <= tp:
            return True, f"TAKE PROFIT hit at ${tp:.2f}"
        # Trailing stop
        gain_pct = (entry - current_price) / entry
        if gain_pct > 0.03:
            trail_stop = current_price * (1 + TRAIL_STOP_PCT)
            if high >= trail_stop:
                return True, f"TRAILING STOP hit at ${trail_stop:.2f}"

    return False, ""
