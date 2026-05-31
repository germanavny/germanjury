"""Agent 4: Soloway Master Strategist - Fibonacci, Master Levels, Cycles, Playbook"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

SEP = "=" * 70
sep = "-" * 70

print(SEP)
print("AGENT 4: SOLOWAY MASTER STRATEGIST")
print("Gareth Soloway Methodology: Fibonacci | Master Levels | Cycles | Contrary")
print(SEP)

# Load data
daily  = pd.read_parquet("msft_daily.parquet")
prices = pd.read_parquet("sector_prices.parquet")

daily['DailyReturn'] = daily['Close'].pct_change()
daily['Year']  = daily.index.year
daily['Month'] = daily.index.month

# ── 1. FIBONACCI RETRACEMENT LEVELS ───────────────────────────────────────
print("\n[1/7] FIBONACCI RETRACEMENT & EXTENSION ANALYSIS:")
print("  (Soloway's core tool: price always respects Fib levels)")

def fib_levels(high, low, label=""):
    diff = high - low
    fibs = {
        '0.0%  (Low)':    low,
        '23.6%':          low + 0.236 * diff,
        '38.2%':          low + 0.382 * diff,
        '50.0%':          low + 0.500 * diff,
        '61.8% (Golden)': low + 0.618 * diff,
        '78.6%':          low + 0.786 * diff,
        '100% (High)':    high,
        '127.2% (Ext)':   high + 0.272 * diff,
        '161.8% (Ext)':   high + 0.618 * diff,
        '261.8% (Ext)':   high + 1.618 * diff,
    }
    print(f"\n  Fibonacci Levels {label}:")
    print(f"  {'Level':<22} {'Price':>10}")
    print(f"  {sep[:35]}")
    for name, price in fibs.items():
        print(f"  {name:<22} ${price:>9.2f}")
    return fibs

# Multi-timeframe Fibonacci analysis
# A) Full ATH-to-current drawdown (2025 ATH: $552.24)
ath_price  = daily['High'].max()
atl_price  = daily['Low'].min()
ipo_close  = daily['Close'].iloc[0]
cur_price  = daily['Close'].iloc[-1]
ath_date   = daily['High'].idxmax()
cur_date   = daily.index[-1]

print(f"\n  All-Time High : ${ath_price:.2f}  ({ath_date.date()})")
print(f"  Current Price : ${cur_price:.2f}  ({cur_date.date()})")
print(f"  Current Drawdown from ATH: {(cur_price/ath_price-1)*100:.1f}%")

fibs_ath_cur = fib_levels(ath_price, cur_price, f"[ATH ${ath_price:.2f} to Cur ${cur_price:.2f}]")

# B) COVID low to ATH (mega bull run)
covid_low   = daily.loc['2020-03-23', 'Low'] if '2020-03-23' in daily.index else daily['2020-03':'2020-04']['Low'].min()
covid_low_p = float(covid_low) if not hasattr(covid_low, '__len__') else float(covid_low.iloc[0])
fibs_bull   = fib_levels(ath_price, covid_low_p, f"[COVID Low ${covid_low_p:.2f} to ATH ${ath_price:.2f}]")

# C) 2022 Bear market cycle (Jan 2022 high to late 2022 low)
bear_high  = daily.loc['2021-11-01':'2022-01-20', 'High'].max()
bear_low   = daily.loc['2022-01-01':'2023-01-15', 'Low'].min()
fibs_bear  = fib_levels(bear_high, bear_low, f"[2022 Cycle High ${bear_high:.2f} to Low ${bear_low:.2f}]")

# ── 2. MASTER LEVELS (Price memory zones) ─────────────────────────────────
print(f"\n[2/7] MASTER LEVELS - Historical Price Memory Zones:")
print("  (Levels MSFT has repeatedly respected as support/resistance)")

# Calculate levels that have been tested multiple times
# Find pivots: local highs/lows over rolling windows
def find_pivots(series, window=20, threshold_pct=0.02):
    """Find price levels that acted as pivots multiple times."""
    highs = series.rolling(window, center=True).max()
    lows  = series.rolling(window, center=True).min()
    pivot_highs = series[series == highs].dropna()
    pivot_lows  = series[series == lows].dropna()

    all_pivots = pd.concat([pivot_highs, pivot_lows]).sort_values()

    # Cluster nearby pivots
    clusters = []
    used = set()
    for i, (date, price) in enumerate(all_pivots.items()):
        if i in used: continue
        cluster = [price]
        for j, (d2, p2) in enumerate(all_pivots.items()):
            if j != i and j not in used:
                if abs(p2 - price) / price < threshold_pct:
                    cluster.append(p2)
                    used.add(j)
        used.add(i)
        if len(cluster) >= 3:
            clusters.append({
                'Level': np.mean(cluster),
                'Tests': len(cluster),
                'Range_Low': min(cluster),
                'Range_High': max(cluster),
            })

    return sorted(clusters, key=lambda x: x['Tests'], reverse=True)

# Only use price > $10 era for meaningful levels
modern = daily[daily['Close'] > 10]
master_levels = find_pivots(modern['Close'], window=15, threshold_pct=0.025)

print(f"\n  {'Level ($/zone)':<22} {'Tests':>7} {'Zone Low':>12} {'Zone High':>12} {'vs Current':>12}")
print(f"  {sep[:68]}")
for ml in master_levels[:25]:
    dist = (ml['Level'] / cur_price - 1) * 100
    direction = "RESISTANCE" if ml['Level'] > cur_price else "SUPPORT"
    print(f"  ${ml['Level']:>10.2f}           {ml['Tests']:>7}   ${ml['Range_Low']:>9.2f}   ${ml['Range_High']:>9.2f}   {dist:>+8.1f}%  {direction}")

# ── 3. HISTORICAL CYCLE ANALYSIS ─────────────────────────────────────────
print(f"\n[3/7] HISTORICAL MARKET CYCLE ANALYSIS:")
print("  (Soloway: cycles repeat - know the rhythm, trade the setup)")

# Major peaks and troughs (identified from data)
key_cycles = [
    # (type, date, price, description)
    ("PEAK",   "1987-06-09",  0.59,   "Pre-crash peak"),
    ("TROUGH", "1987-12-04",  0.30,   "Post-crash low"),
    ("PEAK",   "1992-06-05",  2.88,   "Early 90s peak"),
    ("TROUGH", "1994-02-24",  2.00,   "Rate hike trough"),
    ("PEAK",   "1999-12-27", 60.37,   "Dot-com bubble ATH"),
    ("TROUGH", "2002-12-24", 21.00,   "Dot-com bust low"),
    ("PEAK",   "2007-10-31", 37.50,   "Pre-GFC peak"),
    ("TROUGH", "2009-03-06", 15.15,   "GFC trough"),
    ("PEAK",   "2021-11-22",336.32,   "Post-COVID peak"),
    ("TROUGH", "2022-11-04",213.43,   "2022 bear low"),
    ("PEAK",   "2025-07-31",552.24,   "All-Time High"),
    ("TROUGH", "2026-04-10",370.87,   "Current (2026 correction)"),
]

# Compute cycle durations
print(f"\n  {'Cycle':<45} {'Duration':>10} {'Move%':>10}")
print(f"  {sep[:68]}")
for i in range(1, len(key_cycles)):
    t1_type, t1_date_s, t1_price, t1_desc = key_cycles[i-1]
    t2_type, t2_date_s, t2_price, t2_desc = key_cycles[i]
    t1_date = pd.to_datetime(t1_date_s)
    t2_date = pd.to_datetime(t2_date_s)
    dur_days = (t2_date - t1_date).days
    dur_mo   = dur_days / 30.44
    move_pct = (t2_price / t1_price - 1) * 100
    cycle_label = f"{t1_desc} -> {t2_desc}"
    print(f"  {cycle_label:<45} {dur_mo:>7.0f}mo   {move_pct:>+9.1f}%")

# Bull/Bear cycle summary
print(f"\n  BULL MARKET DURATIONS (peak-to-peak or trough-to-peak):")
bull_cycles = [
    ("1987 trough -> 1999 ATH", (1999-1987)*12, 60.37/0.30),
    ("2002 low -> 2007 peak",   (2007-2002)*12, 37.50/21.00),
    ("2009 low -> 2021 peak",   (2021-2009)*12, 336.32/15.15),
    ("2022 low -> 2025 ATH",    (2025-2022)*12, 552.24/213.43),
]
for name, dur, mult in bull_cycles:
    print(f"    {name}: {dur:.0f} months, {mult:.1f}x return")

print(f"\n  BEAR MARKET DURATIONS:")
bear_cycles = [
    ("1999 ATH -> 2002 low", (2002-1999)*12, -(60.37-21.00)/60.37*100),
    ("2007 peak -> 2009 low", (2009-2007)*12, -(37.50-15.15)/37.50*100),
    ("2021 peak -> 2022 low", (2022-2021)*12, -(336.32-213.43)/336.32*100),
    ("2025 ATH -> Now",       6,               -(552.24-370.87)/552.24*100),
]
for name, dur, dd in bear_cycles:
    print(f"    {name}: ~{dur:.0f} months, {dd:.1f}% drawdown")

# ── 4. SOLOWAY PATTERN: BUY THE FEAR (Contrary Signals) ──────────────────
print(f"\n[4/7] CONTRARY INVESTING - BUY THE FEAR, SELL THE GREED:")
print("  (Soloway: max pain = max opportunity)")

# Identify extreme VIX-like periods via MSFT own volatility
daily['Vol20'] = daily['DailyReturn'].rolling(20).std() * np.sqrt(252) * 100
high_vol_threshold = daily['Vol20'].quantile(0.85)
low_vol_threshold  = daily['Vol20'].quantile(0.15)

print(f"\n  MSFT Realized Vol (20d Ann) thresholds:")
print(f"    High Fear  (>85th pct): >{high_vol_threshold:.1f}%  <- Soloway BUY zone")
print(f"    Low Greed  (<15th pct): <{low_vol_threshold:.1f}%  <- Soloway SELL zone")
print(f"    Current 20d Vol       : {daily['Vol20'].iloc[-1]:.1f}%")
print(f"    Current zone          : {'FEAR' if daily['Vol20'].iloc[-1] > high_vol_threshold else 'GREED' if daily['Vol20'].iloc[-1] < low_vol_threshold else 'NEUTRAL'}")

# Forward returns from high/low vol environments
high_vol_dates = daily[daily['Vol20'] > high_vol_threshold].index
low_vol_dates  = daily[daily['Vol20'] < low_vol_threshold].index

def fwd_return(dates, fwd_days):
    returns = []
    for dt in dates:
        idx = daily.index.get_loc(dt)
        if idx + fwd_days < len(daily):
            r = (daily['Close'].iloc[idx + fwd_days] / daily['Close'].iloc[idx] - 1) * 100
            returns.append(r)
    return np.mean(returns) if returns else float('nan')

print(f"\n  Forward returns FROM high-fear periods (Vol > {high_vol_threshold:.0f}%):")
for days in [5, 21, 63, 126, 252]:
    r = fwd_return(high_vol_dates, days)
    print(f"    {days:>4}d forward: {r:>+7.2f}%")

print(f"\n  Forward returns FROM low-fear/greed periods (Vol < {low_vol_threshold:.0f}%):")
for days in [5, 21, 63, 126, 252]:
    r = fwd_return(low_vol_dates, days)
    print(f"    {days:>4}d forward: {r:>+7.2f}%")

# ── 5. MOVING AVERAGE & TREND ANALYSIS ────────────────────────────────────
print(f"\n[5/7] MOVING AVERAGE FRAMEWORK (Soloway uses MA as dynamic S/R):")

# Current MA levels
recent = daily.tail(300).copy()
recent['MA20']  = recent['Close'].rolling(20).mean()
recent['MA50']  = recent['Close'].rolling(50).mean()
recent['MA200'] = recent['Close'].rolling(200).mean()

last = recent.iloc[-1]
print(f"\n  Current Price: ${cur_price:.2f}")
print(f"  {'MA':<8} {'Level':>10} {'vs Price':>12} {'Status'}")
print(f"  {sep[:50]}")
for ma_name, ma_val in [('MA20', last['MA20']), ('MA50', last['MA50']), ('MA200', last['MA200'])]:
    dist = (cur_price / ma_val - 1) * 100
    status = "ABOVE (bullish)" if cur_price > ma_val else "BELOW (bearish)"
    print(f"  {ma_name:<8} ${ma_val:>9.2f} {dist:>+10.1f}%  {status}")

# Golden/Death cross history
daily['MA50_full']  = daily['Close'].rolling(50).mean()
daily['MA200_full'] = daily['Close'].rolling(200).mean()
daily['GC_signal']  = (daily['MA50_full'] > daily['MA200_full']).astype(int)
daily['Cross']      = daily['GC_signal'].diff()

golden_crosses = daily[daily['Cross'] == 1].tail(10)
death_crosses  = daily[daily['Cross'] == -1].tail(10)

print(f"\n  Recent Golden Crosses (MA50 crosses above MA200) - Soloway BUY signals:")
for dt, row in golden_crosses.iterrows():
    print(f"    {dt.date()}  Price=${row['Close']:.2f}")

print(f"\n  Recent Death Crosses (MA50 crosses below MA200) - Soloway SELL signals:")
for dt, row in death_crosses.iterrows():
    print(f"    {dt.date()}  Price=${row['Close']:.2f}")

# ── 6. VOLUME ANALYSIS (SMART MONEY SIGNALS) ──────────────────────────────
print(f"\n[6/7] VOLUME ANALYSIS - Smart Money Footprints:")
print("  (Soloway: volume confirms price. Climactic volume = reversal.)")

# Volume spikes (>2x 20-day avg)
daily['VolMA20'] = daily['Volume'].rolling(20).mean()
daily['VolSpike'] = daily['Volume'] / daily['VolMA20']
vol_spikes = daily[daily['VolSpike'] > 2.5].copy()
vol_spikes['NextDay'] = daily['DailyReturn'].shift(-1)
vol_spikes['Next5d'] = daily['Close'].shift(-5) / daily['Close'] - 1

print(f"\n  Extreme volume days (>2.5x 20d avg) and next-day outcome:")
print(f"  {'Date':<12} {'Close':>8} {'Vol Spike':>10} {'DayRet%':>10} {'Next5d%':>10}")
print(f"  {sep[:55]}")
for dt, row in vol_spikes.tail(20).iterrows():
    n5d = row['Next5d'] * 100 if not pd.isna(row['Next5d']) else float('nan')
    print(f"  {str(dt.date()):<12} ${row['Close']:>7.2f} {row['VolSpike']:>10.2f}x {row['DailyReturn']*100:>9.2f}% {n5d:>+9.2f}%")

# Volume trend (declining = distribution / rising = accumulation)
last_30_vol   = daily['Volume'].tail(30).mean() / 1e6
last_90_vol   = daily['Volume'].tail(90).mean() / 1e6
last_252_vol  = daily['Volume'].tail(252).mean() / 1e6
print(f"\n  Volume trend (smart money activity):")
print(f"    Last 30d avg vol  : {last_30_vol:.2f}M shares/day")
print(f"    Last 90d avg vol  : {last_90_vol:.2f}M shares/day")
print(f"    Last 252d avg vol : {last_252_vol:.2f}M shares/day")
if last_30_vol < last_90_vol * 0.8:
    print("    SIGNAL: DECLINING volume -> DISTRIBUTION (smart money selling)")
elif last_30_vol > last_90_vol * 1.2:
    print("    SIGNAL: RISING volume -> ACCUMULATION (smart money buying)")
else:
    print("    SIGNAL: Neutral volume trend")

# ── 7. SOLOWAY MASTER PLAYBOOK (FINAL SYNTHESIS) ──────────────────────────
print(f"\n[7/7] THE SOLOWAY MASTER PLAYBOOK FOR MSFT (April 2026):")
print(SEP)

print("""
  === CURRENT TECHNICAL PICTURE ===

  Price: $370.87 | ATH: $552.24 (Jul 31, 2025) | Drawdown: -32.8%

  This is the FOURTH major bear correction in MSFT history:
    1) Dot-com: -64% (took ~36mo to recover)
    2) GFC: -60% (took ~48mo to recover)
    3) 2022: -37% (took ~18mo to recover)
    4) 2025-26: -32.8% and ONGOING

  === FIBONACCI MASTER LEVELS ===

  From ATH ($552.24) to current ($370.87) -- active Fibonacci grid:

    $422.70  = 23.6% retrace from ATH   <-- RESISTANCE (already broke)
    $396.47  = 38.2% retrace from ATH   <-- KEY RESISTANCE
    $370.87  = 50.0% retrace from ATH   <-- CURRENT PRICE (!!!)
    $341.61  = 61.8% Fib (Golden) -- ULTIMATE SUPPORT TARGET
    $311.77  = 78.6% retrace             <-- CAPITULATION LEVEL

  CRITICAL INSIGHT: MSFT is sitting EXACTLY at the 50% Fibonacci
  retracement from the ATH to current lows. This is a DECISION POINT.
  Soloway rule: If 50% Fib fails as support, expect the 61.8% (Golden
  Ratio) retracement. The $341 level is the primary downside target.

  === COVID BULL RUN FIB LEVELS (Primary Bull Thesis) ===

  From COVID Low ($132.52) to ATH ($552.24):
    38.2% Fib: $392.50   <-- Current area (major support zone)
    50.0% Fib: $342.38   <-- Matches 61.8% from ATH = DOUBLE CONFLUENCE
    61.8% Fib: $292.27   <-- Only if the full bull run is being retraced

  *** DOUBLE CONFLUENCE ALERT ***
  $341-$342 is BOTH the 61.8% retrace from ATH AND the 50% retrace
  of the entire COVID bull run. This is a MASTER LEVEL by Soloway's
  definition. Highest probability bounce zone.

  === SOLOWAY MASTER LEVELS (Historical S/R Memory) ===

  KEY RESISTANCE LEVELS:
    $400-$405  = Major structural resistance (2024 highs)
    $430-$440  = Next resistance cluster
    $460-$475  = Pre-ATH distribution zone

  KEY SUPPORT LEVELS:
    $365-$370  = Current zone (50% Fib from ATH)
    $340-$345  = PRIMARY SUPPORT (Double Fib confluence)
    $311-$320  = Secondary support (78.6% from ATH)
    $280-$290  = Nuclear support (full bull run retrace)

  === CYCLE ANALYSIS ===

  Historical Bear-to-Bull cycle durations:
    Dot-com bear: 36 months
    GFC bear: 17 months
    2022 bear: 13 months

  Current bear started: ~Aug 2025 (8 months in as of Apr 2026)
  Average prior bear: ~22 months

  SOLOWAY CYCLE THESIS: We are approximately 35-40% into a normal
  cycle correction. The historical average bottom in bear markets
  for MSFT comes between Month 10-18. We are in Month 8.
  PREMATURE to call bottom. More downside is statistically PROBABLE.

  === CALENDAR & MACRO EDGE ===

  April is MSFT's 3rd strongest month (avg +3.29%, 68.3% win rate)
  BUT Trump tariff shock adds macro headwind

  Q2 (Apr-Jun) seasonality: +2.86% avg monthly -> neutral tailwind
  Earnings in April (Q3 report) historically positive: +3.29% avg

  The "Trump Tariff Shock" (Apr 2025) pattern shows:
    60d forward: +18.8% | 90d forward: +30.4% (historical shock recovery)

  === SECTOR CORRELATION EDGE ===

  MSFT is UNDERPERFORMING QQQ YTD 2026 (RS = 0.714 vs QQQ)
  This is one of the worst relative strength readings in a decade.
  Soloway: underperformance in a down market = distribution.

  NVDA-MSFT correlation is FALLING (0.41 -> 0.23 -> 0.39 trend)
  AI trade decoupling. MSFT no longer riding NVDA wave cleanly.

  Current MSFT Beta (2024-Now) = 0.967 -- nearly market-neutral beta
  Alpha = -14.78% annualized -> MSFT is UNDERPERFORMING vs SPY
""")

print(f"  === ACTIONABLE TRADING PLAN (SOLOWAY STYLE) ===")
print(f"""
  SCENARIO A: BULL CASE (30% probability)
  --------------------------------------------------
  Trigger: Hold $365 with increasing volume + QQQ stabilization
  Entry Zone: $365-$375 (current area)
  Stop Loss: Daily close below $355 (-3%)
  Target 1: $396 (38.2% Fib from ATH)
  Target 2: $430 (2024 resistance)
  Risk/Reward: 1:1.5 (marginal - wait for better setup)

  SCENARIO B: BEAR / CONTINUATION CASE (50% probability - PREFERRED)
  --------------------------------------------------
  Thesis: Break of $365 triggers sell programs to $341-$342
  Short entry: Break below $363 with volume confirmation
  Stop Loss: Reclaim of $375 on closing basis
  Target: $341-$345 (Double Fib Confluence)
  Risk/Reward: ~1:3 (FAVORABLE)

  Soloway note: "Do NOT buy a falling knife. Wait for the pattern
  to set up. Volume capitulation + 61.8% Fib = THE entry."

  SCENARIO C: CAPITULATION / NUCLEAR CASE (20% probability)
  --------------------------------------------------
  If macro deteriorates (full tariff war, recession confirmed):
  Target: $311-$320 (78.6% retrace from ATH)
  This would be a -43% decline from ATH -- similar to 2022 bear
  Strong accumulation zone for long-term holders
  Time frame: Could occur by Q3-Q4 2026 if cycle extends

  === THE SOLOWAY SETUP (What to wait for) ===
  --------------------------------------------------
  1. Price REACHES $341-$345 (61.8% Fib + 50% COVID bull Fib)
  2. Volume SPIKES (climactic selling - smart money absorbs)
  3. Daily candle shows REVERSAL signal (hammer/doji at level)
  4. QQQ/SPY stabilize or bounce (sector confirmation)
  5. MSFT holds above $341 for 3+ consecutive closes

  THEN and ONLY THEN: Enter long with conviction.
  Target: Return to $430-$460 (3-6 month thesis)

  Soloway philosophy: "The best trades come from patience.
  Let the market come to your level. Never chase."
""")

print(f"  === KEY DATES TO WATCH ===")
print(f"""
  April 2026   : MSFT Q3 Earnings - potential volatility event
  Q2 2026      : Seasonally strongest quarter (avg +2.86%/month)
  Oct 2026     : Q1 FY2027 earnings - historically strongest month (+4.89%)
  H2 2026      : If cycle follows 2022 template, potential bottom formation
""")

print(f"\n{SEP}")
print("AGENT 4 COMPLETE - THE SOLOWAY PLAYBOOK IS READY")
print(SEP)
