"""Agent 3: Sector & Correlation Specialist"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

SEP = "=" * 70
sep = "-" * 70

print(SEP)
print("AGENT 3: SECTOR & CORRELATION SPECIALIST")
print(SEP)

# ── 1. FETCH ALL TICKERS ─────────────────────────────────────────────────
tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA', 'QQQ', 'SPY', 'XLK']
print(f"\n[1/6] Fetching daily close data for: {', '.join(tickers)}")
print("  Using max available history...")

all_data = {}
for t in tickers:
    try:
        tk = yf.Ticker(t)
        hist = tk.history(start="1990-01-01", end=datetime.today().strftime("%Y-%m-%d"), auto_adjust=True)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        all_data[t] = hist['Close']
        print(f"  {t}: {len(hist):,} rows  ({hist.index[0].date()} to {hist.index[-1].date()})")
    except Exception as e:
        print(f"  {t}: ERROR - {e}")

prices = pd.DataFrame(all_data).dropna(how='all')
returns = prices.pct_change().dropna()
print(f"\n  Combined price matrix: {prices.shape[0]} rows x {prices.shape[1]} cols")
print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# ── 2. ROLLING CORRELATIONS (Full, 5Y, 3Y, 1Y) ───────────────────────────
print(f"\n[2/6] Correlation of MSFT daily returns vs peers:")
periods = {
    'Full History': returns,
    'Last 5 Years': returns[returns.index >= '2020-01-01'],
    'Last 3 Years': returns[returns.index >= '2022-01-01'],
    'Last 1 Year':  returns[returns.index >= '2024-01-01'],
}

for period_name, ret_df in periods.items():
    if 'MSFT' not in ret_df.columns: continue
    corrs = ret_df.corr()['MSFT'].drop('MSFT').sort_values(ascending=False)
    print(f"\n  {period_name}:")
    print(f"  {'Ticker':<8} {'Corr':>8}  {'Strength':}")
    print(f"  {sep[:40]}")
    for t, c in corrs.items():
        bar_len = int(abs(c) * 30)
        bar = ("#" * bar_len)
        print(f"  {t:<8} {c:>8.4f}  {bar}")

# ── 3. BETA ANALYSIS ─────────────────────────────────────────────────────
print(f"\n[3/6] MSFT Beta vs SPY (market benchmark):")
if 'SPY' in returns.columns:
    beta_periods = [
        ('Full History', returns),
        ('2010-2015',    returns[(returns.index>='2010-01-01') & (returns.index<'2015-01-01')]),
        ('2015-2020',    returns[(returns.index>='2015-01-01') & (returns.index<'2020-01-01')]),
        ('2020-2022',    returns[(returns.index>='2020-01-01') & (returns.index<'2022-01-01')]),
        ('2022-2024',    returns[(returns.index>='2022-01-01') & (returns.index<'2024-01-01')]),
        ('2024-Now',     returns[returns.index>='2024-01-01']),
    ]
    print(f"  {'Period':<20} {'Beta':>8} {'Alpha(ann%)':>13} {'R^2':>8}")
    print(f"  {sep[:52]}")
    for pname, ret_df in beta_periods:
        if 'MSFT' not in ret_df.columns or 'SPY' not in ret_df.columns: continue
        df2 = ret_df[['MSFT','SPY']].dropna()
        if len(df2) < 20: continue
        cov = np.cov(df2['MSFT'], df2['SPY'])
        beta = cov[0,1] / cov[1,1]
        alpha = (df2['MSFT'].mean() - beta * df2['SPY'].mean()) * 252 * 100
        corr  = df2['MSFT'].corr(df2['SPY'])
        r2    = corr ** 2
        print(f"  {pname:<20} {beta:>8.3f} {alpha:>13.2f} {r2:>8.4f}")

# ── 4. LEAD / LAG ANALYSIS ────────────────────────────────────────────────
print(f"\n[4/6] Lead/Lag analysis: Does any ticker LEAD MSFT?")
print("  (positive lag means the other ticker moves BEFORE MSFT)")
if 'AAPL' in returns.columns and 'GOOGL' in returns.columns:
    peers_ll = [t for t in ['AAPL','GOOGL','QQQ','NVDA','AMZN'] if t in returns.columns]
    recent_ret = returns[returns.index >= '2020-01-01']
    msft_r = recent_ret['MSFT'].dropna()

    print(f"\n  {'Ticker':<8} {'Lag-2d':>8} {'Lag-1d':>8} {'Lag0':>8} {'Lead1':>8} {'Lead2':>8}")
    print(f"  {sep[:55]}")
    for t in peers_ll:
        if t not in recent_ret.columns: continue
        peer_r = recent_ret[t].dropna()
        aligned = pd.DataFrame({'msft': msft_r, 'peer': peer_r}).dropna()
        row = [t]
        for lag in [-2, -1, 0, 1, 2]:
            c = aligned['msft'].corr(aligned['peer'].shift(lag))
            row.append(c)
        print(f"  {row[0]:<8} {row[1]:>8.4f} {row[2]:>8.4f} {row[3]:>8.4f} {row[4]:>8.4f} {row[5]:>8.4f}")
    print("  NOTE: Lead1/Lead2 = peer today predicts MSFT tomorrow/day-after")

# ── 5. SECTOR RELATIVE STRENGTH ───────────────────────────────────────────
print(f"\n[5/6] MSFT relative strength vs QQQ and SPY:")
# Rolling 252-day relative performance
if 'QQQ' in prices.columns and 'SPY' in prices.columns:
    rel_qqq = (prices['MSFT'] / prices['MSFT'].shift(252)) / \
              (prices['QQQ']  / prices['QQQ'].shift(252))
    rel_spy = (prices['MSFT'] / prices['MSFT'].shift(252)) / \
              (prices['SPY']  / prices['SPY'].shift(252))

    print(f"\n  Rolling 1Y Relative Strength (MSFT vs QQQ) by year-end:")
    print(f"  {'Year':<6} {'RS vs QQQ':>12} {'RS vs SPY':>12}  {'Edge':<10}")
    print(f"  {sep[:45]}")
    for yr in range(2005, datetime.today().year + 1):
        yr_data = rel_qqq[rel_qqq.index.year == yr]
        spy_data = rel_spy[rel_spy.index.year == yr]
        if len(yr_data) < 5: continue
        rs_q = yr_data.iloc[-1]
        rs_s = spy_data.iloc[-1] if len(spy_data) > 0 else float('nan')
        edge = "OUTPERFORM" if rs_q > 1 else "UNDERPERFORM"
        print(f"  {yr:<6} {rs_q:>12.3f} {rs_s:>12.3f}  {edge}")

# ── 6. NVDA/MSFT CORRELATION (AI ERA) ────────────────────────────────────
print(f"\n[6/6] NVDA vs MSFT correlation in AI era (2023-present):")
if 'NVDA' in returns.columns:
    ai_ret = returns[returns.index >= '2023-01-01']
    if 'MSFT' in ai_ret and 'NVDA' in ai_ret:
        ai_df = ai_ret[['MSFT','NVDA']].dropna()
        corr_nvda = ai_df['MSFT'].corr(ai_df['NVDA'])

        # Rolling 30d correlation
        roll_corr = ai_df['MSFT'].rolling(30).corr(ai_df['NVDA'])

        print(f"  Full AI era correlation (2023-now): {corr_nvda:.4f}")
        print(f"  Rolling 30d correlation stats:")
        print(f"    Mean  : {roll_corr.mean():.4f}")
        print(f"    Max   : {roll_corr.max():.4f}")
        print(f"    Min   : {roll_corr.min():.4f}")
        print(f"    Std   : {roll_corr.std():.4f}")

        # Recent 90 days
        recent_90 = roll_corr.dropna().tail(90)
        print(f"\n  Last 90 days rolling 30d correlation trend:")
        q1 = recent_90.iloc[:30].mean()
        q2 = recent_90.iloc[30:60].mean()
        q3 = recent_90.iloc[60:].mean()
        print(f"    Days 1-30  : {q1:.4f}")
        print(f"    Days 31-60 : {q2:.4f}")
        print(f"    Days 61-90 : {q3:.4f}")
        trend = "RISING" if q3 > q1 else "FALLING"
        print(f"    Trend      : {trend} correlation")

# Save prices for Agent 4
prices.to_parquet("sector_prices.parquet")
returns.to_parquet("sector_returns.parquet")
print(f"\n  [Saved sector_prices.parquet and sector_returns.parquet]")

print(f"\n{SEP}")
print("AGENT 3 COMPLETE")
print(SEP)
