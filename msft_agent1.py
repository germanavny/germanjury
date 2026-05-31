"""Agent 1: Data Extraction & Quant Engineer"""
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
print("AGENT 1: DATA EXTRACTION & QUANT ENGINEER")
print(SEP)

# ── 1. FULL DAILY HISTORY ─────────────────────────────────────────────────
print("\n[1/6] Fetching MSFT full daily history from IPO (1986-03-13)...")
msft = yf.Ticker("MSFT")
daily = msft.history(start="1986-03-13", end=datetime.today().strftime("%Y-%m-%d"), auto_adjust=True)
daily.index = pd.to_datetime(daily.index).tz_localize(None)
print(f"  Rows: {len(daily):,}  |  {daily.index[0].date()} to {daily.index[-1].date()}")

daily['DailyReturn'] = daily['Close'].pct_change()
daily['LogReturn']   = np.log(daily['Close'] / daily['Close'].shift(1))
daily['DailyRange']  = (daily['High'] - daily['Low']) / daily['Open'] * 100
daily['Year']        = daily.index.year
daily['Month']       = daily.index.month
daily['DayOfWeek']   = daily.index.dayofweek   # 0=Mon

ath_date = daily['High'].idxmax().date()
atl_date = daily['Low'].idxmin().date()
total_ret = (daily['Close'].iloc[-1] / daily['Close'].iloc[0] - 1) * 100

print(f"\n  IPO Price       : ${daily['Close'].iloc[0]:.4f}")
print(f"  Latest Close    : ${daily['Close'].iloc[-1]:.2f}")
print(f"  All-Time High   : ${daily['High'].max():.2f}  on {ath_date}")
print(f"  All-Time Low    : ${daily['Low'].min():.4f}  on {atl_date}")
print(f"  Total IPO Return: {total_ret:,.1f}%")

# save for other agents
daily.to_parquet("msft_daily.parquet")
print("  [Saved msft_daily.parquet]")

# ── 2. DECADE PERFORMANCE ─────────────────────────────────────────────────
print(f"\n[2/6] Decade-by-decade performance:")
print(f"  {'Decade':<10} {'Return%':>10} {'AnnVol%':>9} {'AvgVol(M)':>11} {'AvgRange%':>11}")
print(f"  {sep[:60]}")
for ds in range(1980, 2030, 10):
    de = ds + 10
    d = daily[(daily['Year'] >= ds) & (daily['Year'] < de)]
    if len(d) < 5: continue
    ret = (d['Close'].iloc[-1] / d['Close'].iloc[0] - 1) * 100
    vol = d['DailyReturn'].std() * np.sqrt(252) * 100
    avg_v = d['Volume'].mean() / 1e6
    avg_r = d['DailyRange'].mean()
    print(f"  {str(ds)+'s':<10} {ret:>10.1f} {vol:>9.2f} {avg_v:>11.2f} {avg_r:>11.3f}")

# ── 3. ANNUAL STATS ───────────────────────────────────────────────────────
print(f"\n[3/6] Annual return, volatility, and max-drawdown:")
print(f"  {'Year':<6} {'Return%':>9} {'AnnVol%':>9} {'MaxDD%':>8} {'AvgVol(M)':>11}")
print(f"  {sep[:50]}")
annual_data = {}
for yr in range(1986, datetime.today().year + 1):
    d = daily[daily['Year'] == yr]
    if len(d) < 5: continue
    ret  = (d['Close'].iloc[-1] / d['Close'].iloc[0] - 1) * 100
    vol  = d['DailyReturn'].std() * np.sqrt(252) * 100
    cm   = d['Close'].cummax()
    mdd  = ((d['Close'] - cm) / cm * 100).min()
    avg_v = d['Volume'].mean() / 1e6
    annual_data[yr] = {'ret': ret, 'vol': vol, 'mdd': mdd, 'vol_m': avg_v}
    print(f"  {yr:<6} {ret:>9.1f} {vol:>9.2f} {mdd:>8.2f} {avg_v:>11.2f}")

# ── 4. MONTHLY SEASONALITY ────────────────────────────────────────────────
print(f"\n[4/6] Monthly seasonality (average return % since 1986):")
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
# Monthly close-to-close
daily_copy = daily.copy()
daily_copy['YearMonth'] = daily_copy.index.to_period('M')
monthly = daily_copy.groupby('YearMonth')['Close'].last()
monthly_ret = monthly.pct_change() * 100
monthly_df = monthly_ret.reset_index()
monthly_df['Month'] = monthly_df['YearMonth'].apply(lambda x: x.month)
monthly_df['Year']  = monthly_df['YearMonth'].apply(lambda x: x.year)
monthly_df.dropna(inplace=True)

month_stats = monthly_df.groupby('Month')['Close'].agg(['mean','std','count'])
month_stats['pos_rate'] = monthly_df.groupby('Month')['Close'].apply(lambda x: (x>0).mean()*100)
print(f"  {'Month':<6} {'AvgRet%':>9} {'StdDev':>8} {'WinRate%':>10}")
print(f"  {sep[:40]}")
for m in range(1, 13):
    if m in month_stats.index:
        r = month_stats.loc[m,'mean']
        s = month_stats.loc[m,'std']
        w = month_stats.loc[m,'pos_rate']
        bar = "+" * int(max(0, r)) if r >= 0 else "-" * int(abs(min(0, r)))
        print(f"  {month_names[m]:<6} {r:>9.2f} {s:>8.2f} {w:>10.1f}   {bar}")

# ── 5. DAY-OF-WEEK EFFECT ─────────────────────────────────────────────────
print(f"\n[5/6] Day-of-week return effect:")
dow_names = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday'}
dow_stats = daily.groupby('DayOfWeek')['DailyReturn'].agg(['mean','std'])
dow_stats['pos_rate'] = daily.groupby('DayOfWeek')['DailyReturn'].apply(lambda x: (x>0).mean()*100)
print(f"  {'Day':<12} {'AvgRet%':>9} {'StdDev':>8} {'WinRate%':>10}")
print(f"  {sep[:45]}")
for d in range(5):
    r = dow_stats.loc[d,'mean'] * 100
    s = dow_stats.loc[d,'std'] * 100
    w = dow_stats.loc[d,'pos_rate']
    print(f"  {dow_names[d]:<12} {r:>9.4f} {s:>8.4f} {w:>10.1f}")

# ── 6. INTRADAY VOLATILITY (1h bars, last 2y) ─────────────────────────────
print(f"\n[6/6] Intraday hourly volatility profile (last 2 years, 1h bars):")
try:
    intra = msft.history(period="2y", interval="1h", auto_adjust=True)
    intra.index = pd.to_datetime(intra.index).tz_localize(None)
    intra['Return']    = intra['Close'].pct_change()
    intra['AbsReturn'] = intra['Return'].abs()
    intra['Hour']      = intra.index.hour
    intra['Range']     = (intra['High'] - intra['Low']) / intra['Open'] * 100

    h_vol  = intra.groupby('Hour')['AbsReturn'].mean() * 100
    h_rng  = intra.groupby('Hour')['Range'].mean()
    h_volu = intra.groupby('Hour')['Volume'].mean() / 1e6

    intra.to_parquet("msft_intraday.parquet")

    labels = {9:'OPEN (9-10)',10:'Post-open',11:'Pre-lunch',
              12:'MIDDAY',13:'Midday2',14:'Pre-power',
              15:'POWER HOUR',16:'Close'}

    print(f"\n  Hourly AbsReturn%, Range%, Volume(M) [Eastern Time]:")
    print(f"  {'Hour':<8} {'AbsRet%':>9} {'Range%':>8} {'Vol(M)':>8}  Label")
    print(f"  {sep[:60]}")
    for h in sorted(h_vol.index):
        if h < 9 or h > 16: continue
        lbl = labels.get(h, '')
        print(f"  {h:02d}:00    {h_vol[h]:>9.4f} {h_rng[h]:>8.4f} {h_volu[h]:>8.2f}  {lbl}")

    print(f"\n  Top 3 MOST volatile hours:")
    for h, v in h_vol.sort_values(ascending=False).head(3).items():
        print(f"    Hour {h:02d}:00 -> {v:.4f}% avg abs return")

    print(f"  Top 3 LEAST volatile hours (midday lull):")
    for h, v in h_vol.sort_values().head(3).items():
        if 9 <= h <= 16:
            print(f"    Hour {h:02d}:00 -> {v:.4f}% avg abs return")

except Exception as e:
    print(f"  Intraday error: {e}")

# ── SAVE ANNUAL SUMMARY for downstream agents ──────────────────────────────
import json
with open("msft_annual_summary.json", "w") as f:
    json.dump({str(k): v for k, v in annual_data.items()}, f, indent=2)

print(f"\n{SEP}")
print("AGENT 1 COMPLETE")
print(SEP)
