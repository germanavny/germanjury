"""Agent 2: Macro & Calendar Analyst"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

SEP = "=" * 70
sep = "-" * 70

print(SEP)
print("AGENT 2: MACRO & CALENDAR ANALYST")
print(SEP)

# Load data saved by Agent 1
daily = pd.read_parquet("msft_daily.parquet")
daily['DailyReturn'] = daily['Close'].pct_change()
daily['Year']  = daily.index.year
daily['Month'] = daily.index.month
daily['DayOfWeek'] = daily.index.dayofweek

# ── 1. MAJOR MACRO EVENTS & MSFT REACTION ────────────────────────────────
print("\n[1/5] MSFT price action around major macro / crisis events:")
print(f"  {'Event':<42} {'Before':>8} {'After':>8} {'6M':>8} {'12M':>8}")
print(f"  {sep[:73]}")

macro_events = [
    ("Black Monday (1987-10-19)",          "1987-10-19", 30, 90, 180, 365),
    ("Gulf War Start (1990-08-02)",         "1990-08-02", 30, 90, 180, 365),
    ("Tech IPO Boom peak (1999-12-31)",     "1999-12-31", 30, 90, 180, 365),
    ("Dot-com Crash bottom (2002-10-09)",   "2002-10-09", 30, 90, 180, 365),
    ("9/11 Attack (2001-09-11)",            "2001-09-11", 10, 30, 90,  180),
    ("Iraq War start (2003-03-20)",         "2003-03-20", 10, 30, 90,  180),
    ("Housing Bubble peak (2006-06-30)",    "2006-06-30", 30, 90, 180, 365),
    ("Lehman Collapse (2008-09-15)",        "2008-09-15", 30, 90, 180, 365),
    ("GFC Bottom (2009-03-09)",             "2009-03-09", 30, 90, 180, 365),
    ("EU Debt Crisis (2011-08-05)",         "2011-08-05", 30, 90, 180, 365),
    ("China Devaluation (2015-08-24)",      "2015-08-24", 10, 30, 90,  180),
    ("Brexit Vote (2016-06-24)",            "2016-06-24", 10, 30, 90,  180),
    ("Fed Rate Hike Cycle (2018-10-01)",    "2018-10-01", 30, 90, 180, 365),
    ("COVID Crash (2020-02-20)",            "2020-02-20", 30, 90, 180, 365),
    ("COVID Bottom (2020-03-23)",           "2020-03-23", 30, 90, 180, 365),
    ("Meme Stock Mania (2021-01-27)",       "2021-01-27", 10, 30, 90,  180),
    ("Fed Hike Cycle 2022 (2022-01-03)",    "2022-01-03", 30, 90, 180, 365),
    ("SVB Collapse (2023-03-10)",           "2023-03-10", 10, 30, 90,  180),
    ("AI Boom (ChatGPT, 2023-01-20)",       "2023-01-20", 30, 90, 180, 365),
    ("ATH $552 (2025-07-31)",               "2025-07-31", 30, 90, 180, 365),
    ("Trump Tariff Shock (2025-04-02)",     "2025-04-02", 10, 30, 60,  90),
]

for event_name, event_date_str, d1, d2, d3, d4 in macro_events:
    try:
        event_date = pd.to_datetime(event_date_str)
        # Find nearest trading day
        idx = daily.index.searchsorted(event_date)
        if idx >= len(daily): continue
        base_price = daily['Close'].iloc[idx]

        def pct_fwd(days):
            fwd_idx = idx + days
            if fwd_idx >= len(daily): return float('nan')
            return (daily['Close'].iloc[fwd_idx] / base_price - 1) * 100

        def pct_back(days):
            back_idx = max(0, idx - days)
            base_back = daily['Close'].iloc[back_idx]
            return (base_price / base_back - 1) * 100

        before = pct_back(30)
        r1 = pct_fwd(d1)
        r2 = pct_fwd(d2)
        r3 = pct_fwd(d3)

        def fmt(x):
            if np.isnan(x): return "  N/A  "
            return f"{x:+7.1f}%"

        print(f"  {event_name:<42} {fmt(before)} {fmt(r1)} {fmt(r2)} {fmt(r3)}")
    except Exception as e:
        print(f"  {event_name:<42} ERROR: {e}")

print(f"\n  * Before = 30-day return going INTO event; columns = {d1}d / {d2}d / {d3}d / {d4}d after event")

# ── 2. Q1-Q4 SEASONAL ANALYSIS ────────────────────────────────────────────
print(f"\n[2/5] Quarterly seasonality analysis:")
daily['Quarter'] = daily.index.quarter
q_names = {1:'Q1 (Jan-Mar)',2:'Q2 (Apr-Jun)',3:'Q3 (Jul-Sep)',4:'Q4 (Oct-Dec)'}

# Compute monthly returns per year-quarter
daily_copy = daily.copy()
daily_copy['YearMonth'] = daily_copy.index.to_period('M')
monthly = daily_copy.groupby('YearMonth')['Close'].last()
monthly_ret = monthly.pct_change() * 100
monthly_df = monthly_ret.reset_index()
monthly_df['Quarter'] = monthly_df['YearMonth'].apply(lambda x: x.month)
monthly_df['Quarter'] = monthly_df['Quarter'].map(
    {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4})
monthly_df.dropna(inplace=True)

print(f"  {'Quarter':<16} {'AvgMonthlyRet%':>16} {'StdDev':>9} {'WinRate%':>10}")
print(f"  {sep[:55]}")
for q in range(1, 5):
    qd = monthly_df[monthly_df['Quarter'] == q]['Close']
    mean_r = qd.mean()
    std_r  = qd.std()
    win_r  = (qd > 0).mean() * 100
    print(f"  {q_names[q]:<16} {mean_r:>16.3f} {std_r:>9.3f} {win_r:>10.1f}")

# Annual Q4 effect (Oct-Dec):
print(f"\n  Year-by-Year Q4 (Oct-Dec) vs Full-Year performance:")
print(f"  {'Year':<6} {'Q4 Ret%':>9} {'Full Yr%':>10}  {'Q4 vs FY':>10}")
print(f"  {sep[:40]}")
for yr in range(2000, datetime.today().year + 1):
    d_q4  = daily[(daily['Year'] == yr) & (daily['Quarter'] == 4)]
    d_fy  = daily[daily['Year'] == yr]
    if len(d_q4) < 5 or len(d_fy) < 5: continue
    r_q4 = (d_q4['Close'].iloc[-1] / d_q4['Close'].iloc[0] - 1) * 100
    r_fy = (d_fy['Close'].iloc[-1] / d_fy['Close'].iloc[0] - 1) * 100
    flag = "** STRONG Q4 **" if r_q4 > 10 else ("  WEAK Q4" if r_q4 < -5 else "")
    print(f"  {yr:<6} {r_q4:>9.1f} {r_fy:>10.1f}  {flag}")

# ── 3. HOLIDAY / WEEKEND EFFECTS ─────────────────────────────────────────
print(f"\n[3/5] Pre/Post holiday & weekend effects:")

# Days before and after major US market holidays (using common calendar dates)
# We approximate: Christmas Dec 25 -> look at Dec 23-24 and Dec 26-28
# Thanksgiving last Thu of Nov -> Nov last week
# Look at returns on specific calendar windows

def window_avg(month, day_range_start, day_range_end):
    mask = (daily['Month'] == month) & \
           (daily.index.day >= day_range_start) & \
           (daily.index.day <= day_range_end)
    return daily[mask]['DailyReturn'].mean() * 100, \
           daily[mask]['DailyReturn'].std() * 100, \
           (daily[mask]['DailyReturn'] > 0).mean() * 100

holiday_windows = [
    ("Pre-Christmas (Dec 20-23)",    12, 20, 23),
    ("Christmas week (Dec 24-26)",   12, 24, 26),
    ("Santa Rally (Dec 27-31)",      12, 27, 31),
    ("New Year week (Jan 2-7)",       1,  2,  7),
    ("MLK Week (Jan 14-20)",          1, 14, 20),
    ("Presidents Day week (Feb)",     2, 14, 20),
    ("Tax Day (Apr 12-18)",           4, 12, 18),
    ("July 4th week",                 7,  1,  7),
    ("Pre-Thanksgiving (Nov 20-26)", 11, 20, 26),
    ("Post-Thanksgiving (Nov 27-30)",11, 27, 30),
    ("Halloween week (Oct 28-31)",   10, 28, 31),
]

print(f"  {'Window':<32} {'AvgDayRet%':>12} {'WinRate%':>10}")
print(f"  {sep[:57]}")
for name, m, d_s, d_e in holiday_windows:
    mean_r, std_r, win_r = window_avg(m, d_s, d_e)
    print(f"  {name:<32} {mean_r:>12.4f} {win_r:>10.1f}")

# ── 4. END-OF-MONTH / START-OF-MONTH ────────────────────────────────────────
print(f"\n[4/5] End-of-month / Start-of-month rebalancing effect:")
# Last 3 trading days of month vs first 3 trading days
daily['IsEOM'] = False
daily['IsBOM'] = False

for period, group in daily.groupby(daily.index.to_period('M')):
    sorted_idx = group.index.sort_values()
    if len(sorted_idx) >= 3:
        daily.loc[sorted_idx[-3:], 'IsEOM'] = True
        daily.loc[sorted_idx[:3],  'IsBOM'] = True

mid_mask = (~daily['IsEOM']) & (~daily['IsBOM'])
eom_ret  = daily[daily['IsEOM']]['DailyReturn'].mean() * 100
bom_ret  = daily[daily['IsBOM']]['DailyReturn'].mean() * 100
mid_ret  = daily[mid_mask]['DailyReturn'].mean() * 100

eom_win  = (daily[daily['IsEOM']]['DailyReturn'] > 0).mean() * 100
bom_win  = (daily[daily['IsBOM']]['DailyReturn'] > 0).mean() * 100
mid_win  = (daily[mid_mask]['DailyReturn'] > 0).mean() * 100

print(f"  {'Period':<25} {'AvgDayRet%':>12} {'WinRate%':>10}")
print(f"  {sep[:50]}")
print(f"  {'End-of-Month (last 3d)':<25} {eom_ret:>12.4f} {eom_win:>10.1f}")
print(f"  {'Start-of-Month (first 3d)':<25} {bom_ret:>12.4f} {bom_win:>10.1f}")
print(f"  {'Mid-Month':<25} {mid_ret:>12.4f} {mid_win:>10.1f}")

# ── 5. EARNINGS SEASONALITY ────────────────────────────────────────────────
print(f"\n[5/5] MSFT earnings seasonality (fiscal quarters end Jan/Apr/Jul/Oct):")
print(f"  Microsoft fiscal year ends June 30.")
print(f"  Earnings typically reported: Oct (Q1), Jan (Q2), Apr (Q3), Jul (Q4)")
print(f"\n  Average MSFT return in earnings months (all history):")

earnings_months = {
    'Oct (Q1 earnings)': 10,
    'Jan (Q2 earnings)': 1,
    'Apr (Q3 earnings)': 4,
    'Jul (Q4 earnings)': 7,
}
non_earnings = [2, 3, 5, 6, 8, 9, 11, 12]

daily_copy2 = daily.copy()
daily_copy2['YearMonth'] = daily_copy2.index.to_period('M')
monthly2 = daily_copy2.groupby('YearMonth')['Close'].last()
monthly_ret2 = monthly2.pct_change() * 100
monthly_df2 = monthly_ret2.reset_index()
monthly_df2['MonthNum'] = monthly_df2['YearMonth'].apply(lambda x: x.month)
monthly_df2.dropna(inplace=True)

print(f"  {'Month/Type':<28} {'AvgRet%':>9} {'WinRate%':>10}")
print(f"  {sep[:50]}")
for label, m in earnings_months.items():
    d2 = monthly_df2[monthly_df2['MonthNum'] == m]['Close']
    print(f"  {label:<28} {d2.mean():>9.2f} {(d2>0).mean()*100:>10.1f}")

ne_data = monthly_df2[monthly_df2['MonthNum'].isin(non_earnings)]['Close']
print(f"  {'Non-earnings months (avg)':<28} {ne_data.mean():>9.2f} {(ne_data>0).mean()*100:>10.1f}")

print(f"\n{SEP}")
print("AGENT 2 COMPLETE")
print(SEP)
