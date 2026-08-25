"""双腿回测: 做空(MA5/90 死叉反弹) + 做多(MA fast/slow 金叉) 2017-2019"""
import os, sys, argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--sym', default='BTC')
ap.add_argument('--l_fast', type=int, default=10)
ap.add_argument('--l_slow', type=int, default=120)
ap.add_argument('--l_atr_n', type=int, default=20)
ap.add_argument('--l_trail', type=float, default=3.0)
ap.add_argument('--l_src', default='close', choices=['close', 'high'])
args = ap.parse_args()

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_cache',
                              f'{args.sym}-USDT-SWAP_4h_20170101_20200101.csv'))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# === 参数 ===
SLIP, FEE = 0.0005, 0.0005

# 做空参数 (跟主策略一致)
S_FAST, S_SLOW, S_DELAY, S_TOUCH, S_ATR_TP, S_ATR_SL, S_MAX_BARS = 5, 90, 6, 0.5, 10, 10, 40
# 做多参数
L_FAST, L_SLOW, L_TRAIL_ATR, L_ATR_N, L_SRC = args.l_fast, args.l_slow, args.l_trail, args.l_atr_n, args.l_src

# === 指标 ===
df['s_ma_f'] = df['close'].rolling(S_FAST).mean()
df['s_ma_s'] = df['close'].rolling(S_SLOW).mean()
df['l_ma_f'] = df[L_SRC].rolling(L_FAST).mean()
df['l_ma_s'] = df[L_SRC].rolling(L_SLOW).mean()

tr = np.maximum(df['high'] - df['low'],
                np.maximum((df['high'] - df['close'].shift()).abs(),
                           (df['low'] - df['close'].shift()).abs()))
df['atr14'] = tr.rolling(14).mean()
df['atr20'] = tr.rolling(20).mean()
l_atr_col = f'atr{L_ATR_N}'
df[l_atr_col] = tr.rolling(L_ATR_N).mean()

# 交叉信号
df['death_cross'] = (df['s_ma_f'] < df['s_ma_s']) & (df['s_ma_f'].shift(1) >= df['s_ma_s'].shift(1))
df['golden_cross'] = (df['l_ma_f'] > df['l_ma_s']) & (df['l_ma_f'].shift(1) <= df['l_ma_s'].shift(1))

# === 回测 ===
capital = 10000
equity = 10000
curve = []
trades = []

short_pos = None  # {entry, bars, atr_entry}
long_pos = None   # {entry, bars, ext, atr_entry}

# 窗口计数
bars_since_d = -1
bars_since_g = -1
short_gc_active = False  # 本死叉窗口已入场
long_gc_active = False   # 本金叉窗口已入场

n = len(df)
warm = max(S_SLOW, L_SLOW, L_ATR_N) + 2

for i in range(warm, n):
    c, h, l, o = df['close'].iloc[i], df['high'].iloc[i], df['low'].iloc[i], df['open'].iloc[i]
    s_f = df['s_ma_f'].iloc[i]
    s_s = df['s_ma_s'].iloc[i]
    l_f = df['l_ma_f'].iloc[i]
    l_s = df['l_ma_s'].iloc[i]
    atr14 = df['atr14'].iloc[i]
    l_atr = df[l_atr_col].iloc[i]

    if not np.isfinite(c) or not np.isfinite(s_f) or not np.isfinite(s_s):
        continue

    # --- 窗口更新 ---
    if s_f <= s_s:
        bars_since_g = -1
    if s_f >= s_s:
        bars_since_d = -1
    if df['death_cross'].iloc[i]:
        bars_since_d = 0
        short_gc_active = False
    elif bars_since_d >= 0:
        bars_since_d += 1
    if df['golden_cross'].iloc[i]:
        bars_since_g = 0
        long_gc_active = False
    elif bars_since_g >= 0:
        bars_since_g += 1

    # --- 做空出场 ---
    if short_pos is not None:
        short_pos['bars'] += 1
        # ATR 止盈 (跌到 entry - 10·ATR)
        if np.isfinite(l) and np.isfinite(short_pos['atr_entry']) and short_pos['atr_entry'] > 0:
            tp = short_pos['entry'] - S_ATR_TP * short_pos['atr_entry']
            sl = short_pos['entry'] + S_ATR_SL * short_pos['atr_entry']
            if l <= tp:
                pnl = (short_pos['entry'] - tp) / short_pos['entry'] - FEE
                equity *= (1 + pnl); trades.append(('short', pnl)); curve.append((df.index[i], equity))
                short_pos = None; continue
            if h >= sl:
                pnl = (short_pos['entry'] - sl) / short_pos['entry'] - FEE
                equity *= (1 + pnl); trades.append(('short', pnl)); curve.append((df.index[i], equity))
                short_pos = None; continue
        # MA 出场
        if np.isfinite(c) and np.isfinite(s_s) and c > s_s:
            exit_px = c * (1 - SLIP)
            pnl = (short_pos['entry'] - exit_px) / short_pos['entry'] - FEE
            equity *= (1 + pnl); trades.append(('short', pnl)); curve.append((df.index[i], equity))
            short_pos = None; continue
        # 时间止损
        if short_pos['bars'] >= S_MAX_BARS:
            exit_px = c * (1 - SLIP)
            pnl = (short_pos['entry'] - exit_px) / short_pos['entry'] - FEE
            equity *= (1 + pnl); trades.append(('short', pnl)); curve.append((df.index[i], equity))
            short_pos = None; continue

    # --- 做多出场 ---
    if long_pos is not None:
        long_pos['bars'] += 1
        if np.isfinite(h):
            long_pos['ext'] = max(long_pos['ext'], h)
        if np.isfinite(l_atr) and l_atr > 0:
            stop = long_pos['ext'] - L_TRAIL_ATR * long_pos['atr_entry']
            if np.isfinite(l) and l <= stop:
                exit_px = stop * (1 - SLIP)
                pnl = (exit_px - long_pos['entry']) / long_pos['entry'] - FEE
                equity *= (1 + pnl); trades.append(('long', pnl)); curve.append((df.index[i], equity))
                long_pos = None; continue

    # --- 做空入场 (死叉反弹) ---
    if short_pos is None and long_pos is None:
        if not short_gc_active and bars_since_d >= S_DELAY and s_f < s_s:
            gap = s_s - h
            touch = (gap >= 0) and (gap <= S_TOUCH * atr14) if np.isfinite(atr14) else False
            bear = c < o
            if touch and bear:
                entry = c * (1 - SLIP)
                equity -= equity * FEE
                short_pos = {'entry': entry, 'bars': 0, 'atr_entry': atr14 if np.isfinite(atr14) else 0}
                short_gc_active = True
                continue

    # --- 做多入场 (金叉) ---
    if short_pos is None and long_pos is None:
        if not long_gc_active and bars_since_g >= 0 and l_f > l_s:
            entry = c * (1 + SLIP)
            equity -= equity * FEE
            long_pos = {'entry': entry, 'bars': 0, 'ext': entry, 'atr_entry': l_atr if np.isfinite(l_atr) else 0}
            long_gc_active = True

# 平未平仓
if short_pos is not None:
    c_last = df['close'].iloc[-1]
    pnl = (short_pos['entry'] - c_last * (1 - SLIP)) / short_pos['entry'] - FEE
    equity *= (1 + pnl); trades.append(('short', pnl)); curve.append((df.index[-1], equity))
if long_pos is not None:
    c_last = df['close'].iloc[-1]
    pnl = (c_last * (1 - SLIP) - long_pos['entry']) / long_pos['entry'] - FEE
    equity *= (1 + pnl); trades.append(('long', pnl)); curve.append((df.index[-1], equity))

# === 指标 ===
trades_arr = np.array([t[1] for t in trades])
long_a = np.array([t[1] for t in trades if t[0] == 'long'])
short_a = np.array([t[1] for t in trades if t[0] == 'short'])

# 基础指标
ann = (equity / capital) ** (365 * 24 * 60 / (4 * 60) / n) - 1
daily = pd.Series({t: v for t, v in curve}).resample('1D').last().ffill().pct_change().dropna()
sr = daily.mean() / daily.std() * np.sqrt(365) if len(daily) > 1 and daily.std() > 0 else 0
eqs = pd.Series({t: v for t, v in curve})
mdd = (eqs / eqs.cummax() - 1).min() * 100
btc_ret = df['close'].iloc[-1] / df['close'].iloc[warm] - 1
calmar = ann / (abs(mdd) / 100) if mdd else 0

# 交易统计
w = trades_arr[trades_arr > 0]; l = trades_arr[trades_arr < 0]
wr = len(w) / len(trades_arr) * 100 if len(trades_arr) else 0
pf = w.sum() / -l.sum() if len(l) else 999
avg_win = w.mean() * 100 if len(w) else 0
avg_loss = l.mean() * 100 if len(l) else 0
largest_win = w.max() * 100 if len(w) else 0
largest_loss = l.min() * 100 if len(l) else 0
wl_ratio = avg_win / abs(avg_loss) if avg_loss else 0

# 连续胜/负
consec_w = consec_l = max_w = max_l = 0
for t in trades_arr:
    if t > 0: consec_w += 1; consec_l = 0; max_w = max(max_w, consec_w)
    else: consec_l += 1; consec_w = 0; max_l = max(max_l, consec_l)

# Alpha / Beta vs BTC
btc_daily = df['close'].resample('1D').last().ffill().pct_change().dropna()
aligned = pd.concat([daily.rename('s'), btc_daily.rename('b')], axis=1).dropna()
if len(aligned) > 30:
    cov = np.cov(aligned['s'].values, aligned['b'].values)
    beta = cov[0, 1] / cov[1, 1]
    alpha = (aligned['s'].mean() - beta * aligned['b'].mean()) * 365 * 100
else:
    beta = alpha = 0

# 按年
years = {}
for i, (t, v) in enumerate(curve):
    y = t.year
    if y not in years: years[y] = {'start': v, 'end': v, 'peak': v}
    years[y]['end'] = v
    years[y]['peak'] = max(years[y]['peak'], v)

# 月度
monthly = pd.Series({t: v for t, v in curve}).resample('1M').last().ffill().pct_change().dropna() * 100
best_month = monthly.max()
worst_month = monthly.min()
pos_months = (monthly > 0).sum()
neg_months = (monthly <= 0).sum()
month_win_rate = pos_months / (pos_months + neg_months) * 100 if (pos_months + neg_months) else 0

# 持仓期 (简化为总天数/笔数)
days_held = (curve[-1][0] - curve[0][0]).days
avg_hold_days = days_held / len(trades) if len(trades) else 0

# 时间占比 (持仓根数 / 总交易根数)
# 粗略估算: 每笔假设平均持仓 ~ 40根 (MAX_BARS), 总持仓根数 ≈ 40 * 笔数
total_bars = n - warm
est_hold_bars = min(40 * len(trades), total_bars)
time_in_market = est_hold_bars / total_bars * 100

# 按腿统计
def leg_stats(arr):
    if len(arr) == 0: return (0, 0, 0, 0, 0, 0, 0)
    w = arr[arr > 0]; l = arr[arr < 0]
    return (len(arr), len(w)/len(arr)*100,
            w.sum()/-l.sum() if len(l) else 999,
            arr.mean()*100, arr.std()*100,
            w.mean()*100 if len(w) else 0,
            l.mean()*100 if len(l) else 0)

long_n, long_wr, long_pf, long_avg, long_std, long_aw, long_al = leg_stats(long_a)
short_n, short_wr, short_pf, short_avg, short_std, short_aw, short_al = leg_stats(short_a)

# 总 R 倍数
r_mult = w.sum() / abs(l.sum()) if len(l) else 999

out = []
out.append('=' * 70)
out.append(f'{args.sym} 双腿策略 详细指标  (做空 MA{S_FAST}/{S_SLOW} + 做多 MA{L_FAST}/{L_SLOW} {L_SRC}/{L_ATR_N}ATR)')
out.append(f'区间: {df.index[warm]} ~ {df.index[-1]}  ({days_held} 天)')
out.append('=' * 70)
out.append('')
out.append('【收益概览】')
out.append(f'  起始权益: {capital:>8,.0f} USDT')
out.append(f'  期末权益: {equity:>8,.0f} USDT')
out.append(f'  总收益率: {((equity/capital-1)*100):>+8.1f}%')
out.append(f'  {args.sym} 同期: {btc_ret:>+8.1%}')
out.append(f'  Alpha:    {alpha:>+8.1f}%  (日频, 年化)')
out.append(f'  Beta:     {beta:>8.2f}')
out.append('')
out.append('【风险指标】')
out.append(f'  年化收益率: {ann*100:>8.1f}%')
out.append(f'  年化波动率: {daily.std()*np.sqrt(365)*100:>8.1f}%')
out.append(f'  夏普比率:   {sr:>8.2f}')
out.append(f'  最大回撤:   {mdd:>8.1f}%')
out.append(f'  Calmar:     {calmar:>8.2f}')
out.append(f'  日胜率:     {month_win_rate:>8.1f}%  ({pos_months}+/{neg_months}- 月)')
out.append(f'  最佳月:     {best_month:>+8.1f}%')
out.append(f'  最差月:     {worst_month:>+8.1f}%')
out.append('')
out.append('【交易统计】')
out.append(f'  总笔数:     {len(trades_arr):>8d}')
out.append(f'  胜率:       {wr:>8.1f}%')
out.append(f'  利润因子:   {pf:>8.2f}')
out.append(f'  盈亏比:     {wl_ratio:>8.2f}')
out.append(f'  单笔均:     {trades_arr.mean()*100:>+8.2f}%')
out.append(f'  单笔中位:   {np.median(trades_arr)*100:>+8.2f}%')
out.append(f'  单笔标准差: {trades_arr.std()*100:>8.2f}%')
out.append(f'  平均盈利:   {avg_win:>+8.2f}%')
out.append(f'  平均亏损:   {avg_loss:>+8.2f}%')
out.append(f'  最大盈利:   {largest_win:>+8.2f}%')
out.append(f'  最大亏损:   {largest_loss:>+8.2f}%')
out.append(f'  最大连赢:   {max_w:>8d} 笔')
out.append(f'  最大连亏:   {max_l:>8d} 笔')
out.append(f'  总盈利:     {w.sum()*100:>+8.1f}%')
out.append(f'  总亏损:     {l.sum()*100:>+8.1f}%')
out.append('')
out.append('【持仓效率】')
out.append(f'  持仓时间占比: {time_in_market:>7.1f}%  (估算)')
out.append(f'  平均持仓:     {avg_hold_days:>7.1f} 天')
out.append(f'  交易频率:     {len(trades)/days_held*365:>7.1f} 笔/年')
out.append('')
out.append('【分腿统计】')
out.append(f'  {"":>10s}{"笔数":>6s}{"胜率":>7s}{"PF":>6s}{"均单笔":>8s}{"标准差":>8s}{"均盈利":>8s}{"均亏损":>8s}')
out.append(f'  {"做多":>10s}{long_n:>6d}{long_wr:>6.1f}%{long_pf:>6.2f}{long_avg:>+8.2f}%{long_std:>8.2f}%{long_aw:>+8.2f}%{long_al:>+8.2f}%')
out.append(f'  {"做空":>10s}{short_n:>6d}{short_wr:>6.1f}%{short_pf:>6.2f}{short_avg:>+8.2f}%{short_std:>8.2f}%{short_aw:>+8.2f}%{short_al:>+8.2f}%')
out.append(f'  {"合计":>10s}{len(trades_arr):>6d}{wr:>6.1f}%{pf:>6.2f}{trades_arr.mean()*100:>+8.2f}%{trades_arr.std()*100:>8.2f}%{avg_win:>+8.2f}%{avg_loss:>+8.2f}%')
out.append('')
out.append('【分年收益】')
out.append(f'  {"年份":>6s}{"起始":>10s}{"期末":>10s}{"收益率":>10s}')
for y in sorted(years.keys()):
    ys = years[y]['start']; ye = years[y]['end']
    if ys > 0:
        out.append(f'  {y:>6d}{ys:>10,.0f}{ye:>10,.0f}{((ye/ys-1)*100):>+9.1f}%')
out.append(f'  {"合计":>6s}{capital:>10,.0f}{equity:>10,.0f}{((equity/capital-1)*100):>+9.1f}%')
out.append('')
out.append('【月度分布】')
out.append(f'  最佳月: {best_month:>+6.1f}%   最差月: {worst_month:>+6.1f}%')
out.append(f'  盈利月占比: {month_win_rate:.0f}%  ({pos_months}/{pos_months+neg_months})')
out.append(f'  月收益率标准差: {monthly.std():.2f}%')
out.append('')
out.append('【t 检验 (信号层显著性)】')
t_val = (trades_arr.mean() / (trades_arr.std(ddof=1) / np.sqrt(len(trades_arr)))) if len(trades_arr) > 1 and trades_arr.std(ddof=1) > 0 else 0
out.append(f'  单笔R t值: {t_val:.2f}  ({"显著" if abs(t_val) > 2 else "不显著"})')
out.append('')

with open(os.path.join(os.path.dirname(__file__), 'btc_dual_result.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('\n'.join(out))