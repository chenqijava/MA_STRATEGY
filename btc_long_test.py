"""标的做多回测: 金叉入场 + N×ATR trail (2017-2019)。支持 src=close/high"""
import os, sys, argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--sym', default='BTC')
ap.add_argument('--fast', type=int, default=10)
ap.add_argument('--slow', type=int, default=120)
ap.add_argument('--atr_n', type=int, default=20)
ap.add_argument('--trail', type=float, default=3.0)
ap.add_argument('--src', default='close', choices=['close', 'high'])
args = ap.parse_args()

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data_cache',
                              f'{args.sym}-USDT-SWAP_4h_20170101_20200101.csv'))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

MA_F, MA_S, TRAIL_ATR, SLIP, FEE = args.fast, args.slow, args.trail, 0.0005, 0.0005
ATR_LEN = args.atr_n
SRC = df[args.src]

df['ma_f'] = SRC.rolling(MA_F).mean()
df['ma_s'] = SRC.rolling(MA_S).mean()
tr = np.maximum(df['high'] - df['low'],
                np.maximum((df['high'] - df['close'].shift()).abs(),
                           (df['low'] - df['close'].shift()).abs()))
df['atr'] = tr.rolling(ATR_LEN).mean()
df['gc'] = (df['ma_f'] > df['ma_s']) & (df['ma_f'].shift(1) <= df['ma_s'].shift(1))

eq, capital = 10000, 10000
curve, trades = [], []
in_pos, gc_active = False, False
entry, ext, atr_entry = 0, 0, 0

for i in range(max(MA_S, ATR_LEN) + 2, len(df)):
    c, h, l, a = df['close'].iloc[i], df['high'].iloc[i], df['low'].iloc[i], df['atr'].iloc[i]
    if not np.isfinite(c):
        continue

    # 出场
    if in_pos:
        if np.isfinite(h):
            ext = max(ext, h)
        if np.isfinite(a):
            stop = ext - TRAIL_ATR * atr_entry
        else:
            stop = 0
        if np.isfinite(l) and l <= stop and np.isfinite(stop):
            pnl = (stop * (1 - SLIP) - entry) / entry - FEE
            eq *= (1 + pnl)
            trades.append(pnl)
            curve.append((df.index[i], eq))
            in_pos = False
            continue

    # 入场
    if not in_pos:
        if df['gc'].iloc[i]:
            gc_active = False
        if gc_active:
            continue
        if df['ma_f'].iloc[i] > df['ma_s'].iloc[i] and np.isfinite(df['ma_f'].iloc[i]):
            entry = c * (1 + SLIP)
            eq -= eq * FEE
            ext = entry
            atr_entry = a if np.isfinite(a) else 0
            in_pos = True
            gc_active = True

if in_pos:
    pnl = (df['close'].iloc[-1] * (1 - SLIP) - entry) / entry - FEE
    eq *= (1 + pnl)
    trades.append(pnl)
    curve.append((df.index[-1], eq))

trades = np.array(trades)
w, l = trades[trades > 0], trades[trades < 0]
wr = len(w) / len(trades) * 100 if len(trades) else 0
pf = w.sum() / -l.sum() if len(l) else 999
ann = (eq / capital) ** (365 * 24 * 60 / (4 * 60) / len(df)) - 1

daily = pd.Series({t: v for t, v in curve}).resample('1D').last().ffill().pct_change().dropna()
sr = daily.mean() / daily.std() * np.sqrt(365) if len(daily) > 1 and daily.std() > 0 else 0
eqs = pd.Series({t: v for t, v in curve})
mdd = (eqs / eqs.cummax() - 1).min() * 100 if len(eqs) > 1 else 0
hold_ret = df['close'].iloc[-1] / df['close'].iloc[max(MA_S, ATR_LEN) + 2] - 1

print(f'=== {args.sym} 做多 MA{MA_F}>{MA_S}({args.src}) + {TRAIL_ATR:.0f}×ATR({ATR_LEN}) trail ===')
print(f'区间: {df.index[max(MA_S, ATR_LEN) + 2]} ~ {df.index[-1]}')
print(f'权益: {capital:.0f} -> {eq:.0f}  ({(eq/capital-1)*100:+.1f}%)')
print(f'{args.sym} 同期: {hold_ret:+.1%}')
print(f'年化: {ann*100:.1f}%')
print(f'夏普: {sr:.2f}')
print(f'最大回撤: {mdd:.1f}%')
print(f'Calmar: {ann / (abs(mdd)/100) if mdd else 0:.2f}')
print(f'笔数: {len(trades)}')
print(f'胜率: {wr:.1f}%')
print(f'PF: {pf:.2f}')
print(f'单笔均: {trades.mean()*100:+.2f}%')