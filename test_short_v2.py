"""验证做空信号新定义 v2, 与当前做空定义对比 (镜像 test_long_v2)。

新定义 (更严格的反弹确认):
  1. 死叉后 CROSS_DELAY 根起, MA5<MA30 期间
  2. 触碰: TOUCH_LO·ATR <= (ma_slow - high) <= TOUCH_HI·ATR  (默认 -0.25~+0.25, 允许插针略破)
  3. 阴线: close < open
  4. 收盘压回慢线下方: close < ma_slow  (新增, 阻力确认; 镜像做多的 close>ma_slow)
出场沿用 ATR止盈4/止损2。口径 R倍数, 与 analyze_long / test_long_v2 可比。

当前定义: 触碰[0, 0.5]·ATR + 阴线 (无 close<ma_slow)。样本外 t=+8.29。
本脚本检验新定义是更好还是更差。
"""
import os, sys, glob
import numpy as np
import pandas as pd
import ma_pullback as mp
from analyze_long import BASE, full_span

if sys.stdout.encoding and not sys.stdout.encoding.lower().startswith('utf'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
SYMS = open(os.path.join(os.path.dirname(__file__), 'universe_liquid.txt'),
            encoding='utf-8').read().strip().split(',')
TOUCH_LO = float(os.environ.get('TOUCH_LO', '-0.25'))
TOUCH_HI = float(os.environ.get('TOUCH_HI', '0.25'))


def gen_short_v2(df, p):
    """新做空: 触碰窗口双侧[LO,HI]·ATR + 阴线 + 收盘<慢线。"""
    df = mp.calc_indicators(df, p).copy()
    ma_fast = df['ma_fast'].values
    ma_slow = df['ma_slow'].values
    dcross = df['death_cross'].values
    high = df['high'].values
    close = df['close'].values
    open_ = df['open'].values
    atr = df['atr'].values
    n = len(df)
    sig = np.zeros(n, dtype=int)
    bars_since_d = -1
    for i in range(n):
        if not (np.isfinite(ma_fast[i]) and np.isfinite(ma_slow[i])):
            continue
        if ma_fast[i] >= ma_slow[i]:
            bars_since_d = -1
        if dcross[i]:
            bars_since_d = 0
        elif bars_since_d >= 0:
            bars_since_d += 1
        if bars_since_d >= p['CROSS_DELAY'] and ma_fast[i] < ma_slow[i] and np.isfinite(atr[i]) and atr[i] > 0:
            gap = ma_slow[i] - high[i]
            touch = (TOUCH_LO * atr[i] <= gap <= TOUCH_HI * atr[i])
            bear = close[i] < open_[i]
            if touch and bear:
                sig[i] = -1
    df['entry_sig'] = sig
    return df


def r_trades(df, p):
    ds = df.iloc[p['MA_SLOW'] + 2:]
    if len(ds) < 40:
        return []
    trades, _, _ = mp.run_backtest(ds, p)
    atr_at = ds['atr']
    out = []
    for t in trades:
        a = atr_at.get(t['entry_time'])
        if a is None or not np.isfinite(a) or a <= 0:
            continue
        sign = 1 if t['type'] == 'LONG' else -1
        out.append(sign * (t['exit_price'] - t['entry_price']) / (p['ATR_SL'] * a))
    return out


def path(s):
    h = sorted(glob.glob(os.path.join(DATA, f'{s}-USDT-SWAP_4h_*.csv')))
    return h[-1] if h else None


def agg(pnls):
    arr = np.array(pnls or [], float)
    n = len(arr)
    if n < 2 or arr.std(ddof=1) == 0:
        return n, 0.0, 0.0, 0.0
    return n, float(arr.sum()), float(arr.mean()/(arr.std(ddof=1)/np.sqrt(n))), float((arr > 0).mean()*100)


def run(which, start=None, end=None):
    p = dict(BASE, ENABLE_LONG=False, ENABLE_SHORT=True, MA_SLOW=30,
             EXIT_MODE='atr', ATR_TP=4.0, ATR_SL=2.0, TOUCH_MODE='atr', TOUCH_ATR=0.5)
    allr = []
    for s in SYMS:
        pt = path(s)
        if not pt:
            continue
        df = mp.cc.load_data(pt)
        if start is not None:
            df = df[(df.index >= start) & (df.index < end)]
        if len(df) < 60:
            continue
        if which == 'v2':
            ds = gen_short_v2(df, p)
        else:
            ds = mp.gen_signals(mp.calc_indicators(df, p), p)
        allr += r_trades(ds, p)
    return allr


def main():
    lo, mid, hi = full_span()
    hi_end = hi + pd.Timedelta(hours=1)
    print('=' * 76)
    print(f'做空信号新定义 v2 验证 ({len(SYMS)}币, MA_SLOW=30)')
    print(f'  触碰窗口 [{TOUCH_LO}, {TOUCH_HI}]·ATR + 阴线 (无收盘<慢线)')
    print(f'  对比当前 [0, 0.5]·ATR + 阴线 (无收盘<慢线)   切分={mid.date()}')
    print('=' * 76)
    for name, which in [('当前做空', 'cur'), ('新做空v2', 'v2')]:
        nf, rf, tf, wf = agg(run(which))
        ni, ri, ti, wi = agg(run(which, lo, mid))
        no, ro, to, wo = agg(run(which, mid, hi_end))
        print(f'\n### {name}')
        print(f'  全样本: 笔={nf:5d}  R={rf:8.1f}  t={tf:6.2f}  胜率={wf:.1f}%')
        print(f'  样本内: 笔={ni:5d}  R={ri:8.1f}  t={ti:6.2f}  胜率={wi:.1f}%')
        print(f'  样本外: 笔={no:5d}  R={ro:8.1f}  t={to:6.2f}  胜率={wo:.1f}%')


if __name__ == '__main__':
    main()
