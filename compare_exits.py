"""出场方式横评: 在当前配置(50币/MA30/TOUCH0.5/做空)下对比各种出场。

对比:
  A. ma_slow  收盘升破MA30平仓
  B. ma_fast  收盘升破MA5平仓
  C. atr      固定ATR止盈/止损 (扫多组 TP/SL)
  D. atr_trail 移动止损 (扫 TRAIL_TRIG/TRAIL_W)
  各自叠加 MAX_BARS=40 时间止损做对照。
口径: R倍数(盈亏/止损风险, 跨币可比) 全样本 + 样本外 t。
注: ma_slow/ma_fast 无固定"止损风险", R用 2·ATR 归一以便与atr模式可比。
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


def path(s):
    h = sorted(glob.glob(os.path.join(DATA, f'{s}-USDT-SWAP_4h_*.csv')))
    return h[-1] if h else None


# 预载信号(信号与出场无关, 只需算一次); 出场在 run_backtest 里按 p 变化
_CACHE = {}
def signals(s):
    if s not in _CACHE:
        pbase = dict(BASE, ENABLE_LONG=False, ENABLE_SHORT=True, MA_SLOW=30,
                     TOUCH_MODE='atr', TOUCH_ATR=0.5)
        ds = mp.gen_signals(mp.calc_indicators(mp.cc.load_data(path(s)), pbase), pbase)
        _CACHE[s] = ds.iloc[pbase['MA_SLOW'] + 2:]
    return _CACHE[s]


def r_trades(ds, p):
    if len(ds) < 40:
        return []
    trades, _, _ = mp.run_backtest(ds, p)
    atr_at = ds['atr']
    sl = p.get('ATR_SL', 2.0) or 2.0
    out = []
    for t in trades:
        a = atr_at.get(t['entry_time'])
        if a is None or not np.isfinite(a) or a <= 0:
            continue
        sign = 1 if t['type'] == 'LONG' else -1
        out.append(sign * (t['exit_price'] - t['entry_price']) / (sl * a))
    return out


def agg(pnls):
    arr = np.array(pnls or [], float)
    n = len(arr)
    if n < 2 or arr.std(ddof=1) == 0:
        return n, 0.0, 0.0, 0.0
    return n, float(arr.sum()), float(arr.mean()/(arr.std(ddof=1)/np.sqrt(n))), float((arr > 0).mean()*100)


def run(p, mid=None, hi_end=None):
    allf, allo = [], []
    for s in SYMS:
        if not path(s):
            continue
        ds = signals(s)
        allf += r_trades(ds, p)
        if mid is not None:
            allo += r_trades(ds[ds.index >= mid], p)
    return agg(allf), agg(allo)


def main():
    lo, mid, hi = full_span()
    he = hi + pd.Timedelta(hours=1)
    base = dict(BASE, ENABLE_LONG=False, ENABLE_SHORT=True, MA_SLOW=30,
                TOUCH_MODE='atr', TOUCH_ATR=0.5, ATR_LEN=14)
    configs = [
        ('ma_slow(破MA30)',        dict(base, EXIT_MODE='ma_slow', MAX_BARS=0)),
        ('ma_fast(破MA5)',         dict(base, EXIT_MODE='ma_fast', MAX_BARS=0)),
        ('atr TP3/SL2',            dict(base, EXIT_MODE='atr', ATR_TP=3.0, ATR_SL=2.0, MAX_BARS=0)),
        ('atr TP4/SL2 (当前无时停)', dict(base, EXIT_MODE='atr', ATR_TP=4.0, ATR_SL=2.0, MAX_BARS=0)),
        ('atr TP4/SL2 +时停40 ★当前', dict(base, EXIT_MODE='atr', ATR_TP=4.0, ATR_SL=2.0, MAX_BARS=40)),
        ('atr TP5/SL2',            dict(base, EXIT_MODE='atr', ATR_TP=5.0, ATR_SL=2.0, MAX_BARS=0)),
        ('atr TP6/SL3',            dict(base, EXIT_MODE='atr', ATR_TP=6.0, ATR_SL=3.0, MAX_BARS=0)),
        ('atr TP4/SL1.5',          dict(base, EXIT_MODE='atr', ATR_TP=4.0, ATR_SL=1.5, MAX_BARS=0)),
        ('atr TP4/SL3',            dict(base, EXIT_MODE='atr', ATR_TP=4.0, ATR_SL=3.0, MAX_BARS=0)),
        ('atr_trail T3/W3',        dict(base, EXIT_MODE='atr_trail', ATR_SL=2.0, TRAIL_TRIG=3.0, TRAIL_W=3.0, MAX_BARS=0)),
        ('atr_trail T2/W2',        dict(base, EXIT_MODE='atr_trail', ATR_SL=2.0, TRAIL_TRIG=2.0, TRAIL_W=2.0, MAX_BARS=0)),
        ('atr_trail T3/W3 +时停40', dict(base, EXIT_MODE='atr_trail', ATR_SL=2.0, TRAIL_TRIG=3.0, TRAIL_W=3.0, MAX_BARS=40)),
    ]
    print('=' * 92)
    print(f'出场方式横评 (做空, {len(SYMS)}币, MA_SLOW=30, TOUCH=0.5ATR)  切分={mid.date()}')
    print('R = 每笔盈亏 / (2·ATR), 跨币可比。')
    print('=' * 92)
    print(f'{"出场方式":<26} {"全笔":>5} {"全R":>7} {"全t":>6} {"外R":>7} {"外t":>6} {"胜率":>5}')
    print('-' * 92)
    rows = []
    for name, p in configs:
        (nf, rf, tf, wf), (no, ro, to, wo) = run(p, mid, he)
        rows.append((name, tf, to, rf))
        print(f'{name:<26} {nf:>5} {rf:>7.1f} {tf:>6.2f} {ro:>7.1f} {to:>6.2f} {wf:>4.1f}%')


if __name__ == '__main__':
    main()
