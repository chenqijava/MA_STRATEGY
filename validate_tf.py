"""在指定周期(12h/1d 等)上验证做空腿, 与 4h 对比。

复用 analyze_long 的 R倍数做空口径。对每个周期跑 MA_SLOW=24 与 30 两档,
看做空 edge 在更高周期上是否依然成立(t 是否仍显著为正)。

用法: python validate_tf.py 12h 1d    (默认 4h 12h 1d)
"""
import os, sys, glob
import numpy as np
import pandas as pd
import ma_pullback as mp
from analyze_long import BASE, norm_trades

if sys.stdout.encoding and not sys.stdout.encoding.lower().startswith('utf'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
SYMS = open(os.path.join(os.path.dirname(__file__), 'universe_pass.txt'),
            encoding='utf-8').read().strip().split(',')


def path_tf(s, tf):
    h = sorted(glob.glob(os.path.join(DATA, f'{s}-USDT-SWAP_{tf}_*.csv')))
    return h[-1] if h else None


def span(tf):
    lo = hi = None
    for s in SYMS:
        p = path_tf(s, tf)
        if not p:
            continue
        df = mp.cc.load_data(p)
        if len(df) == 0:
            continue
        lo = df.index[0] if lo is None else min(lo, df.index[0])
        hi = df.index[-1] if hi is None else max(hi, df.index[-1])
    return lo, (lo + (hi - lo) / 2) if lo is not None else None, hi


def agg(pnls):
    arr = np.array(pnls or [], float)
    n = len(arr)
    if n < 2 or arr.std(ddof=1) == 0:
        return n, 0.0, 0.0, 0.0
    return n, float(arr.sum()), float(arr.mean() / (arr.std(ddof=1) / np.sqrt(n))), float((arr > 0).mean()*100)


def pool(tf, ma_slow, start=None, end=None):
    p = dict(BASE, ENABLE_LONG=False, ENABLE_SHORT=True, MA_SLOW=ma_slow)
    out, ncoin = [], 0
    for s in SYMS:
        pt = path_tf(s, tf)
        if not pt:
            continue
        df = mp.cc.load_data(pt)
        if start is not None:
            df = df[(df.index >= start) & (df.index < end)]
        r = norm_trades(df, p)
        if r:
            out += r
            ncoin += 1
    return out, ncoin


def main():
    tfs = sys.argv[1:] or ['4h', '12h', '1d']
    pd.set_option('display.width', 200)
    print('=' * 84)
    print(f'多周期做空腿验证 ({len(SYMS)}币, MA_FAST=5, CROSS_DELAY=6, TOUCH=0.5ATR, ATR TP4/SL2)')
    print('=' * 84)
    rows = []
    for tf in tfs:
        lo, mid, hi = span(tf)
        if lo is None:
            print(f'[{tf}] 无数据, 跳过')
            continue
        hi_end = hi + pd.Timedelta(hours=1)
        for ms in (24, 30):
            pnl_f, nc = pool(tf, ms)
            nf, rf, tf_full, wf = agg(pnl_f)
            pnl_o, _ = pool(tf, ms, mid, hi_end)
            no, ro, to, wo = agg(pnl_o)
            rows.append(dict(周期=tf, MA慢=ms, 覆盖币=nc, 数据起=str(lo.date()),
                             全笔=nf, 全R=round(rf, 1), 全t=round(tf_full, 2), 全胜=round(wf, 1),
                             外笔=no, 外R=round(ro, 1), 外t=round(to, 2)))
    print(pd.DataFrame(rows).to_string(index=False))
    print('\n注: 周期越高, 单币交易越少, t值天然偏弱; 看方向(R>0)与显著性是否维持。')


if __name__ == '__main__':
    main()
