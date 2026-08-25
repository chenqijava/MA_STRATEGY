"""慢线 MA_SLOW 扫描 (20~60, step=1), 61币做空腿。

口径复用 analyze_long: R倍数(盈亏/止损风险)跨币汇总, 全样本+样本外 t。
其余参数固定为已定配置 (MA_FAST=5, CROSS_DELAY=6, TOUCH=0.5ATR, ATR TP4/SL2)。
找慢线的稳健区间, 而非单点最优 (避免过拟合)。
"""
import os, sys
import numpy as np
import pandas as pd
import ma_pullback as mp
from analyze_long import BASE, path, norm_trades, full_span

if sys.stdout.encoding and not sys.stdout.encoding.lower().startswith('utf'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SYMS = open(os.path.join(os.path.dirname(__file__), 'universe_pass.txt'),
            encoding='utf-8').read().strip().split(',')


def agg(pnls):
    arr = np.array(pnls or [], float)
    n = len(arr)
    if n < 2 or arr.std(ddof=1) == 0:
        return 0, 0.0, 0.0
    return n, float(arr.sum()), float(arr.mean() / (arr.std(ddof=1) / np.sqrt(n)))


def pool(ma_slow, start=None, end=None):
    p = dict(BASE, ENABLE_LONG=False, ENABLE_SHORT=True, MA_SLOW=ma_slow)
    out = []
    for s in SYMS:
        pt = path(s)
        if not pt:
            continue
        df = mp.cc.load_data(pt)
        if start is not None:
            df = df[(df.index >= start) & (df.index < end)]
        r = norm_trades(df, p)
        if r:
            out += r
    return out


def main():
    pd.set_option('display.width', 200)
    lo, mid, hi = full_span()
    hi_end = hi + pd.Timedelta(hours=1)
    print('=' * 78)
    print(f'慢线扫描 MA_SLOW 20~60 (step1), {len(SYMS)}币做空腿, MA_FAST=5 固定')
    print(f'切分点={mid.date()}')
    print('=' * 78)
    rows = []
    for ms in range(20, 61):
        nf, rf, tf = agg(pool(ms))
        no, ro, to = agg(pool(ms, mid, hi_end))
        rows.append(dict(MA慢=ms, 全笔=nf, 全R=round(rf, 1), 全t=round(tf, 2),
                         外笔=no, 外R=round(ro, 1), 外t=round(to, 2)))
        print(f'\rMA_SLOW={ms} 完成', end='')
    print()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print('\n--- 按样本外 t 排序 top10 ---')
    print(df.sort_values('外t', ascending=False).head(10).to_string(index=False))
    print(f'\n当前配置 MA_SLOW=24:')
    print(df[df['MA慢'] == 24].to_string(index=False))


if __name__ == '__main__':
    main()
