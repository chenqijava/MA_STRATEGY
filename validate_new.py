"""验证新增币种能否纳入做空腿池子。

口径完全复用 analyze_long: 做空腿 R倍数(盈亏/止损风险), 全样本 + 样本内/外 t 值。
判定: 新币样本外 R 总和 > 0 且不显著拖累 16币整体 t, 才建议纳入。
"""
import os, glob
import numpy as np
import pandas as pd
import ma_pullback as mp
from analyze_long import BASE, path, norm_trades, full_span

OLD = ['BTC','ETH','SOL','BNB','XRP','DOGE','ADA','AVAX','LINK','LTC','DOT','TRX','ATOM','UNI','FIL','APT']
NEW = ['SUI','NEAR','ARB','OP','INJ','TIA']


def short_pnls(sym, start=None, end=None):
    pt = path(sym)
    if not pt:
        return None
    df = mp.cc.load_data(pt)
    if start is not None:
        df = df[(df.index >= start) & (df.index < end)]
    p = dict(BASE, ENABLE_LONG=False, ENABLE_SHORT=True)
    return norm_trades(df, p)


def agg(pnls):
    arr = np.array(pnls, float)
    n = len(arr)
    if n < 2 or arr.std(ddof=1) == 0:
        return dict(n=n, sum=arr.sum() if n else 0, mean=arr.mean() if n else 0,
                    win=(arr > 0).mean()*100 if n else 0, t=0)
    return dict(n=n, sum=arr.sum(), mean=arr.mean(), win=(arr > 0).mean()*100,
                t=arr.mean()/(arr.std(ddof=1)/np.sqrt(n)))


def span_of(sym):
    pt = path(sym)
    df = mp.cc.load_data(pt)
    return df.index[0], df.index[-1]


def main():
    pd.set_option('display.width', 200)
    pd.set_option('display.unicode.east_asian_width', True)
    lo, mid, hi = full_span()
    hi_end = hi + pd.Timedelta(hours=1)
    print('='*90)
    print(f'新增币做空腿验证 (4h, CROSS_DELAY=6, TOUCH=0.5ATR, ATR TP4/SL2)  切分点={mid.date()}')
    print('='*90)

    rows = []
    for s in NEW:
        s0, s1 = span_of(s)
        full = agg(short_pnls(s))
        oos = agg(short_pnls(s, mid, hi_end))
        rows.append(dict(币=s, 数据起=str(s0.date()), 全样本笔=full['n'],
                         全样本R=round(full['sum'], 2), 全胜率=round(full['win'], 1),
                         样本外笔=oos['n'], 样本外R=round(oos['sum'], 2),
                         样本外t=round(oos['t'], 2),
                         判定=('纳入' if oos['sum'] > 0 and full['sum'] > 0 else '不纳入')))
    print('\n### 逐币(新增)')
    print(pd.DataFrame(rows).to_string(index=False))

    # 整体 t: 16币 vs 16+通过的新币
    def pool_pnls(syms, start, end):
        out = []
        for s in syms:
            p = short_pnls(s, start, end)
            if p:
                out += p
        return out

    keep = [r['币'] for r in rows if r['判定'] == '纳入']
    print(f'\n通过样本外筛选的新币: {keep or "无"}')
    for tag, start, end in [('全样本', None, None), ('样本外', mid, hi_end)]:
        base16 = agg(pool_pnls(OLD, start, end))
        combo = agg(pool_pnls(OLD + keep, start, end))
        print(f'\n[{tag}] 原16币:  笔={base16["n"]}  R={base16["sum"]:.2f}  '
              f'胜率={base16["win"]:.1f}%  t={base16["t"]:.2f}')
        print(f'[{tag}] +新币:   笔={combo["n"]}  R={combo["sum"]:.2f}  '
              f'胜率={combo["win"]:.1f}%  t={combo["t"]:.2f}  '
              f'(t {"↑升" if combo["t"] >= base16["t"] else "↓降"})')


if __name__ == '__main__':
    main()
