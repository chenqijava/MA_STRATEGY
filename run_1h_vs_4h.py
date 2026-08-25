"""1h vs 4h 稳健性对比: 同一验证配置(做空 delay6 + ATR止盈4/损2), 全样本+样本外, 全币种。

若 edge 在 1h 也成立 → 抓的是真实结构; 若只 4h 成立 → 4h 特有样本效应。
"""
import os, glob
import numpy as np
import pandas as pd
import ma_pullback as mp

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def syms_for(tf):
    hits = glob.glob(os.path.join(DATA, f'*-USDT-SWAP_{tf}_*.csv'))
    return sorted(set(os.path.basename(h).split('-')[0] for h in hits))


def path(s, tf):
    return sorted(glob.glob(os.path.join(DATA, f'{s}-USDT-SWAP_{tf}_*.csv')))[-1]


CFG = dict(mp.DEFAULTS, MA_TYPE='sma', TOUCH_TH=10, SLOPE_FILTER=False, CROSS_DELAY=6,
           ENABLE_LONG=False, ENABLE_SHORT=True, EXIT_MODE='atr', ATR_TP=4, ATR_SL=2)


def full_stats(tf, syms):
    pnls = []
    for s in syms:
        full = mp.cc.load_data(path(s, tf))
        m, tr, cu = mp.evaluate(full, dict(CFG))
        pnls += [t['pnl'] for t in tr]
    a = np.array(pnls, float); n = len(a)
    t = a.mean() / (a.std(ddof=1) / np.sqrt(n)) if n > 1 and a.std(ddof=1) > 0 else 0
    return n, a.sum(), (a > 0).mean() * 100, t


def oos_stats(tf, syms):
    pnls = []
    for s in syms:
        full = mp.cc.load_data(path(s, tf))
        months = pd.date_range(full.index[0].normalize().replace(day=1), full.index[-1], freq='MS')
        i = 0
        while i + 6 < len(months):
            te_s = months[i + 6]; tei = i + 8
            te_e = months[tei] if tei < len(months) else full.index[-1] + pd.Timedelta(hours=1)
            di = mp.calc_indicators(full, dict(CFG)); ds = mp.gen_signals(di, dict(CFG))
            ds = ds[(ds.index >= te_s) & (ds.index < te_e)]
            if len(ds) >= 30:
                tr, eq, cu = mp.run_backtest(ds, dict(CFG)); pnls += [t['pnl'] for t in tr]
            i += 2
    a = np.array(pnls, float); n = len(a)
    t = a.mean() / (a.std(ddof=1) / np.sqrt(n)) if n > 1 and a.std(ddof=1) > 0 else 0
    return n, a.sum(), t


def main():
    rows = []
    for tf in ['1h', '4h']:
        syms = syms_for(tf)
        fn, ftot, fwin, ft = full_stats(tf, syms)
        on, otot, ot = oos_stats(tf, syms)
        rows.append(dict(周期=tf, 币种=len(syms), 全样本交易=fn, 全样本盈亏=round(ftot, 0),
                         全样本胜率=round(fwin, 1), 全样本t=round(ft, 2),
                         样本外交易=on, 样本外盈亏=round(otot, 0), 样本外t=round(ot, 2)))
    r = pd.DataFrame(rows)
    pd.set_option('display.width', 220); pd.set_option('display.unicode.east_asian_width', True)
    print('== 1h vs 4h 稳健性对比 (做空 delay6 + ATR止盈4/损2) ==')
    print(r.to_string(index=False))


if __name__ == '__main__':
    main()
