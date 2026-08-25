"""决定性验证: 做空 delay=6 + ATR出场, 跨16币种 walk-forward。

全样本已显示 t=2.76(显著为正), 但全样本会被后见之明污染(前八轮教训)。
walk-forward 滚动选参(train6/test2月), 训练窗搜 ATR止盈/止损倍数, 测试窗盲测。
若拼接样本外 t 仍显著为正 → 出场调整是真突破, 站得住的策略。
"""
import os, glob
import numpy as np
import pandas as pd
import ma_pullback as mp

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
SYMS = sorted(set(os.path.basename(h).split('-')[0]
                  for h in glob.glob(os.path.join(DATA, '*-USDT-SWAP_4h_*.csv'))))
# 训练窗搜索的出场参数网格(含破MA24作对照, 让训练自己选)
GRID = [
    dict(EXIT_MODE='atr', ATR_TP=4, ATR_SL=2),
    dict(EXIT_MODE='atr', ATR_TP=6, ATR_SL=2),
    dict(EXIT_MODE='atr', ATR_TP=3, ATR_SL=2),
    dict(EXIT_MODE='atr_trail', TRAIL_TRIG=3, TRAIL_W=3, ATR_SL=2),
    dict(EXIT_MODE='ma_slow'),
]
BASE = dict(mp.DEFAULTS, MA_TYPE='sma', TOUCH_TH=10, SLOPE_FILTER=False,
            CROSS_DELAY=6, ENABLE_LONG=False, ENABLE_SHORT=True)
TRAIN_MONTHS, TEST_MONTHS, MIN_TRADES = 6, 2, 6


def path(s):
    return sorted(glob.glob(os.path.join(DATA, f'{s}-USDT-SWAP_4h_*.csv')))[-1]


def bt(full, p, start, end):
    di = mp.calc_indicators(full, p)
    ds = mp.gen_signals(di, p)
    ds = ds[(ds.index >= start) & (ds.index < end)]
    if len(ds) < 30:
        return None, []
    trades, eq, curve = mp.run_backtest(ds, p)
    days = (ds.index[-1] - ds.index[0]).days or 1
    return mp.cc.metrics(trades, curve, ds.index, days), trades


def walk(full):
    months = pd.date_range(full.index[0].normalize().replace(day=1), full.index[-1], freq='MS')
    pnls, folds, picks = [], [], []
    i = 0
    while i + TRAIN_MONTHS < len(months):
        tr_s, tr_e = months[i], months[i + TRAIN_MONTHS]
        te_s = tr_e
        te_i = i + TRAIN_MONTHS + TEST_MONTHS
        te_e = months[te_i] if te_i < len(months) else full.index[-1] + pd.Timedelta(hours=1)
        best_p, best_sh, best_g = None, -1e9, None
        for g in GRID:
            p = dict(BASE, **g)
            m, _ = bt(full, p, tr_s, tr_e)
            if m and m['trades'] >= MIN_TRADES and m['sharpe'] > best_sh:
                best_sh, best_p, best_g = m['sharpe'], p, g
        if best_p is None:
            i += TEST_MONTHS
            continue
        mt, trades = bt(full, best_p, te_s, te_e)
        if mt:
            pnls += [t['pnl'] for t in trades]
            folds.append(mt['ret'])
            picks.append(best_g.get('EXIT_MODE'))
        i += TEST_MONTHS
    return pnls, folds, picks


def main():
    fulls = {s: mp.cc.load_data(path(s)) for s in SYMS}
    all_pnls, all_folds, all_picks = [], [], []
    per = []
    for s, full in fulls.items():
        pn, fo, pk = walk(full)
        all_pnls += pn; all_folds += fo; all_picks += pk
        if pn:
            arr = np.array(pn, float)
            per.append(dict(币种=s, 折=len(fo), 交易=len(arr),
                            总盈亏=round(arr.sum(), 0), 胜率=round((arr > 0).mean() * 100, 1)))
    pd.set_option('display.width', 200)
    pd.set_option('display.unicode.east_asian_width', True)
    print('=' * 84)
    print('做空 delay=6 + ATR出场 跨16币种 Walk-forward (train6/test2月, 训练窗选出场)')
    print('=' * 84)
    print(pd.DataFrame(per).to_string(index=False))
    arr = np.array(all_pnls, float)
    n = len(arr)
    t = arr.mean() / (arr.std(ddof=1) / np.sqrt(n))
    print(f'\n{"-"*84}')
    print(f'样本外汇总: 币种={len(per)}  总交易={n}  总盈亏={arr.sum():.0f}  '
          f'平均单笔={arr.mean():.2f}  胜率={(arr>0).mean()*100:.1f}%')
    print(f'          正收益测试窗占比={np.mean([f>0 for f in all_folds])*100:.0f}%  t={t:.2f}')
    from collections import Counter
    print(f'          训练窗选中的出场分布: {dict(Counter(all_picks))}')
    if t > 1.96 and arr.sum() > 0:
        v = '显著为正! walk-forward也扛住了 → 出场调整是真突破, 这是站得住的策略'
    elif t > 1.3:
        v = f'接近显著(t={t:.2f}) → 方向可信'
    else:
        v = f'不显著(t={t:.2f}) → 全样本的t=2.76含后见之明成分, 滚动下打折'
    print(f'          判定: {v}')


if __name__ == '__main__':
    main()
