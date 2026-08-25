"""跨币种 Walk-forward: 对比 CROSS_DELAY=3 vs 6, 做空腿 与 双向。

目的: 判断"间隔6根"是真改进还是又一个全样本幻觉。
上一轮基线(delay=3, 做空): 全样本 t=1.07 → walk-forward t=0.24 (崩)。
若 delay=6 的 walk-forward t 明显抬起, 说明"等趋势站稳"是真改进。

滚动: train6/test2月; 训练窗选夏普最高且交易达标的参数(TOUCH_TH×斜率); 出场破MA24。
"""
import os, glob
import numpy as np
import pandas as pd
import ma_pullback as mp

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
SYMS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']
GRID = [dict(TOUCH_TH=th, SLOPE_FILTER=f) for th in (10, 20, 40) for f in (True, False)]
TRAIN_MONTHS, TEST_MONTHS, MIN_TRADES = 6, 2, 6


def path(s):
    h = sorted(glob.glob(os.path.join(DATA, f'{s}-USDT-SWAP_4h_*.csv')))
    return h[-1] if h else None


def bt_slice(full, p, start, end):
    di = mp.calc_indicators(full, p)
    ds = mp.gen_signals(di, p)
    ds = ds[(ds.index >= start) & (ds.index < end)]
    if len(ds) < 30:
        return None, []
    trades, equity, curve = mp.run_backtest(ds, p)
    days = (ds.index[-1] - ds.index[0]).days or 1
    return mp.cc.metrics(trades, curve, ds.index, days), trades


def walk(full, delay, mode):
    months = pd.date_range(full.index[0].normalize().replace(day=1), full.index[-1], freq='MS')
    pnls, folds = [], []
    i = 0
    while i + TRAIN_MONTHS < len(months):
        tr_s, tr_e = months[i], months[i + TRAIN_MONTHS]
        te_s = tr_e
        te_i = i + TRAIN_MONTHS + TEST_MONTHS
        te_e = months[te_i] if te_i < len(months) else full.index[-1] + pd.Timedelta(hours=1)
        best_p, best_sh = None, -1e9
        for g in GRID:
            p = dict(mp.DEFAULTS, MA_TYPE='sma', CROSS_DELAY=delay,
                     ENABLE_LONG=(mode != 'short'), ENABLE_SHORT=(mode != 'long'), **g)
            m, _ = bt_slice(full, p, tr_s, tr_e)
            if m and m['trades'] >= MIN_TRADES and m['sharpe'] > best_sh:
                best_sh, best_p = m['sharpe'], p
        if best_p is None:
            i += TEST_MONTHS
            continue
        mt, trades = bt_slice(full, best_p, te_s, te_e)
        if mt:
            pnls += [t['pnl'] for t in trades]
            folds.append(mt['ret'])
        i += TEST_MONTHS
    return pnls, folds


def summ(name, all_pnls, all_folds):
    arr = np.array(all_pnls, float)
    n = len(arr)
    if n < 2 or arr.std(ddof=1) == 0:
        print(f'  {name}: 样本不足')
        return
    t = arr.mean() / (arr.std(ddof=1) / np.sqrt(n))
    fold_pos = np.mean([f > 0 for f in all_folds]) * 100
    print(f'  {name:12s} 交易={n:4d}  总盈亏={arr.sum():8.1f}  平均单笔={arr.mean():6.2f}  '
          f'胜率={(arr>0).mean()*100:5.1f}%  正窗={fold_pos:3.0f}%  t={t:5.2f}')
    return t


def main():
    fulls = {s: mp.cc.load_data(path(s)) for s in SYMS if path(s)}
    print('=' * 92)
    print('跨币种 Walk-forward 对比: CROSS_DELAY 3 vs 6 (train6/test2月, 出场破MA24)')
    print('=' * 92)
    for mode in ['short', 'both', 'long']:
        label = {'short': '做空腿', 'both': '双向', 'long': '做多腿'}[mode]
        print(f'\n[{label}]')
        for delay in [3, 6]:
            pnls, folds = [], []
            for s, full in fulls.items():
                p2, f2 = walk(full, delay, mode)
                pnls += p2; folds += f2
            summ(f'delay={delay}', pnls, folds)
    print('\n' + '-' * 92)
    print('判读: 上一轮 做空delay=3 walk-forward t=0.24。若 delay=6 的 t 明显更高且>1.96 → 真改进;')
    print('      若仍在 1 附近或更低 → "间隔6根"也是全样本幻觉, 扛不住滚动选参。')


if __name__ == '__main__':
    main()
