"""做空腿(死叉反弹做空) 跨币种 Walk-forward 验证。

单币种4h样本太少(前车之鉴), 故对每个币种各跑滚动 walk-forward, 再把所有测试窗汇总,
用总交易的 t 检验判断做空 edge 是否显著、是否稳健(而非集中在某波行情)。

训练窗选参目标: 训练期夏普最高且交易达标; 参数网格小(TOUCH_TH × 斜率过滤)。
出场固定: 空头收盘升破MA24平仓。
"""
import os, glob
import numpy as np
import pandas as pd
import ma_pullback as mp

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
SYMS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']
GRID = [dict(TOUCH_TH=th, SLOPE_FILTER=f) for th in (10, 20, 40) for f in (True, False)]
TRAIN_MONTHS = 6
TEST_MONTHS = 2
MIN_TRADES = 6


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


def walk_symbol(sym):
    full = mp.cc.load_data(path(sym))
    months = pd.date_range(full.index[0].normalize().replace(day=1), full.index[-1], freq='MS')
    oos_trades = []
    fold_rows = []
    i = 0
    while i + TRAIN_MONTHS < len(months):
        tr_s, tr_e = months[i], months[i + TRAIN_MONTHS]
        te_s = tr_e
        te_i = i + TRAIN_MONTHS + TEST_MONTHS
        te_e = months[te_i] if te_i < len(months) else full.index[-1] + pd.Timedelta(hours=1)
        best_p, best_sh = None, -1e9
        for g in GRID:
            p = dict(mp.DEFAULTS, ENABLE_LONG=False, ENABLE_SHORT=True, MA_TYPE='sma', **g)
            m, _ = bt_slice(full, p, tr_s, tr_e)
            if m and m['trades'] >= MIN_TRADES and m['sharpe'] > best_sh:
                best_sh, best_p = m['sharpe'], p
        if best_p is None:
            i += TEST_MONTHS
            continue
        mt, trades = bt_slice(full, best_p, te_s, te_e)
        if mt:
            for t in trades:
                oos_trades.append(t['pnl'])
            fold_rows.append(dict(币=sym, 测试窗=str(te_s.date())[:7], 选TH=best_p['TOUCH_TH'],
                                  斜率=('开' if best_p['SLOPE_FILTER'] else '关'),
                                  测试交易=mt['trades'], 测试收益=round(mt['ret'] * 100, 1),
                                  测试夏普=round(mt['sharpe'], 2)))
        i += TEST_MONTHS
    return oos_trades, fold_rows


def main():
    all_pnls = []
    all_folds = []
    per_sym = []
    for s in SYMS:
        if not path(s):
            continue
        pnls, folds = walk_symbol(s)
        all_pnls += pnls
        all_folds += folds
        if pnls:
            arr = np.array(pnls, float)
            per_sym.append(dict(币种=s, 折数=len(folds), 交易=len(arr),
                                总盈亏=round(arr.sum(), 1),
                                胜率=round((arr > 0).mean() * 100, 1)))
    print('=' * 80)
    print('做空腿 跨币种 Walk-forward (train6/test2月, 空头收盘升破MA24平仓)')
    print('=' * 80)
    pd.set_option('display.width', 220)
    pd.set_option('display.unicode.east_asian_width', True)
    print(pd.DataFrame(per_sym).to_string(index=False))

    pnls = np.array(all_pnls, float)
    n = len(pnls)
    print(f'\n{"-"*80}\n汇总(所有币种所有测试窗的样本外交易):')
    if n > 1 and pnls.std(ddof=1) > 0:
        t = pnls.mean() / (pnls.std(ddof=1) / np.sqrt(n))
        print(f'  总交易={n}  总盈亏={pnls.sum():.1f}  平均单笔={pnls.mean():.2f}  '
              f'胜率={(pnls>0).mean()*100:.1f}%  t={t:.2f}')
        fold_pos = np.mean([f['测试收益'] > 0 for f in all_folds]) * 100
        print(f'  测试窗数={len(all_folds)}  正收益测试窗占比={fold_pos:.0f}%')
        if t > 1.96 and pnls.sum() > 0:
            v = '显著为正 → 滚动选参下做空edge成立, 值得认真对待'
        elif t > 1.0 and pnls.sum() > 0:
            v = '有苗头但仍不显著 → 需更多样本(加币种)才能定论'
        else:
            v = '不显著/为负 → 之前的+2043是集中在个别行情的运气, 非稳健edge'
        print(f'  判定: {v}')


if __name__ == '__main__':
    main()
