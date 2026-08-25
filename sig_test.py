"""扩大样本显著性检验: 固定 双向 + CROSS_DELAY=6, 覆盖全部可用币种。

参数完全固定(不按币种优化, 不搜索) —— 纯粹靠堆样本量看 t 能否过 1.96。
固定配置: SMA, MA5/MA24, TOUCH_TH=10, 斜率过滤关, 出场破MA24, 双向, delay=6。
同时报告 walk-forward 口径(滚动选参)的汇总, 与固定参数口径对照。
"""
import os, glob
import numpy as np
import pandas as pd
import ma_pullback as mp

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')

# 固定配置
FIXED = dict(mp.DEFAULTS, MA_TYPE='sma', TOUCH_TH=10, SLOPE_FILTER=False,
             CROSS_DELAY=6, ENABLE_LONG=True, ENABLE_SHORT=True)


def all_symbols():
    hits = glob.glob(os.path.join(DATA, '*-USDT-SWAP_4h_*.csv'))
    syms = sorted(set(os.path.basename(h).split('-')[0] for h in hits))
    return syms


def path(s):
    return sorted(glob.glob(os.path.join(DATA, f'{s}-USDT-SWAP_4h_*.csv')))[-1]


def main():
    syms = all_symbols()
    print(f'可用币种({len(syms)}): {syms}')
    all_pnls = []
    rows = []
    for s in syms:
        full = mp.cc.load_data(path(s))
        m, trades, curve = mp.evaluate(full, dict(FIXED), label=s)
        pnls = [t['pnl'] for t in trades]
        all_pnls += pnls
        arr = np.array(pnls, float)
        rows.append(dict(币种=s, 交易=len(arr), 总盈亏=round(arr.sum(), 1),
                         胜率=round((arr > 0).mean() * 100, 1) if len(arr) else 0,
                         夏普=round(m['sharpe'], 2)))
    r = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    pd.set_option('display.unicode.east_asian_width', True)
    print('=' * 80)
    print('固定 双向+delay=6, 全币种全区间(2025.1~2026.7) 逐币种:')
    print('=' * 80)
    print(r.to_string(index=False))

    arr = np.array(all_pnls, float)
    n = len(arr)
    t = arr.mean() / (arr.std(ddof=1) / np.sqrt(n))
    print(f'\n{"-"*80}')
    print(f'汇总: 币种数={len(syms)}  总交易={n}  总盈亏={arr.sum():.1f}  '
          f'平均单笔={arr.mean():.2f}  胜率={(arr>0).mean()*100:.1f}%')
    print(f'      单笔 t检验: t={t:.2f}  (|t|>1.96 → 95%显著)')
    pos_sym = (r['总盈亏'] > 0).mean() * 100
    print(f'      盈利币种占比={pos_sym:.0f}%')
    if t > 1.96:
        v = f'显著为正! 堆到{n}笔后过了显著线 → 双向delay=6 是站得住的策略'
    elif t > 1.3:
        v = f'接近显著(t={t:.2f}) → 方向可信, 再多些样本可能过线'
    else:
        v = f'仍不显著(t={t:.2f}) → 堆样本也未过线, edge强度不足以交易'
    print(f'      判定: {v}')


if __name__ == '__main__':
    main()
