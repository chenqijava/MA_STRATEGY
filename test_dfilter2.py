"""日线趋势过滤 —— 敞口对齐后的公平对比。

上一轮 test_dfilter.py 里过滤版收益更低, 但它的均持仓只有基线的 ~60%
(4.0 -> 2.4 币), 敞口小自然收益小 —— 那个对比不公平。

这里把过滤版的 TOTAL 放大, 令三段平均敞口与基线大致相等, 再看收益/夏普/回撤。
若敞口对齐后仍然输, 说明不是"仓位小了", 而是过滤掉的信号本身是赚钱的。
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import importlib
import numpy as np
import pandas as pd
import analyze_long as al
import portfolio_sim as ps

SPANS = [('20210101_20230101', '2021-01~2023-01'),
         ('20230101_20250101', '2023-01~2025-01'),
         ('20250101_20260801', '2025-01~2026-08')]
N = int(os.environ.get('N', '20'))
BASE_ENV = dict(MA_SLOW='90', MA_EXIT='1', COOLDOWN_ONCE='1',
                ATR_TP='10', ATR_SL='10', SIZING='notional',
                CROSS_DELAY='6', TOUCH_ATR='0.5', MAX_BARS='40',
                UNIVERSE='universe_live.txt')

CASES = [
    ('基线 1.5x',              dict(D_FILTER='0', TOTAL='1.5')),
    ('日线MA20过滤 1.5x',      dict(D_FILTER='1', D_MA='20', TOTAL='1.5')),
    ('日线MA20过滤 2.3x(敞口对齐)', dict(D_FILTER='1', D_MA='20', TOTAL='2.3')),
    ('日线MA20过滤 2.5x',      dict(D_FILTER='1', D_MA='20', TOTAL='2.5')),
]
_RESET = ('D_FILTER', 'D_MA', 'D_SLOPE', 'TOTAL')


def run(span, env):
    os.environ['SPAN'] = span
    for k in _RESET:
        os.environ.pop(k, None)
    for k, v in {**BASE_ENV, **env}.items():
        os.environ[k] = str(v)
    importlib.reload(al)
    importlib.reload(ps)
    idx, C, H, L, A, S, syms = ps.build_panel()
    curve, trades, avg_held = ps.simulate(N, idx, C, H, L, A, S)
    m = ps.metrics(curve, trades, idx)
    # 平均敞口(占权益): 均持仓币数 × 单币名义占比
    m['expo'] = avg_held * ps.TOTAL_NOTIONAL / N
    m['held'] = avg_held
    return m


def main():
    print('=' * 96)
    print(f'敞口对齐对比  N={N}  MA90  币池=universe_live')
    print('=' * 96)
    store = {}
    for span, lab in SPANS:
        print(f'\n### {lab}')
        print(f'{"配置":30s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"Calmar":>8s}'
              f'{"笔数":>7s}{"均持仓":>7s}{"平均敞口":>10s}')
        print('-' * 96)
        for name, env in CASES:
            m = run(span, env)
            store.setdefault(name, []).append(m)
            print(f'{name:30s}{m["ann"]:+8.1f}%{m["sharpe"]:7.2f}{m["mdd"]:7.1f}%'
                  f'{m["calmar"]:8.2f}{m["trades"]:7d}{m["held"]:7.1f}{m["expo"]:9.2f}x')

    print('\n' + '=' * 96)
    print(f'{"配置":30s}{"三段年化":>28s}{"最差":>9s}{"最差夏普":>10s}{"最大回撤":>10s}{"均敞口":>9s}')
    print('-' * 96)
    for name, _ in CASES:
        rs = store[name]
        anns = [m['ann'] for m in rs]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:30s}{s:>28s}{min(anns):+8.1f}%{min(m["sharpe"] for m in rs):10.2f}'
              f'{max(m["mdd"] for m in rs):9.1f}%{np.mean([m["expo"] for m in rs]):8.2f}x')


if __name__ == '__main__':
    main()
