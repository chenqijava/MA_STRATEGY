"""日线过滤的最后一问: 被过滤掉的那批信号本身赚不赚钱?

把信号切成两半, 各自独立跑组合(同 N 同敞口):
  A = 日线MA20向下且价在下方  (过滤器保留的)
  B = 其余                    (过滤器丢掉的, D_INVERT=1)
若 B 的 PF/年化也为正, 过滤器就是在扔钱; 只有 B 明显亏, 过滤才有意义。
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
                UNIVERSE='universe_live.txt', TOTAL='1.5')

CASES = [
    ('全部信号',                dict(D_FILTER='0')),
    ('A: 日线趋势向下(保留的)',   dict(D_FILTER='1', D_MA='20', D_INVERT='0')),
    ('B: 其余(被丢掉的)',        dict(D_FILTER='1', D_MA='20', D_INVERT='1')),
]
_RESET = ('D_FILTER', 'D_MA', 'D_SLOPE', 'D_INVERT')


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
    tr = np.array(trades, float)
    w, l = tr[tr > 0].sum(), -tr[tr < 0].sum()
    m['pf'] = w / l if l > 0 else float('inf')
    m['avg'] = tr.mean() if len(tr) else 0        # 单笔均盈亏(绝对额, INIT=10000)
    m['held'] = avg_held
    return m


def main():
    print('=' * 92)
    print(f'信号切半独立回测  N={N}  MA90  1.5x  币池=universe_live')
    print('=' * 92)
    store = {}
    for span, lab in SPANS:
        print(f'\n### {lab}')
        print(f'{"配置":26s}{"年化":>9s}{"总收益":>9s}{"夏普":>7s}{"回撤":>8s}'
              f'{"笔数":>7s}{"胜率":>7s}{"PF":>6s}{"单笔均":>9s}')
        print('-' * 92)
        for name, env in CASES:
            m = run(span, env)
            store.setdefault(name, []).append(m)
            print(f'{name:26s}{m["ann"]:+8.1f}%{m["ret"]:+8.1f}%{m["sharpe"]:7.2f}'
                  f'{m["mdd"]:7.1f}%{m["trades"]:7d}{m["win"]:6.1f}%{m["pf"]:6.2f}{m["avg"]:+9.1f}')

    print('\n' + '=' * 92)
    print(f'{"配置":26s}{"三段年化":>28s}{"最差PF":>9s}{"总笔数":>8s}')
    print('-' * 92)
    for name, _ in CASES:
        rs = store[name]
        s = '  '.join(f'{m["ann"]:+7.1f}%' for m in rs)
        print(f'{name:26s}{s:>28s}{min(m["pf"] for m in rs):9.2f}'
              f'{sum(m["trades"] for m in rs):8d}')


if __name__ == '__main__':
    main()
