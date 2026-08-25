"""逐币日线趋势过滤 (D_FILTER) 三段检验。

规则: 只在"该币日线 MA(D_MA) 向下 且 日收盘在 MA 下方"时才允许做空。
日线状态用上一日已收盘的值 (shift(1)), 无前视。

为什么要严格验: 之前 32 个全市场牛熊判定器全部没有区分度, BTC过滤在 MA90 下还是
净负的(alpha 从 +37% 砍到 +24%)。所以这里不看单段收益, 看三段最差 + alpha 是否被砍。

用法: UNIVERSE=universe_live.txt TOTAL=1.5 N=20 python test_dfilter.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import importlib
import numpy as np
import pandas as pd
import analyze_long as al
import ma_pullback as mp
import portfolio_sim as ps

SPANS = [('20210101_20230101', '2021-01~2023-01 (BTC -54%)'),
         ('20230101_20250101', '2023-01~2025-01 (BTC +342%)'),
         ('20250101_20260801', '2025-01~2026-08 (BTC -37%)')]
N = int(os.environ.get('N', '20'))

# 基线 = 当前实盘配置 H (MA90/N20/等名义1.5x/MA出场/单次窗口/ATR 10-10 保险)
BASE_ENV = dict(MA_SLOW='90', MA_EXIT='1', COOLDOWN_ONCE='1',
                ATR_TP='10', ATR_SL='10', SIZING='notional',
                CROSS_DELAY='6', TOUCH_ATR='0.5', MAX_BARS='40')

CASES = [
    ('基线(无日线过滤)',        dict(D_FILTER='0')),
    ('日线MA20向下+价在下方',   dict(D_FILTER='1', D_MA='20', D_SLOPE='1')),
    ('日线MA20向下(斜率5日)',   dict(D_FILTER='1', D_MA='20', D_SLOPE='5')),
    ('日线MA10',                dict(D_FILTER='1', D_MA='10', D_SLOPE='1')),
    ('日线MA50',                dict(D_FILTER='1', D_MA='50', D_SLOPE='1')),
]
_RESET = ('D_FILTER', 'D_MA', 'D_SLOPE')


def run(span, env):
    os.environ['SPAN'] = span
    for k in _RESET:
        os.environ.pop(k, None)
    for k, v in {**BASE_ENV, **env}.items():
        os.environ[k] = str(v)
    importlib.reload(al)
    importlib.reload(ps)
    # reload 清空了面板全局, 必须在本次 import 后重建 (否则 MA出场/日线过滤静默失效)
    idx, C, H, L, A, S, syms = ps.build_panel()
    curve, trades, avg_held = ps.simulate(N, idx, C, H, L, A, S)
    m = ps.metrics(curve, trades, idx)
    tr = np.array(trades, float)
    win_sum = tr[tr > 0].sum() if len(tr) else 0.0
    loss_sum = -tr[tr < 0].sum() if len(tr) else 0.0
    m['pf'] = win_sum / loss_sum if loss_sum > 0 else float('inf')
    m['held'] = avg_held
    m['daily'] = pd.Series(curve, index=idx).resample('1D').last().dropna().pct_change().dropna()
    return m


def ols_alpha_beta(rs, rb):
    df = pd.concat([rs.rename('s'), rb.rename('b')], axis=1).dropna()
    if len(df) < 30:
        return np.nan, np.nan
    b, a = np.polyfit(df['b'].values, df['s'].values, 1)
    return a * 365 * 100, b


def btc_daily(span):
    os.environ['SPAN'] = span
    importlib.reload(al)
    c = mp.cc.load_data(al.path('BTC'))['close'].resample('1D').last().dropna()
    return c.pct_change().dropna()


def main():
    pd.set_option('display.width', 220)
    print('=' * 108)
    print(f'逐币日线趋势过滤检验   N={N}  MA_SLOW={BASE_ENV["MA_SLOW"]}  '
          f'总敞口={os.environ.get("TOTAL", "3.0")}x  币池={os.environ.get("UNIVERSE", "universe_pass.txt")}')
    print('=' * 108)
    store = {}
    for span, lab in SPANS:
        rb = btc_daily(span)
        print(f'\n### {lab}')
        print(f'{"配置":26s}{"年化":>9s}{"总收益":>9s}{"夏普":>7s}{"回撤":>8s}'
              f'{"笔数":>7s}{"胜率":>7s}{"PF":>6s}{"均持仓":>7s}{"Alpha年化":>10s}{"Beta":>7s}')
        print('-' * 108)
        for name, env in CASES:
            m = run(span, env)
            a, b = ols_alpha_beta(m['daily'], rb)
            store.setdefault(name, []).append((m, a, b))
            print(f'{name:26s}{m["ann"]:+8.1f}%{m["ret"]:+8.1f}%{m["sharpe"]:7.2f}{m["mdd"]:7.1f}%'
                  f'{m["trades"]:7d}{m["win"]:6.1f}%{m["pf"]:6.2f}{m["held"]:7.1f}{a:+9.1f}%{b:7.2f}')

    print('\n' + '=' * 108)
    print('三段汇总 (决策看"三段最差年化", 单段最优没有意义)')
    print('=' * 108)
    print(f'{"配置":26s}{"三段年化":>28s}{"最差年化":>10s}{"最差PF":>8s}{"最大回撤":>10s}{"总笔数":>8s}{"最差Alpha":>11s}')
    print('-' * 108)
    for name, _ in CASES:
        rows = store[name]
        anns = [m['ann'] for m, a, b in rows]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:26s}{s:>28s}{min(anns):+9.1f}%{min(m["pf"] for m, a, b in rows):8.2f}'
              f'{max(m["mdd"] for m, a, b in rows):9.1f}%{sum(m["trades"] for m, a, b in rows):8d}'
              f'{min(a for m, a, b in rows):+10.1f}%')


if __name__ == '__main__':
    main()
