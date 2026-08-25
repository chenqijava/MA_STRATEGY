"""慢线斜率过滤三段检验: 入场加 ma90[1] > ma90[6] (慢线仍上行)。

为什么这个和前两个过滤器不同, 值得单独测:
  日线MA / ADX 过滤都是"要求趋势已经向下", 三段全败, 我给的失败解释是
  "最赚钱的空单出现在长期趋势还没坏的早期"。ma90[1]>ma90[6] 正是这个解释的
  直接检验 —— 方向相反。若解释成立, 这个过滤器应该有效。

同时测 down (慢线下行) 作对照: 若 up/down 都不如基线, 说明斜率无信息;
若 up 好 down 差, 说明"早期做空"的解释成立。

注意样本分布: 全池做空信号里仅 23% 落在慢线上行时, 所以 up 会砍掉 77% 信号,
必须做敞口对齐和回撤对齐, 否则"收益低"只是仓位小。
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
BASE_ENV = dict(MA_SLOW='90', MA_EXIT='1', COOLDOWN_ONCE='1',
                ATR_TP='10', ATR_SL='10', SIZING='notional',
                CROSS_DELAY='6', TOUCH_ATR='0.5', MAX_BARS='40',
                UNIVERSE='universe_live.txt', TOTAL='1.5')
_RESET = ('D_FILTER', 'D_MODE', 'D_MA', 'D_SLOPE', 'D_ADX_LEN', 'D_ADX_TH',
          'D_INVERT', 'MS_SLOPE', 'MS_L1', 'MS_L2', 'TOTAL')

CASES1 = [
    ('基线(无斜率条件)',        dict(MS_SLOPE='0')),
    ('★ma90[1]>ma90[6] 上行',  dict(MS_SLOPE='up')),
    ('对照: 慢线下行',          dict(MS_SLOPE='down')),
    ('上行 [1]vs[3] 更短',      dict(MS_SLOPE='up', MS_L2='3')),
    ('上行 [1]vs[12] 更长',     dict(MS_SLOPE='up', MS_L2='12')),
    ('上行 [1]vs[24] 一天',     dict(MS_SLOPE='up', MS_L2='24')),
]

# 上行版砍掉 ~77% 信号, 必须放大杠杆到同等敞口/同等回撤再比
CASES2 = [
    ('基线 1.5x',              dict(MS_SLOPE='0')),
    ('上行 1.5x',              dict(MS_SLOPE='up')),
    ('上行 3.0x',              dict(MS_SLOPE='up', TOTAL='3.0')),
    ('上行 4.5x',              dict(MS_SLOPE='up', TOTAL='4.5')),
    ('上行 6.0x',              dict(MS_SLOPE='up', TOTAL='6.0')),
]


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
    m['held'] = avg_held
    m['expo'] = avg_held * ps.TOTAL_NOTIONAL / N
    m['avg'] = tr.mean() if len(tr) else 0
    m['t'] = (tr.mean() / (tr.std(ddof=1) / np.sqrt(len(tr)))
              if len(tr) > 1 and tr.std(ddof=1) > 0 else 0)
    m['daily'] = pd.Series(curve, index=idx).resample('1D').last().dropna().pct_change().dropna()
    return m


def btc_daily(span):
    os.environ['SPAN'] = span
    importlib.reload(al)
    c = mp.cc.load_data(al.path('BTC'))['close'].resample('1D').last().dropna()
    return c.pct_change().dropna()


def alpha_of(rs, rb):
    df = pd.concat([rs.rename('s'), rb.rename('b')], axis=1).dropna()
    if len(df) < 30:
        return np.nan
    b, a = np.polyfit(df['b'].values, df['s'].values, 1)
    return a * 365 * 100


def block(title, cases, show_alpha=True):
    print('\n' + '=' * 108)
    print(title)
    print('=' * 108)
    store = {}
    for span, lab in SPANS:
        rb = btc_daily(span) if show_alpha else None
        print(f'\n### {lab}')
        hdr = (f'{"配置":24s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"Calmar":>8s}{"笔数":>7s}'
               f'{"胜率":>7s}{"PF":>6s}{"单笔均":>8s}{"t值":>6s}{"均持仓":>7s}{"敞口":>8s}')
        if show_alpha:
            hdr += f'{"Alpha":>9s}'
        print(hdr)
        print('-' * 108)
        for name, env in cases:
            m = run(span, env)
            a = alpha_of(m['daily'], rb) if show_alpha else np.nan
            store.setdefault(name, []).append((m, a))
            line = (f'{name:24s}{m["ann"]:+8.1f}%{m["sharpe"]:7.2f}{m["mdd"]:7.1f}%'
                    f'{m["calmar"]:8.2f}{m["trades"]:7d}{m["win"]:6.1f}%{m["pf"]:6.2f}'
                    f'{m["avg"]:+8.1f}{m["t"]:6.2f}{m["held"]:7.1f}{m["expo"]:7.2f}x')
            if show_alpha:
                line += f'{a:+8.1f}%'
            print(line)
    print('\n' + '-' * 108)
    print(f'{"配置":24s}{"三段年化":>28s}{"最差":>9s}{"最差夏普":>10s}{"最大回撤":>10s}'
          f'{"最差PF":>8s}{"总笔数":>8s}')
    print('-' * 108)
    for name, _ in cases:
        rs = store[name]
        anns = [m['ann'] for m, a in rs]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:24s}{s:>28s}{min(anns):+8.1f}%{min(m["sharpe"] for m, a in rs):10.2f}'
              f'{max(m["mdd"] for m, a in rs):9.1f}%{min(m["pf"] for m, a in rs):8.2f}'
              f'{sum(m["trades"] for m, a in rs):8d}')
    return store


def main():
    print(f'慢线斜率过滤检验  N={N}  MA90  币池=universe_live  ATR 10/10')
    block('第一轮: 上行/下行 与回看长度敏感性', CASES1)
    block('第二轮: 敞口与回撤对齐 (上行版砍掉约77%信号, 提杠杆补回)', CASES2, show_alpha=False)


if __name__ == '__main__':
    main()
