"""跌破前低 × ATR止盈止损 交互检验。

上一轮: 跌破前低(收盘<前一根最低)在 ATR 10/10 下三段 +9.7%/+25.1%/+74.0%,
两段超基线但独立验证段(2021-23)输 10.5pp, 按三段最差否决。

本轮问的是交互: 基线上 ATR 4/2 已验证劣于 10/10 (三段最差 +11.0% vs +20.2%),
但那是在"仅阴线"的入场上测的。跌破前低的入场确认更强(胜率+2.1pp, PF+0.17),
下行更确定 -> 更近的止盈(4·ATR)也许能在反弹前锁住利润, 而 2·ATR 止损被噪声
打掉的概率也更低。这个交互不能从两个单独结论推出来, 必须实测。

网格: {仅阴线, 跌破前低} × {10/10, 4/2, 4/4, 6/3, 10/5} + 时停敏感性。
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
                SIZING='notional', CROSS_DELAY='6', TOUCH_ATR='0.5',
                MAX_BARS='40', UNIVERSE='universe_live.txt', TOTAL='1.5')
_RESET = ('UPPER_SHADOW', 'BREAK_PREV_LOW', 'ATR_TP', 'ATR_SL', 'MAX_BARS', 'TOTAL')

BPL = dict(BREAK_PREV_LOW='1')

CASES1 = [
    ('基线 阴线 10/10',        dict(ATR_TP='10', ATR_SL='10')),
    ('基线 阴线 4/2',          dict(ATR_TP='4', ATR_SL='2')),
    ('跌破前低 10/10',         dict(BPL, ATR_TP='10', ATR_SL='10')),
    ('★跌破前低 4/2',          dict(BPL, ATR_TP='4', ATR_SL='2')),
    ('跌破前低 4/4',           dict(BPL, ATR_TP='4', ATR_SL='4')),
    ('跌破前低 6/3',           dict(BPL, ATR_TP='6', ATR_SL='3')),
    ('跌破前低 10/5',          dict(BPL, ATR_TP='10', ATR_SL='5')),
    ('跌破前低 6/6',           dict(BPL, ATR_TP='6', ATR_SL='6')),
]

# 若 4/2 有效, 时停应重新扫 —— 更近的止盈会改变最优持仓上限
CASES2 = [
    ('跌破前低 4/2 时停40',    dict(BPL, ATR_TP='4', ATR_SL='2', MAX_BARS='40')),
    ('跌破前低 4/2 时停20',    dict(BPL, ATR_TP='4', ATR_SL='2', MAX_BARS='20')),
    ('跌破前低 4/2 时停60',    dict(BPL, ATR_TP='4', ATR_SL='2', MAX_BARS='60')),
    ('跌破前低 4/2 无时停',    dict(BPL, ATR_TP='4', ATR_SL='2', MAX_BARS='0')),
    ('基线 10/10 时停40 ★现行', dict(ATR_TP='10', ATR_SL='10', MAX_BARS='40')),
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
        hdr = (f'{"配置":24s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"Calmar":>8s}'
               f'{"笔数":>7s}{"胜率":>7s}{"PF":>6s}{"单笔均":>8s}{"t值":>6s}{"均持仓":>7s}')
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
                    f'{m["avg"]:+8.1f}{m["t"]:6.2f}{m["held"]:7.1f}')
            if show_alpha:
                line += f'{a:+8.1f}%'
            print(line)
    print('\n' + '-' * 108)
    print(f'{"配置":24s}{"三段年化":>28s}{"最差":>9s}{"最差夏普":>10s}{"最大回撤":>10s}'
          f'{"最差PF":>8s}{"最差t":>7s}{"总笔数":>8s}')
    print('-' * 108)
    for name, _ in cases:
        rs = store[name]
        anns = [m['ann'] for m, a in rs]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:24s}{s:>28s}{min(anns):+8.1f}%{min(m["sharpe"] for m, a in rs):10.2f}'
              f'{max(m["mdd"] for m, a in rs):9.1f}%{min(m["pf"] for m, a in rs):8.2f}'
              f'{min(m["t"] for m, a in rs):7.2f}{sum(m["trades"] for m, a in rs):8d}')
    return store


def main():
    print(f'跌破前低 × ATR 交互检验  N={N}  MA90  币池=universe_live')
    block('第一轮: 入场形态 × ATR止盈止损', CASES1)
    block('第二轮: 时停敏感性 (更近的止盈会改变最优持仓上限)', CASES2, show_alpha=False)


if __name__ == '__main__':
    main()
