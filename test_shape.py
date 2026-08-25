"""K线形态加强三段检验: 长上影线 / 跌破前低。

用户提议把现有的"收盘<开盘"(阴线)加强为:
  ① 长上影线: 上影线 > 实体 × 2 (冲高被打回来, 高位卖压确认)
  ② 收盘 < 上一根最低价 (跌破前低, 比阴线更强的下行确认)

注: 提议里的"最高价接近MA90"已由现有 touch 条件实现(gap = MA90 - high 在 0~0.5ATR
之间, 且 >=0 保证未上破), 无需重复, 故只测影线比例与跌破前低。

BTC 单币存活率(2025-26): 基线49 -> 上影>2倍17(35%) / 跌破前低19(39%) / 两者都要2(4%)。
按前五次的规律(砍得越多越差), 预期不乐观, 但形态与前面的趋势/波动过滤维度不同,
且是对入场那一根本身的刻画(不是外部环境), 值得独立验。
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
_RESET = ('UPPER_SHADOW', 'BREAK_PREV_LOW', 'TOTAL')

CASES1 = [
    ('基线(仅阴线)',            dict()),
    ('★上影>实体×2',           dict(UPPER_SHADOW='2')),
    ('上影>实体×1',            dict(UPPER_SHADOW='1')),
    ('上影>实体×0.5',          dict(UPPER_SHADOW='0.5')),
    ('上影>实体×3',            dict(UPPER_SHADOW='3')),
    ('★收盘<前一根最低',        dict(BREAK_PREV_LOW='1')),
    ('两者都要',                dict(UPPER_SHADOW='2', BREAK_PREV_LOW='1')),
]

# 形态过滤砍掉大量信号, 提杠杆补回敞口/回撤再比
CASES2 = [
    ('基线 1.5x',              dict()),
    ('上影×2  1.5x',           dict(UPPER_SHADOW='2')),
    ('上影×2  3.0x',           dict(UPPER_SHADOW='2', TOTAL='3.0')),
    ('上影×2  4.0x',           dict(UPPER_SHADOW='2', TOTAL='4.0')),
    ('跌破前低 1.5x',           dict(BREAK_PREV_LOW='1')),
    ('跌破前低 3.0x',           dict(BREAK_PREV_LOW='1', TOTAL='3.0')),
    ('跌破前低 4.0x',           dict(BREAK_PREV_LOW='1', TOTAL='4.0')),
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
    m['sig'] = int((S.values == -1).sum())        # 信号总数(未受槽位约束)
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
    print('\n' + '=' * 112)
    print(title)
    print('=' * 112)
    store = {}
    for span, lab in SPANS:
        rb = btc_daily(span) if show_alpha else None
        print(f'\n### {lab}')
        hdr = (f'{"配置":22s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"Calmar":>8s}{"信号":>7s}'
               f'{"笔数":>7s}{"胜率":>7s}{"PF":>6s}{"单笔均":>8s}{"t值":>6s}{"均持仓":>7s}')
        if show_alpha:
            hdr += f'{"Alpha":>9s}'
        print(hdr)
        print('-' * 112)
        for name, env in cases:
            m = run(span, env)
            a = alpha_of(m['daily'], rb) if show_alpha else np.nan
            store.setdefault(name, []).append((m, a))
            line = (f'{name:22s}{m["ann"]:+8.1f}%{m["sharpe"]:7.2f}{m["mdd"]:7.1f}%'
                    f'{m["calmar"]:8.2f}{m["sig"]:7d}{m["trades"]:7d}{m["win"]:6.1f}%'
                    f'{m["pf"]:6.2f}{m["avg"]:+8.1f}{m["t"]:6.2f}{m["held"]:7.1f}')
            if show_alpha:
                line += f'{a:+8.1f}%'
            print(line)
    print('\n' + '-' * 112)
    print(f'{"配置":22s}{"三段年化":>28s}{"最差":>9s}{"最差夏普":>10s}{"最大回撤":>10s}'
          f'{"最差PF":>8s}{"最差t":>7s}{"总信号":>8s}{"总笔数":>8s}')
    print('-' * 112)
    for name, _ in cases:
        rs = store[name]
        anns = [m['ann'] for m, a in rs]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:22s}{s:>28s}{min(anns):+8.1f}%{min(m["sharpe"] for m, a in rs):10.2f}'
              f'{max(m["mdd"] for m, a in rs):9.1f}%{min(m["pf"] for m, a in rs):8.2f}'
              f'{min(m["t"] for m, a in rs):7.2f}{sum(m["sig"] for m, a in rs):8d}'
              f'{sum(m["trades"] for m, a in rs):8d}')
    return store


def main():
    print(f'K线形态加强检验  N={N}  MA90  币池=universe_live  ATR 10/10')
    block('第一轮: 影线倍数敏感性 + 跌破前低', CASES1)
    block('第二轮: 敞口与回撤对齐', CASES2, show_alpha=False)


if __name__ == '__main__':
    main()
