"""波动放大过滤三段检验: ATR(14) > 过去20根ATR均值 才开仓。

维度与前四个过滤器不同 —— 那些看趋势方向(日线MA/ADX/慢线斜率), 这个看波动状态,
所以不能直接套用"重复处理"的结论, 必须独立验。

假设: 低波动横盘时死叉回踩信号密集但幅度小, 手续费+滑点占比高; 波动放大才有行情。
反假设(要认真对待): 做空反弹本质是"卖在阻力位", ATR 放大时价格离慢线更远、
止损(10·ATR)更宽, 单笔风险更大; 而且 ATR 放大常发生在暴跌之后 —— 那时反弹动能最强。

三项验证:
  ① 阈值/长度敏感性
  ② 敞口与回撤对齐(过滤后持仓变少, 提杠杆补回)
  ③ 信号切两半(VE_INVERT=1 只做低波动那批), 看被砍的赚不赚钱
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
_RESET = ('D_FILTER', 'D_MODE', 'D_MA', 'D_SLOPE', 'D_ADX_LEN', 'D_ADX_TH', 'D_INVERT',
          'MS_SLOPE', 'MS_L1', 'MS_L2', 'VOL_EXP', 'VE_LEN', 'VE_MULT', 'VE_INVERT', 'TOTAL')
VE = dict(VOL_EXP='1', VE_LEN='20', VE_MULT='1.0')

CASES1 = [
    ('基线(无波动条件)',           dict(VOL_EXP='0')),
    ('★ATR>过去20根均值',         dict(VE)),
    ('ATR>均值×1.2 (更严)',       dict(VE, VE_MULT='1.2')),
    ('ATR>均值×1.5 (最严)',       dict(VE, VE_MULT='1.5')),
    ('ATR>均值×0.9 (更松)',       dict(VE, VE_MULT='0.9')),
    ('长度10',                    dict(VE, VE_LEN='10')),
    ('长度40',                    dict(VE, VE_LEN='40')),
]

CASES2 = [
    ('基线 1.5x',                 dict(VOL_EXP='0')),
    ('波动放大 1.5x',             dict(VE)),
    ('波动放大 2.0x',             dict(VE, TOTAL='2.0')),
    ('波动放大 2.5x',             dict(VE, TOTAL='2.5')),
    ('波动放大 3.0x',             dict(VE, TOTAL='3.0')),
]

CASES3 = [
    ('全部信号',                   dict(VOL_EXP='0')),
    ('A: 波动放大(保留的)',        dict(VE, VE_INVERT='0')),
    ('B: 低波动(被砍的)',          dict(VE, VE_INVERT='1')),
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
    print('\n' + '=' * 110)
    print(title)
    print('=' * 110)
    store = {}
    for span, lab in SPANS:
        rb = btc_daily(span) if show_alpha else None
        print(f'\n### {lab}')
        hdr = (f'{"配置":24s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"Calmar":>8s}{"笔数":>7s}'
               f'{"胜率":>7s}{"PF":>6s}{"单笔均":>8s}{"t值":>6s}{"均持仓":>7s}{"敞口":>8s}')
        if show_alpha:
            hdr += f'{"Alpha":>9s}'
        print(hdr)
        print('-' * 110)
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
    print('\n' + '-' * 110)
    print(f'{"配置":24s}{"三段年化":>28s}{"最差":>9s}{"最差夏普":>10s}{"最大回撤":>10s}'
          f'{"最差PF":>8s}{"最差t":>7s}{"总笔数":>8s}')
    print('-' * 110)
    for name, _ in cases:
        rs = store[name]
        anns = [m['ann'] for m, a in rs]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:24s}{s:>28s}{min(anns):+8.1f}%{min(m["sharpe"] for m, a in rs):10.2f}'
              f'{max(m["mdd"] for m, a in rs):9.1f}%{min(m["pf"] for m, a in rs):8.2f}'
              f'{min(m["t"] for m, a in rs):7.2f}{sum(m["trades"] for m, a in rs):8d}')
    return store


def main():
    print(f'波动放大过滤检验  N={N}  MA90  币池=universe_live  ATR 10/10')
    block('第一轮: 阈值与长度敏感性', CASES1)
    block('第二轮: 敞口与回撤对齐', CASES2, show_alpha=False)
    block('第三轮: 信号切两半 —— 被砍的低波动信号赚不赚钱', CASES3, show_alpha=False)


if __name__ == '__main__':
    main()
