"""日线 ADX 趋势过滤三段检验。

规则: 日线 ADX(14) > 阈值 且 -DI > +DI 才允许做空。日线由 4h 重采样, shift(1) 无前视。

与 test_dfilter.py(日线MA过滤) 的区别: MA 看"价格位置", ADX 看"下跌趋势的力度",
两者不完全重复, 所以单独验, 并额外测 both(两个都要满足)。

同样跑三项验证:
  ① 直接对比三段年化
  ② 敞口对齐(过滤后持仓变少, 提 TOTAL 补回同等敞口再比)
  ③ 信号切两半(D_INVERT=1 只做被丢掉的), 看丢掉的是不是赚钱的单
用法: N=20 python test_dadx.py
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
          'D_INVERT', 'TOTAL')

ADX = dict(D_FILTER='1', D_MODE='adx', D_ADX_LEN='14', D_ADX_TH='25')

# 第一轮: 阈值/长度敏感性 + 与日线MA过滤/both 的对比
CASES1 = [
    ('基线(无过滤)',            dict(D_FILTER='0')),
    ('ADX>25 且 -DI>+DI',      dict(ADX)),
    ('ADX>20',                 dict(ADX, D_ADX_TH='20')),
    ('ADX>30',                 dict(ADX, D_ADX_TH='30')),
    ('ADX>25 (长度20)',        dict(ADX, D_ADX_LEN='20')),
    ('仅 -DI>+DI (不看ADX)',   dict(ADX, D_ADX_TH='0')),
    ('日线MA20过滤(上轮)',      dict(D_FILTER='1', D_MODE='ma', D_MA='20')),
    ('both: MA20 + ADX>25',    dict(ADX, D_MODE='both', D_MA='20')),
]

# 第二轮: 敞口对齐 (ADX 版持仓约为基线 6 成, TOTAL 提到 2.3~2.5x 补回)
CASES2 = [
    ('基线 1.5x',              dict(D_FILTER='0')),
    ('ADX>25  1.5x',           dict(ADX)),
    ('ADX>25  2.3x',           dict(ADX, TOTAL='2.3')),
    ('ADX>25  2.5x',           dict(ADX, TOTAL='2.5')),
]

# 第三轮: 信号切两半
CASES3 = [
    ('全部信号',                dict(D_FILTER='0')),
    ('A: ADX确认下跌(保留的)',   dict(ADX, D_INVERT='0')),
    ('B: 其余(被丢掉的)',        dict(ADX, D_INVERT='1')),
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
    print('\n' + '=' * 104)
    print(title)
    print('=' * 104)
    store = {}
    for span, lab in SPANS:
        rb = btc_daily(span) if show_alpha else None
        print(f'\n### {lab}')
        hdr = (f'{"配置":24s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"Calmar":>8s}'
               f'{"笔数":>7s}{"胜率":>7s}{"PF":>6s}{"均持仓":>7s}{"敞口":>8s}')
        if show_alpha:
            hdr += f'{"Alpha":>9s}'
        print(hdr)
        print('-' * 104)
        for name, env in cases:
            m = run(span, env)
            a = alpha_of(m['daily'], rb) if show_alpha else np.nan
            store.setdefault(name, []).append((m, a))
            line = (f'{name:24s}{m["ann"]:+8.1f}%{m["sharpe"]:7.2f}{m["mdd"]:7.1f}%'
                    f'{m["calmar"]:8.2f}{m["trades"]:7d}{m["win"]:6.1f}%{m["pf"]:6.2f}'
                    f'{m["held"]:7.1f}{m["expo"]:7.2f}x')
            if show_alpha:
                line += f'{a:+8.1f}%'
            print(line)
    print('\n' + '-' * 104)
    print(f'{"配置":24s}{"三段年化":>28s}{"最差":>9s}{"最差夏普":>10s}'
          f'{"最大回撤":>10s}{"最差PF":>8s}{"总笔数":>8s}')
    print('-' * 104)
    for name, _ in cases:
        rs = store[name]
        anns = [m['ann'] for m, a in rs]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:24s}{s:>28s}{min(anns):+8.1f}%{min(m["sharpe"] for m, a in rs):10.2f}'
              f'{max(m["mdd"] for m, a in rs):9.1f}%{min(m["pf"] for m, a in rs):8.2f}'
              f'{sum(m["trades"] for m, a in rs):8d}')
    return store


def main():
    print(f'日线 ADX 过滤检验   N={N}  MA90  币池=universe_live  ATR 10/10')
    block('第一轮: 阈值与长度敏感性', CASES1)
    block('第二轮: 敞口对齐 (过滤后持仓变少, 提 TOTAL 补回同等敞口)', CASES2, show_alpha=False)
    block('第三轮: 信号切两半 —— 被 ADX 丢掉的信号本身赚不赚钱', CASES3, show_alpha=False)


if __name__ == '__main__':
    main()
