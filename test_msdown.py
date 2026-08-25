"""ma90[1] < ma90[6] (慢线下行) 完整检验。

上一轮它是 up 的对照组, 三段 +19.3%/+12.9%/+90.3%, 是四个过滤器里唯一接近基线的:
  · 只砍 21% 信号 (up 砍 77%, ADX 砍 58%, 日线MA 砍 30%)
  · 被它砍掉的那半(=up)  PF 仅 1.04, 接近盈亏平衡 —— 砍的确实是差单
  · 2025-26 段夏普 1.95>1.82、回撤 20.7%<24.6%、Calmar 4.36>3.90, 三项都更好
  · 2021-23 段 +19.3% vs 基线 +20.2%, 几乎持平
  亏就亏在 2023-25 牛市段: +12.9% vs +24.1%。

所以要查清: ① 砍掉近乎平手的单再把敞口补回来, 是不是净赚?
            ② 2023-25 那 11pp 的损失是什么造成的?
            ③ 回看长度是否稳健?
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
    ('基线(无斜率条件)',          dict(MS_SLOPE='0')),
    ('★ma90[1]<ma90[6] 下行',    dict(MS_SLOPE='down')),
    ('下行 [1]vs[3]',            dict(MS_SLOPE='down', MS_L2='3')),
    ('下行 [1]vs[12]',           dict(MS_SLOPE='down', MS_L2='12')),
    ('下行 [1]vs[24] 一天',      dict(MS_SLOPE='down', MS_L2='24')),
    ('下行 [1]vs[42] 一周',      dict(MS_SLOPE='down', MS_L2='42')),
    ('对照: 上行(被它砍掉的半)',   dict(MS_SLOPE='up')),
]

# 下行版砍 21% 信号, 敞口从 0.30x 降到 ~0.25x, 提 TOTAL 补回并继续加到同等回撤
CASES2 = [
    ('基线 1.5x',                dict(MS_SLOPE='0')),
    ('下行 1.5x',                dict(MS_SLOPE='down')),
    ('下行 1.8x(敞口对齐)',       dict(MS_SLOPE='down', TOTAL='1.8')),
    ('下行 2.0x',                dict(MS_SLOPE='down', TOTAL='2.0')),
    ('下行 2.2x(回撤对齐)',       dict(MS_SLOPE='down', TOTAL='2.2')),
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
        return np.nan, np.nan
    b, a = np.polyfit(df['b'].values, df['s'].values, 1)
    return a * 365 * 100, b


def block(title, cases, show_alpha=True):
    print('\n' + '=' * 112)
    print(title)
    print('=' * 112)
    store = {}
    for span, lab in SPANS:
        rb = btc_daily(span) if show_alpha else None
        print(f'\n### {lab}')
        hdr = (f'{"配置":26s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"Calmar":>8s}{"笔数":>7s}'
               f'{"胜率":>7s}{"PF":>6s}{"单笔均":>8s}{"t值":>6s}{"均持仓":>7s}{"敞口":>8s}')
        if show_alpha:
            hdr += f'{"Alpha":>9s}{"Beta":>7s}'
        print(hdr)
        print('-' * 112)
        for name, env in cases:
            m = run(span, env)
            a, bt = alpha_of(m['daily'], rb) if show_alpha else (np.nan, np.nan)
            store.setdefault(name, []).append((m, a))
            line = (f'{name:26s}{m["ann"]:+8.1f}%{m["sharpe"]:7.2f}{m["mdd"]:7.1f}%'
                    f'{m["calmar"]:8.2f}{m["trades"]:7d}{m["win"]:6.1f}%{m["pf"]:6.2f}'
                    f'{m["avg"]:+8.1f}{m["t"]:6.2f}{m["held"]:7.1f}{m["expo"]:7.2f}x')
            if show_alpha:
                line += f'{a:+8.1f}%{bt:7.2f}'
            print(line)
    print('\n' + '-' * 112)
    print(f'{"配置":26s}{"三段年化":>28s}{"最差":>9s}{"最差夏普":>10s}{"最大回撤":>10s}'
          f'{"最差PF":>8s}{"最差t":>7s}{"总笔数":>8s}')
    print('-' * 112)
    for name, _ in cases:
        rs = store[name]
        anns = [m['ann'] for m, a in rs]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:26s}{s:>28s}{min(anns):+8.1f}%{min(m["sharpe"] for m, a in rs):10.2f}'
              f'{max(m["mdd"] for m, a in rs):9.1f}%{min(m["pf"] for m, a in rs):8.2f}'
              f'{min(m["t"] for m, a in rs):7.2f}{sum(m["trades"] for m, a in rs):8d}')
    return store


def main():
    print(f'慢线下行过滤检验  N={N}  MA90  币池=universe_live  ATR 10/10')
    block('第一轮: 下行 + 回看长度敏感性 (含上行对照)', CASES1)
    block('第二轮: 敞口对齐与回撤对齐', CASES2, show_alpha=False)


if __name__ == '__main__':
    main()
