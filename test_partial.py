"""分批止盈三段检验。

实现: PT_LEVELS/PT_FRACS 指定"浮盈达N倍ATR平掉原仓位的X%", 可多档;
PT_BE=1 则第一批后止损移成本价。

口径注意(否则无法与基线比):
  · 一笔交易的盈亏 = 各批次之和, 在最终清仓时作为**一笔**计入 trades。
    若每批各记一笔, 分批本身就会造出一堆小赢单, 胜率和笔数都被稀释。
  · 已分批落袋的盈亏计入盯市权益, 否则回撤/夏普会低估。

先验预期: 分批止盈提高胜率(小赢变多)、降低单笔波动, 但砍掉大赢家的尾部。
本策略胜率仅 31~35%, PF 靠少数大赢家撑起(前10大赢家占总盈利 9~16%),
所以尾部很值钱 —— 这与上一轮追踪均线的结论一致(MA10 胜率+6pp 但年化减半)。
用数据检验, 不预设。
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
                ATR_TP='10', ATR_SL='10', SIZING='notional', CROSS_DELAY='6',
                TOUCH_ATR='0.5', MAX_BARS='40', UNIVERSE='universe_live.txt', TOTAL='1.5')
_RESET = ('PT_LEVELS', 'PT_FRACS', 'PT_BE', 'BE_ATR', 'TRAIL_ATR', 'TRAIL_MA',
          'TRAIL_ARM', 'TOTAL')


def pt(lv, fr, **kw):
    return dict(PT_LEVELS=lv, PT_FRACS=fr, **kw)


CASES1 = [
    ('基线 ★现行',            dict()),
    ('2ATR平半仓',            pt('2', '0.5')),
    ('3ATR平半仓',            pt('3', '0.5')),
    ('4ATR平半仓',            pt('4', '0.5')),
    ('6ATR平半仓',            pt('6', '0.5')),
    ('3ATR平1/3',            pt('3', '0.333')),
    ('3ATR平2/3',            pt('3', '0.667')),
    ('三档 2/4/6 各1/3',      pt('2,4,6', '0.333,0.333,0.333')),
    ('两档 3/6 各50%/25%',    pt('3,6', '0.5,0.25')),
    ('两档 4/8 各50%/25%',    pt('4,8', '0.5,0.25')),
]

CASES2 = [
    ('基线 ★现行',            dict()),
    ('3ATR半仓',              pt('3', '0.5')),
    ('3ATR半仓 + 保本',        pt('3', '0.5', PT_BE='1')),
    ('4ATR半仓 + 保本',        pt('4', '0.5', PT_BE='1')),
    ('6ATR半仓 + 保本',        pt('6', '0.5', PT_BE='1')),
    ('3ATR半仓+追踪MA20',      pt('3', '0.5', TRAIL_MA='20', TRAIL_ARM='4')),
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
    wins = np.sort(tr[tr > 0])[::-1]
    m['top1'] = wins[0] if len(wins) else 0
    m['top10share'] = wins[:10].sum() / w * 100 if len(wins) and w > 0 else 0
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
    print('\n' + '=' * 116)
    print(title)
    print('=' * 116)
    store = {}
    for span, lab in SPANS:
        rb = btc_daily(span) if show_alpha else None
        print(f'\n### {lab}')
        hdr = (f'{"配置":22s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"Calmar":>8s}{"笔数":>7s}'
               f'{"胜率":>7s}{"PF":>6s}{"单笔均":>8s}{"t值":>6s}{"最大单笔":>9s}{"前10占":>7s}')
        if show_alpha:
            hdr += f'{"Alpha":>9s}'
        print(hdr)
        print('-' * 116)
        for name, env in cases:
            m = run(span, env)
            a = alpha_of(m['daily'], rb) if show_alpha else np.nan
            store.setdefault(name, []).append((m, a))
            line = (f'{name:22s}{m["ann"]:+8.1f}%{m["sharpe"]:7.2f}{m["mdd"]:7.1f}%'
                    f'{m["calmar"]:8.2f}{m["trades"]:7d}{m["win"]:6.1f}%{m["pf"]:6.2f}'
                    f'{m["avg"]:+8.1f}{m["t"]:6.2f}{m["top1"]:9.0f}{m["top10share"]:6.0f}%')
            if show_alpha:
                line += f'{a:+8.1f}%'
            print(line)
    print('\n' + '-' * 116)
    print(f'{"配置":22s}{"三段年化":>28s}{"最差":>9s}{"最差夏普":>10s}{"最大回撤":>10s}'
          f'{"最差PF":>8s}{"最差t":>7s}{"最差胜率":>9s}')
    print('-' * 116)
    for name, _ in cases:
        rs = store[name]
        anns = [m['ann'] for m, a in rs]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:22s}{s:>28s}{min(anns):+8.1f}%{min(m["sharpe"] for m, a in rs):10.2f}'
              f'{max(m["mdd"] for m, a in rs):9.1f}%{min(m["pf"] for m, a in rs):8.2f}'
              f'{min(m["t"] for m, a in rs):7.2f}{min(m["win"] for m, a in rs):8.1f}%')
    return store


def main():
    print(f'分批止盈检验  N={N}  MA90  币池=universe_live  ATR 10/10')
    block('第一轮: 档位与比例', CASES1)
    block('第二轮: 分批 + 保本/追踪 组合', CASES2, show_alpha=False)


if __name__ == '__main__':
    main()
