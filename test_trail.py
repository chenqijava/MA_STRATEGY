"""移动止盈 / 保本止损 / 追踪均线 三段检验。

提议的四个部件 (已全部实现, 可独立开关与组合):
  BE_ATR=2    浮盈达2倍ATR后, 止损移到成本价 (保本)
  TRAIL_ATR=3 移动止损 = 持仓期最低价 + 3·ATR
  TRAIL_MA=10 收盘上穿 MA10 即平 (比 MA90 快得多)
  TRAIL_ARM   上述两个追踪部件的启动门槛(浮盈达N倍ATR后才启用)

提议里"必须有止损否则一次逼空就爆仓"这点已经满足: ATR_SL=10 一直在, 且是交易所侧
algo 委托。10·ATR 是极端保险(实测触发率<1%), 不是没有止损。

预期(要用数据检验而非假设): 提议说"保留更多利润, 同时提高胜率"。这两件事通常是
互斥的 —— 提前锁利提高胜率但砍掉大赢家。本策略胜率仅 31~35%, 靠少数大赢家撑起
PF 1.29~1.55, 所以砍大赢家的代价可能很高。先看单部件, 再看组合。
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
_RESET = ('BE_ATR', 'TRAIL_ATR', 'TRAIL_PCT', 'TRAIL_MA', 'TRAIL_ARM', 'TOTAL')

CASES1 = [
    ('基线 ★现行',              dict()),
    ('保本 BE=2',              dict(BE_ATR='2')),
    ('保本 BE=3',              dict(BE_ATR='3')),
    ('保本 BE=4',              dict(BE_ATR='4')),
    ('移动止损 3ATR',           dict(TRAIL_ATR='3')),
    ('移动止损 3ATR arm2',      dict(TRAIL_ATR='3', TRAIL_ARM='2')),
    ('移动止损 5ATR arm2',      dict(TRAIL_ATR='5', TRAIL_ARM='2')),
    ('回撤止损 5% ',            dict(TRAIL_PCT='0.05')),
    ('回撤止损 5% arm2',        dict(TRAIL_PCT='0.05', TRAIL_ARM='2')),
    ('回撤止损 3% arm2',        dict(TRAIL_PCT='0.03', TRAIL_ARM='2')),
    ('回撤止损 8% arm2',        dict(TRAIL_PCT='0.08', TRAIL_ARM='2')),
    ('回撤止损 5% arm4',        dict(TRAIL_PCT='0.05', TRAIL_ARM='4')),
    ('追踪MA10 arm2',          dict(TRAIL_MA='10', TRAIL_ARM='2')),
    ('追踪MA20 arm2',          dict(TRAIL_MA='20', TRAIL_ARM='2')),
    ('追踪MA20 arm4',          dict(TRAIL_MA='20', TRAIL_ARM='4')),
    ('追踪MA20 arm0(即时)',     dict(TRAIL_MA='20')),
]

CASES2 = [
    ('基线 ★现行',                    dict()),
    ('提议全套: BE2+MA10 arm2',        dict(BE_ATR='2', TRAIL_MA='10', TRAIL_ARM='2')),
    ('BE2 + 移动止损3ATR arm2',        dict(BE_ATR='2', TRAIL_ATR='3', TRAIL_ARM='2')),
    ('BE3 + 追踪MA20 arm4',           dict(BE_ATR='3', TRAIL_MA='20', TRAIL_ARM='4')),
    ('BE4 + 移动5ATR arm4',           dict(BE_ATR='4', TRAIL_ATR='5', TRAIL_ARM='4')),
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
    # 最大单笔盈利 & 前10大赢家占总盈利比 —— 检验"是否砍掉了大赢家"
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
        hdr = (f'{"配置":24s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"Calmar":>8s}{"笔数":>7s}'
               f'{"胜率":>7s}{"PF":>6s}{"单笔均":>8s}{"t值":>6s}{"最大单笔":>9s}{"前10占":>7s}')
        if show_alpha:
            hdr += f'{"Alpha":>9s}'
        print(hdr)
        print('-' * 116)
        for name, env in cases:
            m = run(span, env)
            a = alpha_of(m['daily'], rb) if show_alpha else np.nan
            store.setdefault(name, []).append((m, a))
            line = (f'{name:24s}{m["ann"]:+8.1f}%{m["sharpe"]:7.2f}{m["mdd"]:7.1f}%'
                    f'{m["calmar"]:8.2f}{m["trades"]:7d}{m["win"]:6.1f}%{m["pf"]:6.2f}'
                    f'{m["avg"]:+8.1f}{m["t"]:6.2f}{m["top1"]:9.0f}{m["top10share"]:6.0f}%')
            if show_alpha:
                line += f'{a:+8.1f}%'
            print(line)
    print('\n' + '-' * 116)
    print(f'{"配置":24s}{"三段年化":>28s}{"最差":>9s}{"最差夏普":>10s}{"最大回撤":>10s}'
          f'{"最差PF":>8s}{"最差t":>7s}{"最差胜率":>9s}')
    print('-' * 116)
    for name, _ in cases:
        rs = store[name]
        anns = [m['ann'] for m, a in rs]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:24s}{s:>28s}{min(anns):+8.1f}%{min(m["sharpe"] for m, a in rs):10.2f}'
              f'{max(m["mdd"] for m, a in rs):9.1f}%{min(m["pf"] for m, a in rs):8.2f}'
              f'{min(m["t"] for m, a in rs):7.2f}{min(m["win"] for m, a in rs):8.1f}%')
    return store


def main():
    print(f'移动止盈/保本/追踪均线 检验  N={N}  MA90  币池=universe_live  ATR 10/10')
    block('第一轮: 各部件单独', CASES1)
    block('第二轮: 组合 (含提议全套)', CASES2, show_alpha=False)


if __name__ == '__main__':
    main()
