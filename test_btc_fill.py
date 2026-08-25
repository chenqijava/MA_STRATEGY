import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import importlib
import numpy as np
import pandas as pd
import analyze_long as al
import portfolio_sim as ps

SPANS = [('20210101_20230101', '2021-01~2023-01 (BTC -54%)'),
         ('20230101_20250101', '2023-01~2025-01 (BTC +342%)'),
         ('20250101_20260801', '2025-01~2026-08 (BTC -37%)')]
N = 20
BASE_ENV = dict(MA_SLOW='90', MA_EXIT='1', COOLDOWN_ONCE='1',
                ATR_TP='10', ATR_SL='10', SIZING='notional', CROSS_DELAY='6',
                TOUCH_ATR='0.5', MAX_BARS='40', UNIVERSE='universe_live.txt', TOTAL='1.5')
_RESET = ('BTC_FILL', 'BTC_FILL_MA_F', 'BTC_FILL_MA_S', 'BTC_FILL_ONCE', 'BTC_FILL_TRAIL_ATR', 'BTC_FILL_TRAIL_PCT', 'BTC_FILL_TRAIL_ARM', 'BTC_FILL_ATR_LEN', 'BTC_FILL_SRC')

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
    # 统计 btc_fill 交易的贡献
    btc_trades = [t for t in ps._TRADE_TRACE if t[2] == 'btc_fill']
    m['btc_n'] = len(btc_trades)
    m['btc_pnl'] = sum(t[0] for t in btc_trades) * 100
    return m

cases = [
    ('基线 (无BTC填仓)', dict()),
    # 窗口内只认金叉一次 (BTC_FILL_ONCE=1, 默认新行为)
    ('金叉一次 3xATR20',       dict(BTC_FILL='1', BTC_FILL_TRAIL_ATR='3.0')),
    ('金叉一次 4xATR20',       dict(BTC_FILL='1', BTC_FILL_TRAIL_ATR='4.0')),
    ('金叉一次 5xATR20',       dict(BTC_FILL='1', BTC_FILL_TRAIL_ATR='5.0')),
    ('金叉一次 3%trail',       dict(BTC_FILL='1', BTC_FILL_TRAIL_ATR='0', BTC_FILL_TRAIL_PCT='0.03')),
    ('金叉一次 3xATR20 arm2',  dict(BTC_FILL='1', BTC_FILL_TRAIL_ATR='3.0', BTC_FILL_TRAIL_ARM='2')),
    # 对照: 旧水平条件 (BTC_FILL_ONCE=0, 全程开多)
    ('水平条件 3xATR20',       dict(BTC_FILL='1', BTC_FILL_ONCE='0', BTC_FILL_TRAIL_ATR='3.0')),
    ('水平条件 5xATR20',       dict(BTC_FILL='1', BTC_FILL_ONCE='0', BTC_FILL_TRAIL_ATR='5.0')),
    ('水平条件 3%trail',       dict(BTC_FILL='1', BTC_FILL_ONCE='0', BTC_FILL_TRAIL_ATR='0', BTC_FILL_TRAIL_PCT='0.03')),
]
store = {}
for span, lab in SPANS:
    print(f'\n### {lab}')
    hdr = f'{"配置":22s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"笔数":>6s}{"胜率":>6s}{"PF":>6s}{"BTC笔数":>8s}{"BTC收益%":>9s}'
    print(hdr); print('-'*95)
    for name, env in cases:
        m = run(span, env)
        store.setdefault(name, []).append(m)
        print(f'{name:22s}{m["ann"]:+8.1f}%{m["sharpe"]:7.2f}{m["mdd"]:7.1f}%{m["trades"]:6d}'
              f'{m["win"]:6.1f}%{m["pf"]:6.2f}{m["btc_n"]:8d}{m["btc_pnl"]:+9.1f}%')
print('\n' + '-'*95)
print(f'{"配置":22s}{"三段年化":>30s}{"最差":>9s}')
for name, _ in cases:
    anns = [m['ann'] for m in store[name]]
    print(f'{name:22s}' + '  '.join(f'{v:+7.1f}%' for v in anns) + f'  {min(anns):+8.1f}%')
