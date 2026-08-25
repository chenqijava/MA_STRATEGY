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
_FILL_PFX = ('BTC_FILL', 'ETH_FILL')
_RESET = _FILL_PFX + tuple(p + s for p in _FILL_PFX for s in
                           ('_MA_F', '_MA_S', '_ONCE', '_TRAIL_ATR', '_TRAIL_PCT',
                            '_TRAIL_ARM', '_ATR_LEN', '_SRC'))

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
    for leg in ('btc', 'eth'):
        leg_trades = [t for t in ps._TRADE_TRACE if t[2] == f'{leg}_fill']
        m[f'{leg.upper()}_n'] = len(leg_trades)
        m[f'{leg.upper()}_pnl'] = sum(t[0] for t in leg_trades) * 100
    return m

cases = [
    ('基线 (无填仓)',           dict()),
    ('ETH填仓 20/85高中4ATR18', dict(ETH_FILL='1')),
    ('BTC填仓 10/120 3ATR20',   dict(BTC_FILL='1')),
    ('BTC+ETH 双填仓',          dict(BTC_FILL='1', ETH_FILL='1')),
]
store = {}
for span, lab in SPANS:
    print(f'\n### {lab}')
    hdr = f'{"配置":24s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}{"笔数":>6s}{"胜率":>6s}{"PF":>6s}' \
          f'{"ETH笔":>6s}{"ETH收益%":>9s}{"BTC笔":>6s}{"BTC收益%":>9s}'
    print(hdr); print('-' * 100)
    for name, env in cases:
        m = run(span, env)
        store.setdefault(name, []).append(m)
        print(f'{name:24s}{m["ann"]:+8.1f}%{m["sharpe"]:7.2f}{m["mdd"]:7.1f}%{m["trades"]:6d}'
              f'{m["win"]:6.1f}%{m["pf"]:6.2f}{m["ETH_n"]:6d}{m["ETH_pnl"]:+9.1f}%'
              f'{m["BTC_n"]:6d}{m["BTC_pnl"]:+9.1f}%')
print('\n' + '-' * 100)
print(f'{"配置":24s}{"三段年化":>30s}{"最差":>9s}')
for name, _ in cases:
    anns = [m['ann'] for m in store[name]]
    print(f'{name:24s}' + '  '.join(f'{v:+7.1f}%' for v in anns) + f'  {min(anns):+8.1f}%')