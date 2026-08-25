"""跑单段连续回测并输出年化/夏普/回撤/分年 (配置由环境变量给)。

与 run_oos2023.py 的区别: 不做 BTC 过滤对比, 只跑一档, 额外输出分年与滚动最差12月,
用于"这一整段的真年化是多少"这类问题。

用法: SPAN=20230101_20260801 UNIVERSE=universe_live.txt TOTAL=1.5 N=20 \
      MA_SLOW=90 MA_EXIT=1 COOLDOWN_ONCE=1 ATR_TP=10 ATR_SL=10 python run_span.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import portfolio_sim as ps
from analyze_long import SPAN

N = int(os.environ.get('N', '20'))


def main():
    idx, C, H, L, A, S, syms = ps.build_panel()
    curve, trades, avg_held = ps.simulate(N, idx, C, H, L, A, S)
    m = ps.metrics(curve, trades, idx)
    tr = np.array(trades, float)
    w, l = tr[tr > 0].sum(), -tr[tr < 0].sum()
    pf = w / l if l > 0 else float('inf')
    eq = pd.Series(curve, index=idx)
    days = (idx[-1] - idx[0]).days

    print('=' * 84)
    print(f'区间 {idx[0].date()} ~ {idx[-1].date()}  ({len(idx)}根 / {days}天 / {days/365:.2f}年)')
    print(f'币池={len(syms)}  N={N}  总敞口={ps.TOTAL_NOTIONAL:.1f}x  MA_SLOW={ps.CFG["MA_SLOW"]}  '
          f'MA出场={int(ps.MA_EXIT)}  单窗一次={int(ps.COOLDOWN_ONCE)}  '
          f'ATR {ps.CFG["ATR_TP"]:.0f}/{ps.CFG["ATR_SL"]:.0f}  时停={ps.MAX_BARS}')
    print('=' * 84)
    print(f'  总收益 {m["ret"]:+.1f}%   年化 {m["ann"]:+.1f}%   夏普 {m["sharpe"]:.2f}   '
          f'最大回撤 {m["mdd"]:.1f}%   Calmar {m["calmar"]:.2f}')
    print(f'  交易 {m["trades"]}笔   胜率 {m["win"]:.1f}%   利润因子 {pf:.2f}   均持仓 {avg_held:.1f}币')
    print(f'  权益 {ps.INIT:.0f} -> {curve[-1]:.0f}')

    print('\n分年收益:')
    for y, g in eq.groupby(eq.index.year):
        # 用上一年末权益作起点, 否则每年首根被当成起点会漏掉跨年那一段
        prev = eq[eq.index < g.index[0]]
        base = prev.iloc[-1] if len(prev) else ps.INIT
        print(f'  {y}: {(g.iloc[-1]/base - 1)*100:+7.1f}%   (年末权益 {g.iloc[-1]:.0f})')

    d = eq.resample('1D').last().dropna()
    if len(d) > 370:
        r12 = (d / d.shift(365) - 1).dropna() * 100
        print(f'\n滚动12个月收益: 最差 {r12.min():+.1f}%  中位 {r12.median():+.1f}%  '
              f'最好 {r12.max():+.1f}%  为负的占比 {(r12 < 0).mean()*100:.0f}%')


if __name__ == '__main__':
    main()
