"""2023-01-01 ~ 2025-01-01 真样本外检验。

这段数据完全没参与过任何参数选择 (所有寻优都在 2025-01~2026-08 上做的),
且覆盖了 2023 震荡 + 2024 牛市, 是对策略最严的一次考验。

用法:
  SPAN=20230101_20250101 UNIVERSE=universe_live.txt TOTAL=1.5 MAX_BARS=40 \
  BTC_FILTER=0.5 BTC_MA=35 MA_SLOW=30 python run_oos2023.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import portfolio_sim as ps
from analyze_long import SPAN

N = int(os.environ.get('N', '36'))


def yearly(curve, idx):
    eq = pd.Series(curve, index=idx)
    out = []
    for y, g in eq.groupby(eq.index.year):
        out.append((y, (g.iloc[-1] / g.iloc[0] - 1) * 100))
    return out


def main():
    idx, C, H, L, A, S, syms = ps.build_panel()
    bull = ps.btc_bull_mask(idx) if ps.BTC_FILTER != 1.0 else None
    print('=' * 78)
    print(f'样本外检验  区间={SPAN or "最新"}  {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}根)')
    print(f'币池={len(syms)}  N={N}  总敞口={ps.TOTAL_NOTIONAL:.1f}x  MAX_BARS={ps.MAX_BARS}  '
          f'BTC_FILTER={ps.BTC_FILTER} MA{ps.BTC_MA}')
    print('=' * 78)

    for label, b in (('含BTC牛市过滤', bull), ('不含BTC过滤', None)):
        curve, trades, avg_held = ps.simulate(N, idx, C, H, L, A, S, bull=b)
        m = ps.metrics(curve, trades, idx)
        print(f'\n[{label}]')
        print(f'  总收益 {m["ret"]:+.1f}%   年化 {m["ann"]:+.1f}%   夏普 {m["sharpe"]:.2f}   '
              f'最大回撤 {m["mdd"]:.1f}%   Calmar {m["calmar"]:.2f}')
        print(f'  交易 {m["trades"]}笔   胜率 {m["win"]:.1f}%   均持仓 {avg_held:.1f}币')
        print('  分年: ' + '  '.join(f'{y} {r:+.1f}%' for y, r in yearly(curve, idx)))

    # 牛熊分段: 用 BTC 日线 MA35 判定
    if bull is not None:
        curve, _, _ = ps.simulate(N, idx, C, H, L, A, S, bull=bull)
        eq = pd.Series(curve, index=idx)
        r = eq.pct_change().fillna(0).values
        for nm, mask in (('BTC牛市段', bull), ('BTC熊/震荡段', ~bull)):
            n = mask.sum()
            if n < 10:
                continue
            seg = r[mask]
            cum = np.prod(1 + seg) - 1
            days = n / 6
            ann = (1 + cum) ** (365 / days) - 1 if cum > -1 else -1
            print(f'\n  {nm}: 占比 {n/len(idx)*100:.0f}%  ({days:.0f}天)  '
                  f'累计 {cum*100:+.1f}%  折年化 {ann*100:+.1f}%')


if __name__ == '__main__':
    main()
