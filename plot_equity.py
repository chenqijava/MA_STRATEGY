"""画当前配置的收益曲线 + 回撤曲线。

复用 portfolio_sim 的 simulate(当前配置由环境变量传入, 与实盘一致):
  50币候选 / N=36 / 1.5x敞口 / MA30 / TP4-SL2 / 时停40 / 出场损耗+正funding。
输出 PNG 到 results/equity_curve.png。
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import portfolio_sim as P


def main():
    idx, C, H, L, A, S, syms = P.build_panel()
    N = int(os.environ.get('N', '36'))
    # BTC趋势过滤: BTC_FILTER<1 时启用牛市减仓
    bull = P.btc_bull_mask(idx) if P.BTC_FILTER < 1.0 else None
    curve, trades, avg_held = P.simulate(N, idx, C, H, L, A, S, bull=bull)
    m = P.metrics(curve, trades, idx)
    eq = pd.Series(curve, index=idx)
    dd = (eq.cummax() - eq) / eq.cummax() * 100     # 回撤%

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    # 权益曲线
    ax1.plot(eq.index, eq.values, color='#1f77b4', lw=1.2)
    ax1.axhline(P.INIT, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax1.set_ylabel('权益 (U)')
    ax1.set_title(f'做空组合收益曲线  N={N} 1.5x  '
                  f'年化{m["ann"]:.0f}% 夏普{m["sharpe"]:.2f} 回撤{m["mdd"]:.0f}% '
                  f'Calmar{m["calmar"]:.2f} 胜率{m["win"]:.0f}%',
                  fontsize=11)
    ax1.grid(alpha=0.3)
    # 回撤曲线
    ax2.fill_between(dd.index, -dd.values, 0, color='#d62728', alpha=0.4)
    ax2.set_ylabel('回撤 %')
    ax2.set_xlabel('日期')
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    for a in (ax1, ax2):
        try:
            plt.setp(a.get_xticklabels(), rotation=0)
        except Exception:
            pass
    plt.rcParams['axes.unicode_minus'] = False
    for fn in ('Microsoft YaHei', 'SimHei', 'DejaVu Sans'):
        try:
            plt.rcParams['font.sans-serif'] = [fn]
            break
        except Exception:
            continue
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'equity_curve.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f'已保存 {out}')
    print(f'期末权益 {curve[-1]:.0f}U (初始{P.INIT:.0f})  年化{m["ann"]:.1f}% '
          f'夏普{m["sharpe"]:.2f} 最大回撤{m["mdd"]:.1f}% 交易{m["trades"]}笔')


if __name__ == '__main__':
    main()
