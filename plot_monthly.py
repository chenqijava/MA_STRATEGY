"""月度收益表 + 收益曲线 (连续区间, 默认 2023-01~2026-08)。

用连续拼接的数据跑一次 simulate, 再把盯市权益重采样成月度。
月收益 = 本月末权益 / 上月末权益 - 1 (不是月内首根到末根 —— 那样会漏掉跨月的一段)。

输出:
  · 控制台: 年×月矩阵 + 月度统计
  · results/monthly_2023_2026.png: 权益曲线(对数) + 回撤 + 月收益柱状 + 月收益分布

用法: SPAN=20230101_20260801 UNIVERSE=universe_live.txt TOTAL=1.5 N=20 \
      MA_SLOW=90 MA_EXIT=1 COOLDOWN_ONCE=1 ATR_TP=10 ATR_SL=10 python plot_monthly.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import portfolio_sim as ps

N = int(os.environ.get('N', '20'))
MONTHS = ['1月', '2月', '3月', '4月', '5月', '6月',
          '7月', '8月', '9月', '10月', '11月', '12月']


def _font():
    plt.rcParams['axes.unicode_minus'] = False
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for fn in ('Microsoft YaHei', 'SimHei', 'DengXian', 'SimSun'):
        if fn in have:
            plt.rcParams['font.sans-serif'] = [fn]
            return
    print('[warn] 未找到中文字体, 图中中文可能显示为方框')


def monthly_returns(eq):
    """月度收益%: 用月末权益环比。首月以区间起始权益为基准。"""
    me = eq.resample('ME').last().dropna()
    base = pd.concat([pd.Series([ps.INIT], index=[me.index[0]]), me]).iloc[:-1]
    base.index = me.index
    return (me / base.values - 1) * 100


def main():
    idx, C, H, L, A, S, syms = ps.build_panel()
    curve, trades, avg_held = ps.simulate(N, idx, C, H, L, A, S)
    m = ps.metrics(curve, trades, idx)
    eq = pd.Series(curve, index=idx)
    mr = monthly_returns(eq)

    print('=' * 92)
    print(f'月度收益  {idx[0].date()} ~ {idx[-1].date()}   币池={len(syms)}  N={N}  '
          f'总敞口={ps.TOTAL_NOTIONAL:.1f}x  MA{ps.CFG["MA_SLOW"]}')
    print(f'年化 {m["ann"]:+.1f}%  夏普 {m["sharpe"]:.2f}  最大回撤 {m["mdd"]:.1f}%  '
          f'总收益 {m["ret"]:+.1f}%  权益 {ps.INIT:.0f} -> {curve[-1]:.0f}')
    print('=' * 92)

    # 年 × 月 矩阵
    tab = pd.DataFrame(index=sorted(mr.index.year.unique()), columns=MONTHS, dtype=float)
    for t, v in mr.items():
        tab.loc[t.year, MONTHS[t.month - 1]] = v
    tab['全年'] = [(eq[eq.index.year == y].iloc[-1] /
                  (eq[eq.index.year < y].iloc[-1] if (eq.index.year < y).any() else ps.INIT) - 1) * 100
                 for y in tab.index]
    fmt = tab.applymap(lambda v: '' if pd.isna(v) else f'{v:+.1f}')
    print(fmt.to_string())

    pos, neg = mr[mr > 0], mr[mr <= 0]
    print(f'\n月度统计 ({len(mr)}个月): 上涨 {len(pos)} / 下跌 {len(neg)}  '
          f'胜月率 {len(pos)/len(mr)*100:.0f}%')
    print(f'  平均 {mr.mean():+.1f}%   中位 {mr.median():+.1f}%   标准差 {mr.std():.1f}%')
    print(f'  最好 {mr.max():+.1f}% ({mr.idxmax():%Y-%m})   '
          f'最差 {mr.min():+.1f}% ({mr.idxmin():%Y-%m})')
    print(f'  上涨月均 {pos.mean():+.1f}%   下跌月均 {neg.mean():+.1f}%  '
          f'(月度盈亏比 {abs(pos.mean()/neg.mean()):.2f})')
    # 连续亏损月 —— 实盘心理承受的关键
    streak = best = 0
    for v in mr.values:
        streak = streak + 1 if v <= 0 else 0
        best = max(best, streak)
    print(f'  最长连续亏损 {best} 个月')

    # ---------------- 画图 ----------------
    _font()
    dd = (eq.cummax() - eq) / eq.cummax() * 100
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1.6, 1.6], hspace=0.35, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq.index, eq.values, color='#1f77b4', lw=1.3)
    ax1.axhline(ps.INIT, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax1.set_yscale('log')          # 对数轴: 3.5年涨3.8倍, 线性轴会把前期压平看不出结构
    ax1.set_ylabel('权益 (U, 对数轴)')
    ax1.set_title(f'做空组合 {idx[0]:%Y-%m}~{idx[-1]:%Y-%m}  '
                  f'年化{m["ann"]:.1f}%  夏普{m["sharpe"]:.2f}  回撤{m["mdd"]:.1f}%  '
                  f'Calmar{m["calmar"]:.2f}  {m["trades"]}笔  胜率{m["win"]:.1f}%', fontsize=12)
    ax1.grid(alpha=0.3, which='both')

    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax2.fill_between(dd.index, -dd.values, 0, color='#d62728', alpha=0.4)
    ax2.set_ylabel('回撤 %')
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    ax3 = fig.add_subplot(gs[2, :])
    cols = ['#2ca02c' if v > 0 else '#d62728' for v in mr.values]
    ax3.bar(range(len(mr)), mr.values, color=cols, alpha=0.85)
    ax3.axhline(0, color='black', lw=0.8)
    ax3.set_ylabel('月收益 %')
    ax3.set_xticks(range(0, len(mr), 3))
    ax3.set_xticklabels([f'{t:%Y-%m}' for t in mr.index[::3]], rotation=45, ha='right')
    ax3.grid(alpha=0.3, axis='y')
    ax3.set_title('月度收益', fontsize=10)

    ax4 = fig.add_subplot(gs[3, 0])
    ax4.hist(mr.values, bins=18, color='#1f77b4', alpha=0.75, edgecolor='white')
    ax4.axvline(0, color='black', lw=1)
    ax4.axvline(mr.mean(), color='#d62728', ls='--', lw=1.2,
                label=f'均值 {mr.mean():+.1f}%')
    ax4.set_xlabel('月收益 %')
    ax4.set_ylabel('月数')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3, axis='y')
    ax4.set_title('月收益分布', fontsize=10)

    # 滚动12个月: 判断"持有一年会不会亏"
    ax5 = fig.add_subplot(gs[3, 1])
    d = eq.resample('D').last().dropna()
    r12 = (d / d.shift(365) - 1).dropna() * 100
    ax5.plot(r12.index, r12.values, color='#ff7f0e', lw=1.2)
    ax5.axhline(0, color='black', lw=0.8)
    ax5.fill_between(r12.index, r12.values, 0, where=(r12.values < 0),
                     color='#d62728', alpha=0.3)
    ax5.set_ylabel('滚动12月收益 %')
    ax5.grid(alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    ax5.set_title(f'滚动12个月 (为负占{(r12 < 0).mean()*100:.0f}%)', fontsize=10)

    out = os.path.join(os.path.dirname(__file__), 'results', 'monthly_2023_2026.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'\n已保存 {out}')


if __name__ == '__main__':
    main()
