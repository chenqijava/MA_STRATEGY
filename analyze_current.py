"""当前策略 (配置 H) 综合分析。

输出:
  1. 全段指标 (年化/夏普/回撤/Calmar/交易数/胜率/利润因子/均持仓)
  2. 分月收益 + 滚动最差 12 月
  3. 出场原因分布 (ma_exit / time_stop / atr_tp / atr_sl)
  4. 最近 15 笔交易明细
  5. 持仓特征 (空仓比例 / 最长空仓 / 均持仓)

用法: SPAN=20260501_20260901 N=20 ... python analyze_current.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import portfolio_sim as ps

N = int(os.environ.get('N', '20'))


def pf(x):
    if x == 0:
        return '0'
    a = abs(x)
    if a >= 100:
        return f'{x:.2f}'
    if a >= 1:
        return f'{x:.4f}'
    s = f'{x:.10f}'.rstrip('0')
    return s


def main():
    idx, C, H, L, A, S, syms = ps.build_panel()
    curve, trades, avg_held = ps.simulate(N, idx, C, H, L, A, S)
    m = ps.metrics(curve, trades, idx)
    tr = np.array(trades, float)
    w, l = tr[tr > 0].sum(), -tr[tr < 0].sum()
    pf_ = w / l if l > 0 else float('inf')
    eq = pd.Series(curve, index=idx)
    days = (idx[-1] - idx[0]).days

    print('=' * 80)
    print(f'配置 H 综合分析  ({len(syms)}币  N={N}  总敞口={ps.TOTAL_NOTIONAL:.1f}x)')
    print(f'参数: MA_SLOW={ps.CFG["MA_SLOW"]}  MA_EXIT={int(ps.MA_EXIT)}  '
          f'COOLDOWN_ONCE={int(ps.COOLDOWN_ONCE)}  ATR {ps.CFG["ATR_TP"]:.0f}/{ps.CFG["ATR_SL"]:.0f}  '
          f'时停={ps.MAX_BARS}')
    print(f'数据 {idx[0].date()} ~ {idx[-1].date()}  ({len(idx)}根 / {days}天 / {days/365:.2f}年)')
    print('=' * 80)

    # 1. 全段指标
    print('\n[1] 全段指标')
    print(f'  总收益 {m["ret"]:+.1f}%   年化 {m["ann"]:+.1f}%   夏普 {m["sharpe"]:.2f}   '
          f'最大回撤 {m["mdd"]:.1f}%   Calmar {m["calmar"]:.2f}')
    print(f'  交易 {m["trades"]}笔   胜率 {m["win"]:.1f}%   利润因子 {pf_:.2f}   均持仓 {avg_held:.1f}币')
    print(f'  权益 {ps.INIT:.0f} -> {curve[-1]:.0f}')

    # 2. 分月收益
    print('\n[2] 分月收益')
    eq_m = eq.resample('ME').last()
    ret_m = eq_m.pct_change().dropna() * 100
    for d, r in ret_m.items():
        bar = '█' * max(1, int(abs(r) * 2))
        sign = '+' if r >= 0 else '-'
        print(f'  {d:%Y-%m}  {sign}{abs(r):6.2f}%  {bar}')
    # 滚动最差 12 月
    if len(eq_m) >= 13:
        roll_12 = eq_m.pct_change(12).dropna() * 100
        worst_12 = roll_12.min()
        worst_12_end = roll_12.idxmin()
        print(f'  滚动最差 12 月: {worst_12:+.1f}% (截至 {worst_12_end:%Y-%m})')

    # 3. 出场原因分布
    print('\n[3] 出场原因分布')
    det = ps._TRADE_DETAILS or []
    from collections import Counter
    why_counts = Counter(t['why'] for t in det)
    why_pnl = {}
    for t in det:
        why_pnl.setdefault(t['why'], []).append(t['pnl'])
    for why, cnt in why_counts.most_common():
        pnls = why_pnl[why]
        avg_pnl = np.mean(pnls)
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f'  {why:<15} {cnt:>4}笔  ({cnt/len(det)*100:4.1f}%)  '
              f'均盈亏 {avg_pnl:+.1f}U  胜率 {win_rate:.0f}%')

    # 4. 最近 15 笔交易明细
    print('\n[4] 最近 15 笔交易明细')
    det_s = sorted(det, key=lambda t: t['exit_time'])
    n = min(15, len(det_s))
    print(f'  {"出场时间":<17}{"币":<7}{"方向":<3}{"入场价":>11}{"出场价":>11}'
          f'{"入场时间":<17}{"根数":>4}{"原因":<11}{"单笔收益率":>10}')
    for t in det_s[-n:]:
        d = '多' if t['dir'] == 1 else '空'
        et = str(t['entry_time'])[:16]
        xt = str(t['exit_time'])[:16]
        ret = (t['dir'] * (t['exit'] / t['entry'] - 1) - ps.FEE) * 100
        print(f'  {xt:<17}{t["sym"]:<7}{d:<3}{pf(t["entry"]):>11}{pf(t["exit"]):>11}'
              f'{et:<17}{t["bars"]:>4}{t["why"]:<11}{ret:>9.2f}%')

    # 5. 持仓特征
    print('\n[5] 持仓特征')
    held = ps._HELD_TRACE
    idle_pct = (held == 0).mean() * 100
    # 最长连续空仓
    idle_runs = []
    run = 0
    for h in held:
        if h == 0:
            run += 1
        else:
            if run:
                idle_runs.append(run)
            run = 0
    if run:
        idle_runs.append(run)
    max_idle = max(idle_runs) if idle_runs else 0
    print(f'  空仓比例: {idle_pct:.1f}%  最长连续空仓: {max_idle}根 ({max_idle*4/24:.1f}天)')
    print(f'  均持仓: {avg_held:.1f}币  最大持仓: {int(held.max())}币')
    held_last = int(held[-1])
    if held_last:
        print(f'  注意: 数据末尾仍有 {held_last} 个币在持仓 (未平仓, 盈亏在窗口外结算)')

    # 6. 单笔极值
    print('\n[6] 单笔极值')
    if tr.size:
        print(f'  最好单笔: {tr.max():+.1f}U  最差单笔: {tr.min():+.1f}U  中位数: {np.median(tr):+.1f}U')
        print(f'  均笔盈亏: {tr.mean():+.1f}U  笔盈亏标准差: {tr.std():.1f}U')


if __name__ == '__main__':
    main()