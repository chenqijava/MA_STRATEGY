"""在指定数据文件上跑主策略回测, 把权益曲线切片到 [START, END] 窗口内报告。

数据文件含预热段 (2026-05-01 起), 但只报告用户关心的 2026-08-08 ~ 2026-09-01 窗口:
- 权益从窗口首根起算, 以窗口前一根权益为基数 (真实区间收益, 不含预热段虚构起点)。
- trades 是浮点盈亏(无时间戳), 无法按窗口归因, 故交易统计报告全区间; 窗口收益由权益切片给出。

用法:
  SPAN=20260501_20260901 START=2026-08-08 END=2026-09-01 N=20 python run_window.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import portfolio_sim as ps

START = os.environ.get('START', '2026-08-08')
END = os.environ.get('END', '2026-09-01')
N = int(os.environ.get('N', '20'))


def main():
    idx, C, H, L, A, S, syms = ps.build_panel()
    curve, trades, avg_held = ps.simulate(N, idx, C, H, L, A, S)
    eq = pd.Series(curve, index=idx)

    # 窗口切片: 取 [START, END) 的权益段
    m0 = (idx >= START) & (idx < END)
    pre = eq[idx < START]
    base = pre.iloc[-1] if len(pre) else ps.INIT
    win = eq[m0]
    if len(win) == 0:
        print(f'窗口 {START}~{END} 内无K线 (数据 {idx[0].date()}~{idx[-1].date()})')
        return
    win_days = (win.index[-1] - win.index[0]).days or 1
    final = win.iloc[-1]

    ret = (final / base - 1) * 100
    ann = ((final / base) ** (365 / win_days) - 1) * 100
    daily = win.resample('1D').last().dropna()
    dr = daily.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = ((win.cummax() - win) / win.cummax()).max() * 100

    arr = np.array(trades, float) if trades else np.array([])
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    pf = wins.sum() / -losses.sum() if len(losses) else float('inf')
    winrate = (arr > 0).mean() * 100 if len(arr) else 0

    print('=' * 84)
    print(f'数据文件 SPAN={os.environ.get("SPAN","")}  币池={len(syms)}  N={N}')
    print(f'数据 {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}根)  ->  报告窗口 {START} ~ {END} ({len(win)}根 / {win_days}天)')
    print(f'参数: MA_SLOW={ps.CFG["MA_SLOW"]} ATR_TP={ps.CFG["ATR_TP"]} ATR_SL={ps.CFG["ATR_SL"]} '
          f'MA_EXIT={int(ps.MA_EXIT)} COOLDOWN_ONCE={int(ps.COOLDOWN_ONCE)} 总敞口={ps.TOTAL_NOTIONAL:.0f}%')
    print('=' * 84)
    print(f'[窗口] 收益 {ret:+.1f}%   年化 {ann:+.1f}%   夏普 {sharpe:.2f}   '
          f'最大回撤 {mdd:.1f}%   Calmar {ann/mdd if mdd>0 else 0:.2f}')
    print(f'        权益 {base:.0f} -> {final:.0f}')
    print(f'[全区间] 交易 {len(arr)}笔  胜率 {winrate:.1f}%  利润因子 {pf:.2f}  均持仓 {avg_held:.1f}币')
    print(f'        单笔最好 {arr.max():+.1f}%  最差 {arr.min():+.1f}%  中位 {np.median(arr):+.2f}%'
          if len(arr) else '[全区间] 无交易')


if __name__ == '__main__':
    main()