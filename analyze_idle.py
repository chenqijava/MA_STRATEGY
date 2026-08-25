"""空仓期分析: 最长多久没有任何持仓 / 多久没有新交易 / 多久不创新高。

为什么要看这个: 当前配置(MA90 N20)均持仓只有 3.0~3.5 币, 信号稀疏。实盘最难受的
不是回撤, 而是"长时间什么都不发生"—— 人会怀疑程序坏了、参数错了, 从而手动干预。
提前知道正常的空仓期有多长, 才能区分"策略本来就这样"和"真的出故障了"。

三个口径:
  完全空仓   一根 K 线上持仓数=0。最直观的"账户里什么都没有"。
  无新开仓   连续多少根没有新入场(可能仍持有旧仓)。反映信号枯竭程度。
  不创新高   权益曲线距上一次峰值多久 —— 这是心理上最难熬的指标。

用法:
  python analyze_idle.py            # 用 .env 的实盘配置
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import importlib


def runs_of_true(mask):
    """返回 [(起始下标, 长度)] —— mask 中连续 True 的段。"""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(mask) - start))
    return out


def describe(runs, idx, label, bars_per_day=6):
    """打印一组连续段的统计 + 最长的几段。"""
    if not runs:
        print(f'  {label}: 无')
        return
    lens = np.array([n for _, n in runs])
    print(f'  {label}: 共 {len(runs)} 段   '
          f'最长 {lens.max()} 根 ({lens.max()/bars_per_day:.1f}天)   '
          f'中位 {np.median(lens):.0f} 根   '
          f'均值 {lens.mean():.1f} 根   '
          f'占总时长 {lens.sum()/len(idx)*100:.0f}%')
    top = sorted(runs, key=lambda x: -x[1])[:3]
    for s, n in top:
        print(f'      {idx[s].date()} ~ {idx[min(s+n-1, len(idx)-1)].date()}  '
              f'{n} 根 = {n/bars_per_day:.1f} 天')


def main():
    import live_ma_short as L
    N = L.MAX_POSITIONS
    TOTAL = L.MAX_TOTAL_NOTIONAL_PCT * L.LEVERAGE
    os.environ.update(SIZING=L.SIZING, MA_EXIT='1' if L.MA_EXIT else '0',
                      COOLDOWN_ONCE='1' if L.COOLDOWN_ONCE else '0',
                      MA_SLOW=str(L.CFG['MA_SLOW']), TOTAL=f'{TOTAL:.4f}',
                      MAX_BARS=str(L.CFG['MAX_BARS']), UNIVERSE='universe_live.txt',
                      LONG='0', SHORT='1')
    for k in ('REGIME', 'SHORT_BULL', 'SHORT_BEAR', 'BTC_MA', 'BTC_SLOPE'):
        os.environ.pop(k, None)
    import analyze_long as al
    import portfolio_sim as ps
    import test_alpha_beta as tb

    print('=' * 86)
    print(f'空仓期分析  MA{L.CFG["MA_SLOW"]} N={N} {L.SIZING} 敞口{TOTAL:.2f}x  (4h周期, 每日6根)')
    print('=' * 86)
    for span, lab in tb.SPANS:
        os.environ['SPAN'] = span
        importlib.reload(al)
        importlib.reload(ps)
        idx, C, H, Lo, A, S, syms = ps.build_panel()
        curve, trades, ah = ps.simulate(N, idx, C, H, Lo, A, S, bull=None)
        held = ps._HELD_TRACE
        # 新开仓: 持仓数比上一根增加, 说明本根有入场
        opened = np.diff(held, prepend=held[0]) > 0
        eq = pd.Series(curve, index=idx)
        peak = eq.cummax()
        at_peak = (eq >= peak * 0.9999).values     # 允许浮点误差

        print(f'\n【{lab}】 {idx[0].date()} ~ {idx[-1].date()}  {len(idx)}根  均持仓{ah:.1f}币')
        print(f'  空仓根数占比 {(held == 0).mean()*100:.0f}%   '
              f'持仓1币以下占比 {(held <= 1).mean()*100:.0f}%')
        describe(runs_of_true(held == 0), idx, '完全空仓')
        describe(runs_of_true(~opened), idx, '无新开仓')
        describe(runs_of_true(~at_peak), idx, '不创新高')


if __name__ == '__main__':
    main()
