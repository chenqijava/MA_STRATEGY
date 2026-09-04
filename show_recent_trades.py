"""打印最近几笔交易的明细 (用 _TRADE_DETAILS 旁路, 由 simulate 填充)。

单笔收益率 = 方向×(出场/入场 - 1) - 费率, 与名义无关, 跨币可比。
用法: SPAN=20260501_20260901 N=20 ... python show_recent_trades.py [笔数]
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import portfolio_sim as ps

K = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def pf(x):
    """自适应价格格式: 大价 4位小数, 小价(如 SHIB 0.00002)保留足够有效位。"""
    if x == 0:
        return '0'
    a = abs(x)
    if a >= 100:
        return f'{x:.2f}'
    if a >= 1:
        return f'{x:.4f}'
    # 找到第一个非零小数的位数, 保留其后3位
    s = f'{x:.10f}'.rstrip('0')
    return s


def main():
    idx, C, H, L, A, S, syms = ps.build_panel()
    curve, trades, avg_held = ps.simulate(20, idx, C, H, L, A, S)
    det = ps._TRADE_DETAILS or []
    if not det:
        print('无交易明细')
        return
    # 按出场时间排序 (已是时间序, 这里显式排稳)
    det = sorted(det, key=lambda t: t['exit_time'])
    n = min(K, len(det))
    print(f'共 {len(det)} 笔交易 (数据 {idx[0].date()}~{idx[-1].date()}), 显示最近 {n} 笔:')
    print(f'{"出场时间":<17}{"币":<7}{"方向":<3}{"入场价":>11}{"出场价":>11}'
          f'{"入场时间":<17}{"根数":>4}{"原因":<11}{"单笔收益率":>10}')
    for t in det[-n:]:
        d = '多' if t['dir'] == 1 else '空'
        et = str(t['entry_time'])[:16]
        xt = str(t['exit_time'])[:16]
        ret = (t['dir'] * (t['exit'] / t['entry'] - 1) - ps.FEE) * 100
        print(f'{xt:<17}{t["sym"]:<7}{d:<3}{pf(t["entry"]):>11}{pf(t["exit"]):>11}'
              f'{et:<17}{t["bars"]:>4}{t["why"]:<11}{ret:>9.2f}%')

    # 窗口末尾持仓状态: 最后一天还在持仓的币 (从 _HELD_TRACE 或复跑拿不到明细,
    # 这里提示数据末尾若仍有持仓, 会在窗口外继续, 列表里看不到这些"未平"笔)
    held_last = int(ps._HELD_TRACE[-1])
    if held_last:
        print(f'\n注意: 数据末尾 (idx={idx[-1].date()}) 仍有 {held_last} 个币在持仓, '
              f'未平仓, 故不在上面列表中; 它们的盈亏在窗口外结算。')


if __name__ == '__main__':
    main()