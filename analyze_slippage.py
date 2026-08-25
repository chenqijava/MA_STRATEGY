"""统计实盘/模拟盘实际滑点, 与回测假设对比。

读 results/fills.csv (由 live_ma_short 的 log_fill 写入), 按成交类型和币种分组:
  - entry: 入场滑点 (回测假设 SLIPPAGE=0.02% = 2bp)
  - sl/tp: 出场滑点 (组合模拟假设 EXIT_SLIP=0.05% = 5bp, sl另有击穿)
约定: 正 bp = 不利成交, 负 bp = 有利。

重点分"大盘币 vs 小币"看——小币滑点若远超假设, 直接下调预期年化。

用法: python analyze_slippage.py [fills.csv路径]
"""
import os, sys
import numpy as np
import pandas as pd

if sys.stdout.encoding and not sys.stdout.encoding.lower().startswith('utf'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BIG = {'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'LTC', 'LINK', 'AVAX', 'BCH'}  # 大盘/高流动
# 回测假设(bp)。entry 用 ma_pullback.SLIPPAGE=2bp; 各类出场用 portfolio_sim.EXIT_SLIP=5bp。
# sl 还额外建模了击穿(SLIP_THRU=0.3, 成交价在 stop 与当根极值之间), 所以 sl 的实测值
# 超过 5bp 未必意味着假设错 —— 要和"止损时段的极值距离"一起看。
ASSUME = {'entry': 2.0, 'sl': 5.0, 'tp': 5.0, 'time_stop': 5.0, 'ma_exit': 5.0}


def main():
    fp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), 'results', 'fills.csv')
    if not os.path.exists(fp):
        print(f'未找到成交记录 {fp}')
        print('先在模拟盘/实盘跑 live_ma_short.py (DRY_RUN=false), 产生成交后再统计。')
        return
    d = pd.read_csv(fp)
    if len(d) == 0:
        print('成交记录为空'); return
    d['组'] = np.where(d['sym'].isin(BIG), '大盘币', '小币')
    print(f'成交记录 {len(d)} 笔  区间 {d.time.iloc[0][:10]} ~ {d.time.iloc[-1][:10]}')
    print('约定: 正bp=不利成交(做空亏), 负bp=有利')
    # 样本量提醒: 策略年均约550笔, 一个月约45笔。不足30笔时任何均值都不足以下结论。
    if len(d) < 30:
        print(f'⚠ 仅{len(d)}笔, 样本太小不足以判读 (建议≥30笔; 按年均550笔算约需20天)')
    # |slip|>30bp 的 entry 通常不是真滑点, 而是"参考价与下单时点不同根K线"造成的:
    # 手动测试(test_open_short)用最后一根已收盘K线的收盘价做参考, 而下单在进行中的那根,
    # 两者最多隔4小时。实盘里信号与下单在同一根收盘后秒级发生, 不会有这种偏离。
    susp = d[(d['kind'] == 'entry') & (d['slip_bp'].abs() > 30)]
    if len(susp):
        print(f'⚠ {len(susp)}笔 entry 的|滑点|>30bp, 疑为手动测试的时间差而非真实滑点 '
              f'(实盘信号与下单同根收盘、秒级间隔)。判读时应剔除:')
        for _, r in susp.iterrows():
            print(f'    {r["time"][:19]} {r["sym"]} {r["slip_bp"]:+.1f}bp')
    # lag_s = 成交距所属K线开始的秒数。参考价是上一根的收盘价, lag 越大它越过期,
    # 记出来的就越不是滑点而是延迟成本。实测 13s→0.0bp / 51s→10.8bp / 6489s→63.7bp。
    if 'lag_s' in d.columns:
        far = d[d['lag_s'] > 900]
        if len(far):
            print(f'⚠ {len(far)}笔 成交距K线开始>15分钟, 参考价已过期(属延迟成本非滑点), 已剔除:')
            for _, r in far.iterrows():
                print(f'    {r["time"][:19]} {r["kind"]} {r["sym"]} lag={int(r["lag_s"])//60}分 '
                      f'{r["slip_bp"]:+.1f}bp')
            d = d.drop(far.index)
    else:
        print('· 该文件无 lag_s 列(2026-08-08 前的旧格式), 无法自动分离延迟成本, '
              '判读时请人工核对成交时刻是否贴近4h边界')
    # 2026-08-08 之前的 time_stop/ma_exit 记录, ref_px 误用了**开仓价**而非触发价,
    # 那样算出来的是"这笔交易的盈亏"不是滑点(AVAX 记了 +174bp, 按触发价重算实为 +40bp)。
    # 已在 live_ma_short 修复(close_position 增 ref_px 参数), 但旧记录无法追溯修正。
    bad = d[(d['kind'].isin(['time_stop', 'ma_exit'])) & (d['time'] < '2026-08-09')]
    if len(bad):
        print(f'⚠ {len(bad)}笔 time_stop/ma_exit 来自 ref_px 修复前(2026-08-09之前), '
              f'其值是交易盈亏而非滑点, 已从统计中剔除:')
        for _, r in bad.iterrows():
            print(f'    {r["time"][:19]} {r["kind"]} {r["sym"]} {r["slip_bp"]:+.1f}bp')
        d = d.drop(bad.index)
        if len(d) == 0:
            print('\n剔除后无有效记录'); return
    print()

    print('=== 按成交类型 (bp) ===')
    print(f'{"类型":<10}{"笔数":>5}{"均值":>7}{"中位":>7}{"P90":>7}{"最差":>7} | {"回测假设":>7}{"超出":>7}')
    for k in ('entry', 'sl', 'tp', 'time_stop', 'ma_exit'):
        s = d[d['kind'] == k]['slip_bp']
        if len(s) == 0:
            continue
        assume = ASSUME.get(k, 0)
        print(f'{k:<10}{len(s):>5}{s.mean():>7.1f}{s.median():>7.1f}{s.quantile(.9):>7.1f}'
              f'{s.max():>7.1f} | {assume:>7.1f}{s.mean()-assume:>+7.1f}')

    print('\n=== 大盘币 vs 小币 (入场+出场合并, bp) ===')
    for g in ('大盘币', '小币'):
        s = d[d['组'] == g]['slip_bp']
        if len(s):
            print(f'{g}: 笔数{len(s)} 均值{s.mean():+.1f}bp 中位{s.median():+.1f}bp P90={s.quantile(.9):+.1f}bp 最差{s.max():+.1f}bp')

    print('\n=== 滑点最大的10个币 (均值bp) ===')
    g = d.groupby('sym')['slip_bp'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print(g.head(10).round(1).to_string())

    print('\n判读: 若小币入场/止损滑点远超假设(2/5bp), 说明回测年化偏乐观, 需用实测值重跑 portfolio_sim。')


if __name__ == '__main__':
    main()
