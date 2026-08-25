"""蒙特卡洛: 用单笔收益率重抽, 估计"这套参数在别的运气下会怎样"。

为什么做: 回测只有 1 条历史路径。三段最差年化 +20.2% 是那条路径的结果, 不是
分布。蒙特卡洛把"顺序运气"和"抽样运气"剥出来, 回答三个实盘要问的问题:
  ① 破产/大回撤概率有多高?
  ② 亏损年的概率有多高? 最差情况有多差?
  ③ 观测到的 +45.6% 在分布里处于什么位置(是不是靠一段好运)?

三种重抽方式, 各答不同问题:
  · 顺序重排(shuffle): 保留全部单笔收益, 只打乱顺序。答"如果同样这些交易换个
    先后顺序, 回撤会差多少" —— 隔离顺序风险(连亏聚堆)。
  · 有放回重抽(bootstrap): 从单笔池里随机抽 n 笔。答"如果重新抽一批同分布的
    交易" —— 隔离抽样风险。这个最保守。
  · 分块重抽(block bootstrap): 按连续块抽, 保留短期自相关(同一段行情里的交易
    往往同向)。独立重抽会低估回撤, 分块能部分还原聚集性。

口径要点:
  · 用**相对收益率**(pnl/开仓时权益)而非绝对金额。名义随权益复利变化, 绝对
    金额把顺序信息混进了幅度, 重排会失真。
  · 复利累计: eq *= (1+r)。策略是按权益比例下仓的, 复利才是它的真实行为。
  · 同时持仓多笔(均持仓4.5币)意味着单笔并非独立事件, 所以逐笔串行累计会
    **低估**真实回撤(真实回撤是多仓同时浮亏叠加)。分块重抽用于缓解, 但
    结论仍偏乐观 —— 回撤分位数应作下限看。

用法: SPAN=20230101_20260801 UNIVERSE=universe_live.txt TOTAL=1.5 N=20 \
      MA_SLOW=90 MA_EXIT=1 COOLDOWN_ONCE=1 ATR_TP=10 ATR_SL=10 python monte_carlo.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import portfolio_sim as ps

N = int(os.environ.get('N', '20'))
RUNS = int(os.environ.get('MC_RUNS', '20000'))
BLOCK = int(os.environ.get('MC_BLOCK', '20'))     # 分块重抽的块长(笔)
SEED = int(os.environ.get('MC_SEED', '42'))
RUIN = float(os.environ.get('MC_RUIN', '50'))     # "破产"阈值: 回撤超此% 视为不可接受


def _font():
    plt.rcParams['axes.unicode_minus'] = False
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for fn in ('Microsoft YaHei', 'SimHei', 'DengXian', 'SimSun'):
        if fn in have:
            plt.rcParams['font.sans-serif'] = [fn]
            return


def path_stats(r, years):
    """一条路径: 复利累计, 返回 (总收益%, 年化%, 最大回撤%)。"""
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    mdd = np.max((peak - eq) / peak) * 100
    total = (eq[-1] - 1) * 100
    ann = ((eq[-1]) ** (1 / years) - 1) * 100 if eq[-1] > 0 else -100
    return total, ann, mdd


def simulate_paths(r0, mode, runs, years, rng):
    n = len(r0)
    out = np.empty((runs, 3))
    for k in range(runs):
        if mode == 'shuffle':
            r = rng.permutation(r0)
        elif mode == 'bootstrap':
            r = r0[rng.integers(0, n, n)]
        else:                                     # block bootstrap
            nb = int(np.ceil(n / BLOCK))
            starts = rng.integers(0, max(1, n - BLOCK), nb)
            r = np.concatenate([r0[s:s + BLOCK] for s in starts])[:n]
        out[k] = path_stats(r, years)
    return out


def report(name, res, obs, years):
    tot, ann, mdd = res[:, 0], res[:, 1], res[:, 2]
    print(f'\n--- {name} ({len(res)}次) ---')
    print(f'{"":14s}{"P5":>9s}{"P25":>9s}{"中位":>9s}{"P75":>9s}{"P95":>9s}{"均值":>9s}')
    for lab, v in (('年化%', ann), ('总收益%', tot), ('最大回撤%', mdd)):
        q = np.percentile(v, [5, 25, 50, 75, 95])
        print(f'{lab:14s}' + ''.join(f'{x:+9.1f}' for x in q) + f'{v.mean():+9.1f}')
    print(f'  亏损概率(年化<0)      {(ann < 0).mean()*100:5.1f}%')
    print(f'  年化<10% 概率         {(ann < 10).mean()*100:5.1f}%')
    print(f'  回撤>30% 概率         {(mdd > 30).mean()*100:5.1f}%')
    print(f'  回撤>{RUIN:.0f}% 概率(视为破产) {(mdd > RUIN).mean()*100:5.1f}%')
    print(f'  实测年化 {obs:+.1f}% 的分位  {(ann < obs).mean()*100:5.1f}%'
          f'  (>50% 说明历史那条路径比典型情况好)')


def main():
    idx, C, H, L, A, S, syms = ps.build_panel()
    curve, trades, avg_held = ps.simulate(N, idx, C, H, L, A, S)
    m = ps.metrics(curve, trades, idx)
    tt = ps._TRADE_TRACE
    if not tt:
        raise SystemExit('_TRADE_TRACE 为空; 请确认 portfolio_sim 是本次 import 的版本')
    r0 = np.array([x[0] for x in tt], float)
    years = (idx[-1] - idx[0]).days / 365.0

    print('=' * 96)
    print(f'蒙特卡洛  {idx[0].date()} ~ {idx[-1].date()} ({years:.2f}年)  '
          f'币池={len(syms)} N={N} {ps.TOTAL_NOTIONAL:.1f}x MA{ps.CFG["MA_SLOW"]}')
    print(f'实测(历史那一条路径): 年化 {m["ann"]:+.1f}%  夏普 {m["sharpe"]:.2f}  '
          f'回撤 {m["mdd"]:.1f}%  {len(r0)}笔')
    print(f'单笔相对收益率: 均值 {r0.mean()*100:+.4f}%  中位 {np.median(r0)*100:+.4f}%  '
          f'标准差 {r0.std()*100:.3f}%  最好 {r0.max()*100:+.2f}%  最差 {r0.min()*100:+.2f}%')
    print(f'重抽 {RUNS} 次 / 每条 {len(r0)} 笔 / 分块块长 {BLOCK} / seed {SEED}')
    print('=' * 96)

    rng = np.random.default_rng(SEED)
    modes = [('顺序重排(隔离顺序运气)', 'shuffle'),
             ('有放回重抽(隔离抽样运气)', 'bootstrap'),
             (f'分块重抽(块长{BLOCK}, 保留聚集性)', 'block')]
    results = {}
    for lab, mode in modes:
        res = simulate_paths(r0, mode, RUNS, years, rng)
        results[lab] = res
        report(lab, res, m['ann'], years)

    # ---- 逐笔串行的两个已量化偏差, 以及用日收益重抽做的修正 ----
    ser = np.prod(1 + r0)
    print('\n' + '=' * 96)
    print('逐笔串行口径的偏差 (必须先说清, 否则上面的绝对水平会被误读)')
    print('=' * 96)
    print(f'  实测权益曲线 期末 {curve[-1]:.0f} (总 {m["ret"]:+.1f}%, 年化 {m["ann"]:+.1f}%)')
    print(f'  串行复利重构 期末 {ps.INIT*ser:.0f} (总 {(ser-1)*100:+.1f}%, '
          f'年化 {(ser**(1/years)-1)*100:+.1f}%)')
    eqs = np.cumprod(1 + r0); pk = np.maximum.accumulate(eqs)
    ser_mdd = np.max((pk - eqs) / pk) * 100
    print(f'  串行回撤 {ser_mdd:.1f}% vs 实测 {m["mdd"]:.1f}%  -> 低估 {m["mdd"]/ser_mdd:.2f} 倍')
    print(f'  根因: 同时持仓约 {avg_held:.1f} 币, 各笔按"开仓时权益"折相对收益率, '
          f'串行相乘等于假设它们依次独立发生 -> 高估复利、低估并发浮亏叠加。')
    print('  => 上面三档的**绝对水平偏高**; 它们真正可用的是分布形状(跨度/回撤/连亏)。')

    # 日收益分块重抽: 天然含并发效应, 绝对水平可直接对齐实测
    daily = pd.Series(curve, index=idx).resample('1D').last().dropna().pct_change().dropna().values
    dblk = int(os.environ.get('MC_DBLOCK', '20'))     # 块长(天)
    rng4 = np.random.default_rng(SEED + 3)
    nd = len(daily)
    res_d = np.empty((RUNS, 3))
    for k in range(RUNS):
        nb = int(np.ceil(nd / dblk))
        st = rng4.integers(0, max(1, nd - dblk), nb)
        r = np.concatenate([daily[s:s + dblk] for s in st])[:nd]
        res_d[k] = path_stats(r, nd / 365.0)
    results[f'日收益分块重抽(块长{dblk}天) ★口径最正'] = res_d
    report(f'日收益分块重抽(块长{dblk}天) ★口径最正 —— 含并发效应, 水平可直接对齐实测',
           res_d, m['ann'], years)

    # 连续亏损笔数分布 (实盘心理承受)
    print('\n--- 连续亏损笔数 (有放回重抽) ---')
    rng2 = np.random.default_rng(SEED + 1)
    streaks = np.empty(RUNS, dtype=int)
    for k in range(RUNS):
        r = r0[rng2.integers(0, len(r0), len(r0))]
        best = cur = 0
        for v in r:
            cur = cur + 1 if v <= 0 else 0
            best = max(best, cur)
        streaks[k] = best
    q = np.percentile(streaks, [50, 75, 90, 95, 99])
    print(f'  最长连亏: 中位 {q[0]:.0f}  P75 {q[1]:.0f}  P90 {q[2]:.0f}  '
          f'P95 {q[3]:.0f}  P99 {q[4]:.0f}  最坏 {streaks.max()}')
    print(f'  出现≥10连亏的概率 {(streaks >= 10).mean()*100:.0f}%   '
          f'≥15连亏 {(streaks >= 15).mean()*100:.0f}%   ≥20连亏 {(streaks >= 20).mean()*100:.0f}%')

    # ---------------- 画图 ----------------
    _font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    boot = results[modes[1][0]]
    blk = res_d          # 画图用口径最正的那档(日收益分块), 而非逐笔串行

    ax = axes[0, 0]
    for lab, res, col in ((modes[1][0], boot, '#ff7f0e'),
                          (modes[2][0], results[modes[2][0]], '#2ca02c'),
                          ('日收益分块 ★口径最正', res_d, '#1f77b4')):
        ax.hist(res[:, 1], bins=60, histtype='step', lw=1.4, color=col, label=lab.split('(')[0])
    ax.axvline(m['ann'], color='red', ls='--', lw=1.5, label=f'实测 {m["ann"]:.1f}%')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('年化 %'); ax.set_ylabel('次数'); ax.legend(fontsize=8)
    ax.set_title('年化收益分布', fontsize=11); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(blk[:, 2], bins=60, color='#d62728', alpha=0.7, edgecolor='white')
    ax.axvline(m['mdd'], color='black', ls='--', lw=1.5, label=f'实测 {m["mdd"]:.1f}%')
    for p, c in ((95, '#ff7f0e'), (99, '#8c564b')):
        v = np.percentile(blk[:, 2], p)
        ax.axvline(v, color=c, ls=':', lw=1.3, label=f'P{p} {v:.1f}%')
    ax.set_xlabel('最大回撤 %'); ax.set_ylabel('次数'); ax.legend(fontsize=8)
    ax.set_title('最大回撤分布 (日收益分块重抽, 含并发效应)', fontsize=11); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(blk[:, 2], blk[:, 1], s=2, alpha=0.15, color='#1f77b4')
    ax.scatter([m['mdd']], [m['ann']], s=90, color='red', marker='*',
               zorder=5, label='实测')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xlabel('最大回撤 %'); ax.set_ylabel('年化 %'); ax.legend(fontsize=8)
    ax.set_title('年化 vs 回撤 (每点=一条模拟路径)', fontsize=11); ax.grid(alpha=0.3)

    # 权益路径样本
    ax = axes[1, 1]
    rng3 = np.random.default_rng(SEED + 2)
    for _ in range(150):
        nb = int(np.ceil(nd / dblk))
        st = rng3.integers(0, max(1, nd - dblk), nb)
        r = np.concatenate([daily[s:s + dblk] for s in st])[:nd]
        ax.plot(np.cumprod(1 + r) * ps.INIT, color='#1f77b4', lw=0.5, alpha=0.15)
    ax.plot(np.cumprod(1 + daily) * ps.INIT, color='red', lw=1.8, label='实测路径')
    ax.axhline(ps.INIT, color='gray', ls='--', lw=0.8)
    ax.set_yscale('log'); ax.set_xlabel('交易日'); ax.set_ylabel('权益 (U, 对数)')
    ax.legend(fontsize=8); ax.set_title('150条模拟权益路径', fontsize=11); ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'results', 'monte_carlo.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f'\n已保存 {out}')
    print('\n注意: 逐笔串行累计**低估**真实回撤 —— 实际同时持仓约'
          f'{avg_held:.1f}币, 多仓同时浮亏会叠加。回撤分位数请当下限看。')


if __name__ == '__main__':
    main()
