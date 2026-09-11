"""实盘 fills 日志分析: 滑点分布、单笔盈亏、持仓暴露、滞后与滑点相关性。

fills.csv 列: time,kind,sym,ref_px,fill_px,slip_bp,lag_s,extra
用法: python analyze_live.py [csv路径]   默认 results/live_fills_pasted.csv
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), 'results', 'live_fills_pasted.csv')

FEE = 0.0005        # taker 单边费率, 与 portfolio_sim.FEE 一致
EXIT_SLIP_BT = 5.0  # 回测假设出场滑点 5bp (portfolio_sim.EXIT_SLIP=0.0005)
LEV = 3             # 实盘杠杆


def pf(x):
    """按价格量级自适应小数位: 大价(>=100)2位, 小价(如 PEPE)按有效位。"""
    a = abs(float(x))
    if a == 0:
        return '0'
    if a >= 100:
        return f'{x:.2f}'
    if a >= 1:
        return f'{x:.4f}'
    return f'{x:.10f}'.rstrip('0').rstrip('.')


def parse_contracts(extra):
    """extra 形如 'contracts=68.0' / 'fill_long contracts=0.39'。"""
    import re
    m = re.search(r'contracts=([\d.]+)', str(extra))
    return float(m.group(1)) if m else np.nan


def main():
    df = pd.read_csv(CSV, parse_dates=['time'])
    print(f'数据: {os.path.basename(CSV)}')
    print(f'窗口: {df.time.min():%Y-%m-%d %H:%M} ~ {df.time.max():%Y-%m-%d %H:%M} UTC  '
          f'共 {len(df)} 笔记录\n')

    ent = df[df.kind == 'entry'].sort_values('time').reset_index(drop=True)
    exi = df[df.kind == 'ma_exit'].sort_values('time').reset_index(drop=True)
    frl = df[df.kind == 'fill_entry']
    print('=== 概览 ===')
    print(f'入场 {len(ent)} 笔, 出场 {len(exi)} 笔, fill_entry {len(frl)} 笔, '
          f'其他 {len(df) - len(ent) - len(exi) - len(frl)} 笔')
    print(f'独立币种: 入场 {ent.sym.nunique()} 种, 出场 {exi.sym.nunique()} 种\n')

    # ---- 滑点 ----
    print('=== 滑点 (bp, 正值=对手方更差) ===')
    for name, g in [('入场', ent), ('出场', exi)]:
        s = g.slip_bp.astype(float)
        print(f'{name}: n={len(s):2d}  均值={s.mean():6.2f}  中位={s.median():6.2f}  '
              f'|均值|={s.abs().mean():6.2f}  max={s.max():6.2f}  min={s.min():6.2f}')
    print(f'注: 回测假设出场滑点 {EXIT_SLIP_BT:.0f}bp, 实际出场 |均值| 为 '
          f'{exi.slip_bp.abs().mean():.1f}bp\n')

    # ---- 滞后 ----
    print('=== 滞后 lag_s ===')
    lag = df.lag_s.astype(float)
    print(f'全部: 均值={lag.mean():.1f}s 中位={lag.median():.0f}s max={lag.max():.0f}s min={lag.min():.0f}s')
    print(f'入场中位={ent.lag_s.astype(float).median():.0f}s, '
          f'出场中位={exi.lag_s.astype(float).median():.0f}s')
    print(f'lag 与 |slip| 相关系数={lag.corr(df.slip_bp.abs()):.2f}\n')

    # ---- 单笔盈亏: 按时间顺序逐笔配对 (XRP 有两笔, 不能按币种聚合) ----
    ex, ea = exi.copy(), ent.copy()
    print('=== 闭环单笔 (方向=空) ===')
    print(f'{"币":<7}{"入场价":>12}{"出场价":>12}{"参考价收益":>10}{"真实fill收益":>11}'
          f'{"持有h":>7}{"滑点bp":>8}{"入场→出场":>22}')
    used, rows, rets, slips, hs = set(), [], [], [], []
    for _, e in ex.iterrows():
        cand = ea[(ea.sym == e.sym) & (~ea.index.isin(used)) & (ea.time <= e.time)]
        if cand.empty:
            continue
        a = cand.iloc[-1]
        used.add(a.name)
        x_ref, e_ref = float(a.ref_px), float(e.ref_px)
        x_f, e_f = float(a.fill_px), float(e.fill_px)
        ret_ref = 1 - e_ref / x_ref - 2 * FEE       # 空单: 1 - 出场价/入场价
        ret_f = 1 - e_f / x_f - 2 * FEE
        st = float(e.slip_bp) + float(a.slip_bp)
        h = (e.time - a.time).total_seconds() / 3600
        rows.append((e.sym, x_ref, e_ref, ret_ref, ret_f, h, st, a, e))
        rets.append(ret_ref); slips.append(st); hs.append(h)
    rows.sort(key=lambda r: r[3])          # 最亏在前
    for s, xr, er, rr, rf, h, st, a, e in rows:
        print(f'{s:<7}{pf(xr):>12}{pf(er):>12}{rr*100:>9.2f}%{rf*100:>10.2f}%'
              f'{h:>7.0f}{st:>8.1f}{a.time:%m-%d %H}→{e.time:%m-%d %H}')
    r = np.array(rets) * 100
    print(f'\n闭环 n={len(r)}  盈利={int((r>0).sum())} 亏损={int((r<=0).sum())} '
          f'胜率={100*(r>0).mean():.0f}%')
    print(f'均值={r.mean():.2f}%  中位={np.median(r):.2f}%  累计(简单加和)={r.sum():.2f}%')
    print(f'最好={r.max():.2f}%  最差={r.min():.2f}%  '
          f'平均持有={np.mean(hs):.0f}h  中位持有={np.median(hs):.0f}h')
    ss = np.array(slips)
    print(f'滑点(入+出)合计: 均值={ss.mean():.1f}bp  与收益相关={np.corrcoef(ss, r)[0, 1]:.2f}')
    # 真实 fill 收益 vs 参考价收益差异
    dr = np.array([row[4] - row[3] for row in rows]) * 100
    print(f'真实fill收益 - 参考价收益: 均值={dr.mean():+.2f}%  区间[{dr.min():+.2f}%, {dr.max():+.2f}%]\n')

    # ---- 未平仓 ----
    open_rows = ea[~ea.index.isin(used)].sort_values('time')
    print(f'=== 未平仓 (截至 {df.time.max():%Y-%m-%d %H:%M} UTC) ===')
    open_notional = 0
    for _, a in open_rows.iterrows():
        c = parse_contracts(a.extra)
        n = c * float(a.ref_px)
        open_notional += n
        print(f'{a.sym:<7} 入场 {pf(float(a.ref_px)):>12}  名义≈{n:7.1f} USDT  '
              f'{a.time:%Y-%m-%d %H:%M}')
    print(f'未平仓名义合计 ≈ {open_notional:.0f} USDT (保证金 ≈ {open_notional/LEV:.0f} USDT)\n')

    # ---- 名义敞口与资金占用 ----
    print('=== 单笔名义 (contracts × ref_px) ===')
    print(f'{"币":<7}{"contracts":>11}{"ref_px":>13}{"名义USDT":>11}{"入场时间":>16}')
    tots = {}
    for _, a in ent.iterrows():
        c = parse_contracts(a.extra)
        n = c * float(a.ref_px)
        tots[a.time.date()] = tots.get(a.time.date(), 0) + n
        print(f'{a.sym:<7}{c:>11.2f}{pf(float(a.ref_px)):>13}{n:>11.1f}{a.time:%Y-%m-%d %H:%M}')
    print(f'\n单日入场名义峰值 = {max(tots.values()):.1f} USDT '
          f'(日期 {max(tots, key=tots.get):%Y-%m-%d})')
    print(f'全部入场名义加和(未扣出场) ≈ {sum(tots.values()):.0f} USDT\n')

    # ---- 与回测假设的对比 ----
    print('=== 实盘 vs 回测假设 ===')
    print(f'回测退出假设: EXIT_SLIP=5bp 固定, 无入场滑点; 实际入场|slip|均值='
          f'{ent.slip_bp.abs().mean():.1f}bp, 出场|slip|均值={exi.slip_bp.abs().mean():.1f}bp')
    print(f'回测费率 FEE={FEE*10000:.0f}bp/边 (taker); 实盘按 taker 计')
    print(f'回测出场触发=同一 4h 收盘价; 实盘中位滞后 {exi.lag_s.astype(float).median():.0f}s '
          f'(=1~2 分钟, 属 OKX 轮询周期, 非策略延迟)')


if __name__ == '__main__':
    main()
