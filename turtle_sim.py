"""海龟做多腿 组合回测。

成本口径与 portfolio_sim / trend_sim 完全一致(同样的 FEE/SLIP/EXIT_SLIP/SLIP_THRU),
这样海龟腿的结果可以直接和做空腿、趋势腿比较。

与前两者的关键差异:
  仓位由 N 值决定(每单位风险固定占权益 RISK_PER_UNIT), 不是等名义 —— 所以"总敞口"
  不是预设的, 而是市场波动的结果。低波动期同样的风险预算会开出更大名义。
  故用 MAX_TOTAL 兜住总名义, 避免低波动期杠杆失控。

用法:
  UNIVERSE=universe_live.txt SPAN=20250101_20260801 python turtle_sim.py
"""
import os
import sys
import numpy as np
import pandas as pd
import turtle_long as tl
import combo_core as cc
from analyze_long import path

if sys.stdout.encoding and not sys.stdout.encoding.lower().startswith('utf'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

_UNI = os.environ.get('UNIVERSE', 'universe_live.txt')
SYMS = [x.strip() for x in open(os.path.join(os.path.dirname(__file__), _UNI),
                                encoding='utf-8').read().replace('\n', ',').split(',') if x.strip()]
INIT = 10000.0
MAX_SYMS = int(os.environ.get('MAX_SYMS', '20'))          # 同时持仓币数上限
MAX_TOTAL = float(os.environ.get('MAX_TOTAL', '1.5'))     # 总名义/权益 上限
FEE = cc.TAKER_FEE
SLIP = cc.SLIPPAGE
EXIT_SLIP = float(os.environ.get('EXIT_SLIP', '0.0005'))
SLIP_THRU = float(os.environ.get('SLIP_THRU', '0.3'))
CFG = dict(tl.DEFAULTS,
           ENTRY_N=int(os.environ.get('ENTRY_N', '20')),
           EXIT_N=int(os.environ.get('EXIT_N', '10')),
           LONG_ENTRY_N=int(os.environ.get('LONG_ENTRY_N', '55')),
           ATR_LEN=int(os.environ.get('ATR_LEN', '20')),
           STOP_N=float(os.environ.get('STOP_N', '2.0')),
           ADD_STEP_N=float(os.environ.get('ADD_STEP_N', '0.5')),
           MAX_UNITS=int(os.environ.get('MAX_UNITS', '4')),
           RISK_PER_UNIT=float(os.environ.get('RISK_PER_UNIT', '0.01')),
           USE_FILTER=os.environ.get('USE_FILTER', '1') == '1')


def build_panel():
    keys = ('close', 'high', 'low', 'atr', 'entry_sig', 'exit_sig')
    cols = {k: {} for k in keys}
    warm = max(CFG['LONG_ENTRY_N'], CFG['ENTRY_N'], CFG['ATR_LEN']) + 2
    for s in SYMS:
        p = path(s)
        if not p:
            continue
        ds = tl.gen_signals(tl.calc_indicators(cc.load_data(p), CFG), CFG)
        ds = ds.iloc[warm:]
        for k in keys:
            cols[k][s] = ds[k]
    C = pd.DataFrame(cols['close']).sort_index()
    idx = C.index
    out = [pd.DataFrame(cols[k]).reindex(idx) for k in keys[1:]]
    return (idx, C) + tuple(out) + (list(C.columns),)


def simulate(idx, C, H, L, A, S, X):
    """海龟做多。返回 (权益曲线, 逐笔盈亏(按币整体平仓算一笔), 均持仓币数)。"""
    Cv, Hv, Lv, Av, Sv, Xv = C.values, H.values, L.values, A.values, S.values, X.values
    T, nS = Cv.shape
    equity = INIT
    held = {}                    # j -> dict(units=[{entry,notional}], stop, n, last_entry)
    curve = np.empty(T)
    trades = []
    held_sum = 0
    last_win = np.zeros(nS, dtype=bool)   # 上一次系统1突破是否盈利(过滤用)
    has_hist = np.zeros(nS, dtype=bool)

    def close_all(j, px, why):
        """整仓平掉某币, 记一笔盈亏。"""
        nonlocal equity
        st = held[j]
        pnl = sum(u['notional'] * (px - u['entry']) / u['entry'] for u in st['units'])
        fee = sum(u['notional'] for u in st['units']) * FEE
        equity += pnl - fee
        trades.append(pnl - fee)
        last_win[j] = (pnl - fee) > 0
        has_hist[j] = True
        del held[j]

    for i in range(T):
        # ---- 出场: 止损优先, 再看通道反向突破 ----
        for j in list(held.keys()):
            l, c = Lv[i, j], Cv[i, j]
            if not np.isfinite(l):
                continue
            st = held[j]
            if l <= st['stop']:
                thru = st['stop'] + SLIP_THRU * (l - st['stop'])   # 向下击穿
                close_all(j, thru * (1 - EXIT_SLIP), 'stop')
            elif Xv[i, j] == 1 and np.isfinite(c):
                close_all(j, c * (1 - EXIT_SLIP), 'channel')

        # ---- 加仓金字塔: 价格每涨 ADD_STEP_N×N 加一单位, 并上移全部止损 ----
        used = sum(u['notional'] for st in held.values() for u in st['units'])
        for j in list(held.keys()):
            st = held[j]
            c = Cv[i, j]
            if not np.isfinite(c) or len(st['units']) >= CFG['MAX_UNITS']:
                continue
            trigger = st['last_entry'] + CFG['ADD_STEP_N'] * st['n']
            if c < trigger:
                continue
            unit_notional = _unit_notional(equity, st['n'], c)
            if unit_notional <= 0 or used + unit_notional > equity * MAX_TOTAL:
                continue
            entry = c * (1 + SLIP)
            equity -= unit_notional * FEE
            st['units'].append(dict(entry=entry, notional=unit_notional))
            st['last_entry'] = entry
            # 原版: 加仓后全部仓位的止损统一上移到"最新入场 − STOP_N×N"
            st['stop'] = entry - CFG['STOP_N'] * st['n']
            used += unit_notional

        # ---- 入场 ----
        if len(held) < MAX_SYMS:
            used = sum(u['notional'] for st in held.values() for u in st['units'])
            for j in range(nS):
                if len(held) >= MAX_SYMS or j in held:
                    continue
                sig, n, c = Sv[i, j], Av[i, j], Cv[i, j]
                if sig == 0 or not (np.isfinite(n) and n > 0 and np.isfinite(c)):
                    continue
                # 系统1 过滤: 上次突破盈利则跳过, 但系统2(更长通道)不受此限
                if CFG['USE_FILTER'] and sig == 1 and has_hist[j] and last_win[j]:
                    continue
                unit_notional = _unit_notional(equity, n, c)
                if unit_notional <= 0 or used + unit_notional > equity * MAX_TOTAL:
                    continue
                entry = c * (1 + SLIP)
                equity -= unit_notional * FEE
                held[j] = dict(units=[dict(entry=entry, notional=unit_notional)],
                               stop=entry - CFG['STOP_N'] * n, n=n, last_entry=entry)
                used += unit_notional

        mtm = equity
        for j, st in held.items():
            c = Cv[i, j]
            if np.isfinite(c):
                mtm += sum(u['notional'] * (c - u['entry']) / u['entry'] for u in st['units'])
        curve[i] = mtm
        held_sum += len(held)
    return curve, trades, held_sum / T


def _unit_notional(equity, n, price):
    """1单位名义: 令"止损被打到"时亏损 = RISK_PER_UNIT × 权益。
    风险比例 = STOP_N × N / 价格, 故 名义 = 风险预算 / 该比例。"""
    risk_pct = CFG['STOP_N'] * n / price
    if risk_pct <= 0:
        return 0.0
    return equity * CFG['RISK_PER_UNIT'] / risk_pct


def metrics(curve, trades, idx):
    eq = pd.Series(curve, index=idx)
    final = curve[-1]
    days = (idx[-1] - idx[0]).days or 1
    ann = (final / INIT) ** (365 / days) - 1 if final > 0 else -1
    dr = eq.resample('1D').last().dropna().pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = ((eq.cummax() - eq) / eq.cummax()).max()
    a = np.array(trades)
    w, l = a[a > 0], a[a <= 0]
    return dict(ret=(final - INIT) / INIT * 100, ann=ann * 100, sharpe=sharpe,
                mdd=mdd * 100, trades=len(a),
                win=(a > 0).mean() * 100 if len(a) else 0,
                pf=w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else 0,
                plr=abs(w.mean() / l.mean()) if len(l) and l.mean() != 0 else 0)


def main():
    idx, C, H, L, A, S, X, syms = build_panel()
    curve, trades, ah = simulate(idx, C, H, L, A, S, X)
    m = metrics(curve, trades, idx)
    print(f'{len(syms)}币 {idx[0].date()}~{idx[-1].date()}  通道{CFG["ENTRY_N"]}/{CFG["EXIT_N"]} '
          f'N={CFG["ATR_LEN"]} 止损{CFG["STOP_N"]}N 风险{CFG["RISK_PER_UNIT"]:.0%}/单位 '
          f'最多{CFG["MAX_UNITS"]}单位 过滤={CFG["USE_FILTER"]}')
    print(f'  年化 {m["ann"]:+.1f}%  夏普 {m["sharpe"]:.2f}  回撤 {m["mdd"]:.1f}%  '
          f'交易 {m["trades"]}  胜率 {m["win"]:.1f}%  盈亏比 {m["plr"]:.2f}  因子 {m["pf"]:.2f}  '
          f'均持仓 {ah:.1f}币')


if __name__ == '__main__':
    main()
