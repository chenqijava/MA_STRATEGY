"""MNO 双路径做多 组合回测。

风控按描述: 止盈 TP_PCT / 止损 SL_PCT / 最长持仓 MAX_BARS 根。
成本口径与 portfolio_sim / turtle_sim / squeeze_sim 一致, 结果可直接对比。

用法:
  UNIVERSE=universe_live.txt SPAN=20250101_20260801 python mno_sim.py
  KAKU_ONLY=1 python mno_sim.py          # 仅KAKU模式
"""
import os
import sys
import numpy as np
import pandas as pd
import mno_long as ml
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
N = int(os.environ.get('N', '20'))
TOTAL = float(os.environ.get('TOTAL', '1.5'))
FEE = cc.TAKER_FEE
SLIP = cc.SLIPPAGE
EXIT_SLIP = float(os.environ.get('EXIT_SLIP', '0.0005'))
SLIP_THRU = float(os.environ.get('SLIP_THRU', '0.3'))
TP_PCT = float(os.environ.get('TP_PCT', '0.02'))          # 止盈 2%
SL_PCT = float(os.environ.get('SL_PCT', '0.01'))          # 止损 1%
MAX_BARS = int(os.environ.get('MAX_BARS', '30'))          # 最长持仓 30 根
CFG = dict(ml.DEFAULTS,
           USE_MOU=os.environ.get('USE_MOU', '1') == '1',
           USE_KAKU=os.environ.get('USE_KAKU', '1') == '1',
           KAKU_ONLY=os.environ.get('KAKU_ONLY', '0') == '1',
           ZERO_TH=float(os.environ.get('ZERO_TH', '0.001')),
           VOL_MIN=float(os.environ.get('VOL_MIN', '1.3')),
           VOL_MAX=float(os.environ.get('VOL_MAX', '3.0')),
           PB_MIN=float(os.environ.get('PB_MIN', '0.05')),
           PB_MAX=float(os.environ.get('PB_MAX', '0.15')),
           BRK_TH=float(os.environ.get('BRK_TH', '0.003')),
           BODY_TH=float(os.environ.get('BODY_TH', '0.20')),
           KAKU_TREND_BARS=int(os.environ.get('KAKU_TREND_BARS', '10')),
           KAKU_ADX_MIN=float(os.environ.get('KAKU_ADX_MIN', '25')),
           KAKU_VOL_MIN=float(os.environ.get('KAKU_VOL_MIN', '1.5')),
           PIN_RATIO=float(os.environ.get('PIN_RATIO', '2.0')))


def build_panel():
    keys = ('close', 'high', 'low', 'atr', 'entry_sig')
    cols = {k: {} for k in keys}
    for s in SYMS:
        p = path(s)
        if not p:
            continue
        ds = ml.gen_signals(ml.calc_indicators(cc.load_data(p), CFG), CFG)
        ds = ds.iloc[60:]
        for k in keys:
            cols[k][s] = ds[k]
    C = pd.DataFrame(cols['close']).sort_index()
    idx = C.index
    out = [pd.DataFrame(cols[k]).reindex(idx) for k in keys[1:]]
    return (idx, C) + tuple(out) + (list(C.columns),)


def simulate(idx, C, H, L, A, S):
    Cv, Hv, Lv, Sv = C.values, H.values, L.values, S.values
    T, nS = Cv.shape
    equity = INIT
    held = {}
    curve = np.empty(T)
    trades = []
    kinds = []
    held_sum = 0
    for i in range(T):
        for j in list(held.keys()):
            h, l, c = Hv[i, j], Lv[i, j], Cv[i, j]
            if not np.isfinite(h):
                continue
            st = held[j]
            st['bars'] += 1
            if l <= st['stop']:                              # 止损优先
                thru = st['stop'] + SLIP_THRU * (l - st['stop'])
                px = thru * (1 - EXIT_SLIP)
            elif h >= st['target']:
                px = st['target'] * (1 - EXIT_SLIP)
            elif MAX_BARS and st['bars'] >= MAX_BARS and np.isfinite(c):
                px = c * (1 - EXIT_SLIP)
            else:
                continue
            pnl = st['notional'] * (px - st['entry']) / st['entry']
            equity += pnl - st['notional'] * FEE
            trades.append(pnl - st['notional'] * FEE)
            kinds.append(st['kind'])
            del held[j]

        if len(held) < N:
            used = sum(s['notional'] for s in held.values())
            for j in range(nS):
                if len(held) >= N or j in held:
                    continue
                sig, c = Sv[i, j], Cv[i, j]
                if sig == 0 or not np.isfinite(c):
                    continue
                notl = equity * TOTAL / N
                if used + notl > equity * TOTAL:
                    continue
                entry = c * (1 + SLIP)
                equity -= notl * FEE
                held[j] = dict(entry=entry, notional=notl, bars=0, kind=int(sig),
                               stop=entry * (1 - SL_PCT), target=entry * (1 + TP_PCT))
                used += notl

        mtm = equity
        for j, st in held.items():
            c = Cv[i, j]
            if np.isfinite(c):
                mtm += st['notional'] * (c - st['entry']) / st['entry']
        curve[i] = mtm
        held_sum += len(held)
    return curve, trades, held_sum / T, kinds


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
    idx, C, H, L, A, S, syms = build_panel()
    cur, tr, ah, kinds = simulate(idx, C, H, L, A, S)
    m = metrics(cur, tr, idx)
    k = np.array(kinds)
    a = np.array(tr)
    print(f'{len(syms)}币 {idx[0].date()}~{idx[-1].date()}  '
          f'TP{TP_PCT:.1%}/SL{SL_PCT:.1%} 时停{MAX_BARS}根  '
          f'{"仅KAKU" if CFG["KAKU_ONLY"] else "MOU+KAKU"}')
    print(f'  年化 {m["ann"]:+.1f}%  夏普 {m["sharpe"]:.2f}  回撤 {m["mdd"]:.1f}%  '
          f'交易 {m["trades"]}  胜率 {m["win"]:.1f}%  盈亏比 {m["plr"]:.2f}  因子 {m["pf"]:.2f}')
    for code, nm in ((1, 'MOU'), (2, 'KAKU')):
        sel = k == code
        if sel.sum():
            print(f'    {nm}: {sel.sum()}笔 胜率{(a[sel] > 0).mean()*100:.1f}% '
                  f'均{a[sel].mean():+.2f}U')


if __name__ == '__main__':
    main()
