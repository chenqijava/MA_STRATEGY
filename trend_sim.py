"""MA90 趋势跟随做多腿的组合回测。

与 portfolio_sim 分开写, 因为出场规则镜像不同:
  做空腿: 收盘 > 慢线 平仓 (反弹突破阻力)
  做多腿: 收盘 < 慢线 平仓 (跌破支撑), 且要求"入场后第1根起"才检查
成本口径与 portfolio_sim 完全一致 (同样的 FEE/SLIP/EXIT_SLIP/SLIP_THRU), 这样两腿的
结果可以直接比较, 不会因为成本假设不同而失真。

用法:
  UNIVERSE=universe_live.txt N=20 TOTAL=1.5 MA_SLOW=90 python trend_sim.py
"""
import os
import sys
import numpy as np
import pandas as pd
import ma_trend_long as mt
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
TOTAL_NOTIONAL = float(os.environ.get('TOTAL', '1.5'))
FEE = cc.TAKER_FEE                                        # 5bp, 与做空腿同
SLIP = cc.SLIPPAGE                                        # 入场 2bp
EXIT_SLIP = float(os.environ.get('EXIT_SLIP', '0.0005'))  # 出场 5bp
SLIP_THRU = float(os.environ.get('SLIP_THRU', '0.3'))     # 止损击穿
MA_EXIT = os.environ.get('MA_EXIT', '1') == '1'           # 跌破慢线出场
COOLDOWN_ONCE = os.environ.get('COOLDOWN_ONCE', '0') == '1'
CFG = dict(mt.DEFAULTS,
           MA_SLOW=int(os.environ.get('MA_SLOW', '90')),
           ATR_TP=float(os.environ.get('ATR_TP', '4.0')),
           ATR_SL=float(os.environ.get('ATR_SL', '2.0')),
           MAX_BARS=int(os.environ.get('MAX_BARS', '0')),
           MA_BUFFER=float(os.environ.get('MA_BUFFER', '0')),
           MIN_BODY_PCT=float(os.environ.get('MIN_BODY_PCT', '0')))


def build_panel():
    cols = {k: {} for k in ('close', 'high', 'low', 'atr', 'entry_sig', 'ma_slow')}
    for s in SYMS:
        p = path(s)
        if not p:
            continue
        ds = mt.gen_signals(mt.calc_indicators(cc.load_data(p), CFG), CFG)
        ds = ds.iloc[CFG['MA_SLOW'] + 2:]
        for k in cols:
            cols[k][s] = ds[k]
    C = pd.DataFrame(cols['close']).sort_index()
    idx = C.index
    out = [pd.DataFrame(cols[k]).reindex(idx) for k in ('high', 'low', 'atr', 'entry_sig', 'ma_slow')]
    return (idx, C) + tuple(out) + (list(C.columns),)


def simulate(n, idx, C, H, L, A, S, MA):
    """做多腿组合模拟。返回 (权益曲线, 逐笔盈亏, 均持仓数)。"""
    Cv, Hv, Lv, Av, Sv, MAv = C.values, H.values, L.values, A.values, S.values, MA.values
    T, nS = Cv.shape
    equity = INIT
    held = {}
    curve = np.empty(T)
    trades = []
    held_sum = 0
    done = np.zeros(nS, dtype=bool)

    for i in range(T):
        # 结构恢复 -> 解除"本轮已做过"标记 (COOLDOWN_ONCE 用)
        if COOLDOWN_ONCE:
            ok = np.isfinite(MAv[i]) & np.isfinite(Cv[i])
            done[ok & (Cv[i] < MAv[i])] = False

        # ---- 出场 (止损优先, 与做空腿同一套损耗建模, 方向镜像) ----
        for j in list(held.keys()):
            h, l, c, ma = Hv[i, j], Lv[i, j], Cv[i, j], MAv[i, j]
            if not np.isfinite(h):
                continue
            st = held[j]
            st['bars'] += 1
            if l <= st['stop']:
                # 多头止损: 价格向下击穿, 成交介于 stop 与当根最低之间
                thru = st['stop'] + SLIP_THRU * (l - st['stop'])
                exit_px = thru * (1 - EXIT_SLIP)          # 卖出成交更低 = 更差
            elif h >= st['target']:
                exit_px = st['target'] * (1 - EXIT_SLIP)
            elif CFG['MAX_BARS'] and st['bars'] >= CFG['MAX_BARS'] and np.isfinite(c):
                exit_px = c * (1 - EXIT_SLIP)
            elif (MA_EXIT and st['bars'] >= 1 and np.isfinite(ma) and np.isfinite(c)
                  and c < ma):
                # 入场后第1根起: 收盘跌回慢线下方 = 上升结构已破
                exit_px = c * (1 - EXIT_SLIP)
            else:
                exit_px = None
            if exit_px is not None:
                pnl = st['notional'] * (exit_px - st['entry']) / st['entry']
                equity += pnl - st['notional'] * FEE
                trades.append(pnl - st['notional'] * FEE)
                done[j] = True
                del held[j]

        # ---- 入场 ----
        if len(held) < n:
            for j in range(nS):
                if len(held) >= n:
                    break
                if j in held or (COOLDOWN_ONCE and done[j]):
                    continue
                sig, atr, c = Sv[i, j], Av[i, j], Cv[i, j]
                if sig != 1 or not (np.isfinite(atr) and atr > 0 and np.isfinite(c)):
                    continue
                entry = c * (1 + SLIP)                   # 买入: 成交价更高 = 不利
                notional = equity * TOTAL_NOTIONAL / n
                equity -= notional * FEE
                held[j] = dict(entry=entry, notional=notional, bars=0,
                               stop=entry - CFG['ATR_SL'] * atr,
                               target=entry + CFG['ATR_TP'] * atr)

        mtm = equity
        for j, st in held.items():
            c = Cv[i, j]
            if np.isfinite(c):
                mtm += st['notional'] * (c - st['entry']) / st['entry']
        curve[i] = mtm
        held_sum += len(held)
    return curve, trades, held_sum / T


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
    idx, C, H, L, A, S, MA, syms = build_panel()
    curve, trades, ah = simulate(N, idx, C, H, L, A, S, MA)
    m = metrics(curve, trades, idx)
    print(f'{len(syms)}币 {idx[0].date()}~{idx[-1].date()}  N={N} 敞口{TOTAL_NOTIONAL}x '
          f'MA{CFG["MA_SLOW"]} TP{CFG["ATR_TP"]}/SL{CFG["ATR_SL"]} MA出场={MA_EXIT}')
    print(f'  年化 {m["ann"]:+.1f}%  夏普 {m["sharpe"]:.2f}  回撤 {m["mdd"]:.1f}%  '
          f'交易 {m["trades"]}  胜率 {m["win"]:.1f}%  盈亏比 {m["plr"]:.2f}  因子 {m["pf"]:.2f}  '
          f'均持仓 {ah:.1f}')


if __name__ == '__main__':
    main()
