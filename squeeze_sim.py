"""均线收敛发散 + MACD 做多腿 组合回测。

出场是本策略的重点(按要求"完全独立于入场信号"), 三层并行, 取先触发者:
  1. 移动止损: 初始 入场 − TRAIL_ATR×ATR; 记录持仓期间最高价, 止损 = 最高 − TRAIL_ATR×ATR,
     只上移不下移(永不回撤)。
  2. 单线跟踪: 收盘 < MA_TRAIL(默认MA20) 即走。比等四线转向快得多。
  3. 分批: 浮盈达 SCALE_ATR×ATR 平一半, 余仓改用 TRAIL_ATR2(更宽)止损博大趋势。

成本口径与 portfolio_sim / turtle_sim 一致, 结果可直接对比。
用法: UNIVERSE=universe_live.txt SPAN=20250101_20260801 python squeeze_sim.py
"""
import os
import sys
import numpy as np
import pandas as pd
import squeeze_long as sq
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
# 出场
TRAIL_ATR = float(os.environ.get('TRAIL_ATR', '3.0'))     # 初始/常规移动止损宽度
TRAIL_ATR2 = float(os.environ.get('TRAIL_ATR2', '5.0'))   # 分批后余仓的更宽止损
USE_MA_TRAIL = os.environ.get('USE_MA_TRAIL', '1') == '1'
USE_SCALE = os.environ.get('USE_SCALE', '1') == '1'
SCALE_ATR = float(os.environ.get('SCALE_ATR', '3.0'))     # 浮盈达此倍数平一半
CFG = dict(sq.DEFAULTS,
           USE_SQUEEZE=os.environ.get('USE_SQUEEZE', '1') == '1',
           REQUIRE_FRESH=os.environ.get('REQUIRE_FRESH', '1') == '1',
           USE_ADX=os.environ.get('USE_ADX', '1') == '1',
           USE_BBW=os.environ.get('USE_BBW', '0') == '1',
           USE_VOL=os.environ.get('USE_VOL', '1') == '1',
           USE_MACD=os.environ.get('USE_MACD', '1') == '1',
           SQZ_LOOK=int(os.environ.get('SQZ_LOOK', '20')),
           SQZ_TH=float(os.environ.get('SQZ_TH', '0.03')),
           EXT_MAX=float(os.environ.get('EXT_MAX', '0.20')),
           ADX_MIN=float(os.environ.get('ADX_MIN', '20')),
           VOL_MULT=float(os.environ.get('VOL_MULT', '1.2')),
           MACD_WINDOW=int(os.environ.get('MACD_WINDOW', '3')))


def build_panel():
    keys = ('close', 'high', 'low', 'atr', 'entry_sig', 'ma_trail')
    cols = {k: {} for k in keys}
    warm = max(CFG['MA4'], 100) + 5
    for s in SYMS:
        p = path(s)
        if not p:
            continue
        ds = sq.gen_signals(sq.calc_indicators(cc.load_data(p), CFG), CFG)
        ds = ds.iloc[warm:]
        for k in keys:
            cols[k][s] = ds[k]
    C = pd.DataFrame(cols['close']).sort_index()
    idx = C.index
    out = [pd.DataFrame(cols[k]).reindex(idx) for k in keys[1:]]
    return (idx, C) + tuple(out) + (list(C.columns),)


def simulate(idx, C, H, L, A, S, MT):
    """返回 (权益曲线, 逐笔盈亏, 均持仓数)。分批离场记为两笔。"""
    Cv, Hv, Lv, Av, Sv, MTv = C.values, H.values, L.values, A.values, S.values, MT.values
    T, nS = Cv.shape
    equity = INIT
    held = {}
    curve = np.empty(T)
    trades = []
    held_sum = 0

    def sell(j, px, part):
        """卖出 part 比例的仓位(1.0=全平)。"""
        nonlocal equity
        st = held[j]
        notl = st['notional'] * part
        pnl = notl * (px - st['entry']) / st['entry']
        equity += pnl - notl * FEE
        trades.append(pnl - notl * FEE)
        st['notional'] -= notl
        if st['notional'] <= 1e-9:
            del held[j]

    for i in range(T):
        # ---- 出场: 三层并行, 止损优先(同根内先触及的是更差的情形) ----
        for j in list(held.keys()):
            h, l, c, atr, mt = Hv[i, j], Lv[i, j], Cv[i, j], Av[i, j], MTv[i, j]
            if not np.isfinite(h):
                continue
            st = held[j]
            st['bars'] += 1
            # 更新持仓期最高价 -> 移动止损上移(永不回撤)
            if h > st['peak']:
                st['peak'] = h
                w = TRAIL_ATR2 if st['scaled'] else TRAIL_ATR
                st['stop'] = max(st['stop'], st['peak'] - w * st['atr0'])
            # 1) 移动止损
            if l <= st['stop']:
                thru = st['stop'] + SLIP_THRU * (l - st['stop'])
                sell(j, thru * (1 - EXIT_SLIP), 1.0)
                continue
            # 2) 分批: 浮盈达 SCALE_ATR×ATR 平一半, 余仓换更宽止损
            if USE_SCALE and not st['scaled'] and np.isfinite(c) \
                    and c >= st['entry'] + SCALE_ATR * st['atr0']:
                sell(j, c * (1 - EXIT_SLIP), 0.5)
                if j in held:
                    held[j]['scaled'] = True
                    held[j]['stop'] = max(held[j]['stop'],
                                          held[j]['peak'] - TRAIL_ATR2 * held[j]['atr0'])
                continue
            # 3) 单线跟踪: 收盘破 MA20 即走 (比等四线转向快)
            if USE_MA_TRAIL and st['bars'] >= 1 and np.isfinite(mt) and np.isfinite(c) and c < mt:
                sell(j, c * (1 - EXIT_SLIP), 1.0)

        # ---- 入场 ----
        if len(held) < N:
            used = sum(s['notional'] for s in held.values())
            for j in range(nS):
                if len(held) >= N or j in held:
                    continue
                sig, atr, c = Sv[i, j], Av[i, j], Cv[i, j]
                if sig != 1 or not (np.isfinite(atr) and atr > 0 and np.isfinite(c)):
                    continue
                notl = equity * TOTAL / N
                if used + notl > equity * TOTAL:
                    continue
                entry = c * (1 + SLIP)
                equity -= notl * FEE
                held[j] = dict(entry=entry, notional=notl, bars=0, atr0=atr,
                               peak=entry, stop=entry - TRAIL_ATR * atr, scaled=False)
                used += notl

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
    idx, C, H, L, A, S, MT, syms = build_panel()
    cur, tr, ah = simulate(idx, C, H, L, A, S, MT)
    m = metrics(cur, tr, idx)
    on = [k for k, v in (('收敛', CFG['USE_SQUEEZE']), ('刚成立', CFG['REQUIRE_FRESH']),
                         ('ADX', CFG['USE_ADX']), ('BBW', CFG['USE_BBW']),
                         ('放量', CFG['USE_VOL']), ('MACD', CFG['USE_MACD'])) if v]
    print(f'{len(syms)}币 {idx[0].date()}~{idx[-1].date()}  条件[{"+".join(on)}]  '
          f'出场[移动{TRAIL_ATR}ATR' + (f'/分批{SCALE_ATR}ATR→{TRAIL_ATR2}ATR' if USE_SCALE else '')
          + ('/MA20跟踪' if USE_MA_TRAIL else '') + ']')
    print(f'  年化 {m["ann"]:+.1f}%  夏普 {m["sharpe"]:.2f}  回撤 {m["mdd"]:.1f}%  '
          f'交易 {m["trades"]}  胜率 {m["win"]:.1f}%  盈亏比 {m["plr"]:.2f}  因子 {m["pf"]:.2f}  '
          f'均持仓 {ah:.1f}')


if __name__ == '__main__':
    main()
