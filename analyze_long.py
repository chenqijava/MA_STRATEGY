"""做多腿 vs 做空腿 实证对比 (全16币, 4h, 已验证配置)。

口径与做空腿验证一致: ATR归一的单笔盈亏(pnl / (entry_atr·contract·面值)) 做跨币种汇总, 求 t 值。
配置: MA5/MA24, CROSS_DELAY=6, TOUCH=atr(0.5·ATR), 出场=ATR止盈4/止损2。
样本: 2025-01 ~ 2026-08 (数据缓存)。样本内/外按时间中点切分。
"""
import os, glob
import numpy as np
import pandas as pd
import ma_pullback as mp

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
SYMS = ['BTC','ETH','SOL','BNB','XRP','DOGE','ADA','AVAX','LINK','LTC','DOT','TRX','ATOM','UNI','FIL','APT']

BASE = dict(mp.DEFAULTS, MA_TYPE='sma', SLOPE_FILTER=False,
            CROSS_DELAY=6, TOUCH_MODE='atr', TOUCH_ATR=0.5,
            EXIT_MODE='atr', ATR_TP=4.0, ATR_SL=2.0)


# 数据区间: 默认取最新一段 (2025-01~2026-08)。设 SPAN=20230101_20250101 可切到
# 更早的样本外区间 (由 fetch_hist.py 从 Binance 拉取)。
SPAN = os.environ.get('SPAN', '')
# 周期: 默认 4h (主策略)。设 TF=1d 可切到日线数据 (海龟等长周期系统用)。
TF = os.environ.get('DATA_TF', '4h')


def path(s, tf=None):
    tf = tf or TF
    pat = f'{s}-USDT-SWAP_{tf}_{SPAN}*.csv' if SPAN else f'{s}-USDT-SWAP_{tf}_*.csv'
    h = sorted(glob.glob(os.path.join(DATA, pat)))
    return h[-1] if h else None


def norm_trades(df, p):
    """回测并返回每笔 R倍数 (跨币种可比): R = 方向·(出场-进场) / (ATR_SL·进场ATR)。
    R 是"盈亏相对止损风险的倍数", 与币价/张数无关, 是跨币汇总最干净的口径。"""
    di = mp.calc_indicators(df, p)
    ds = mp.gen_signals(di, p)
    warm = p['MA_SLOW'] + 2
    ds = ds.iloc[warm:]
    if len(ds) < 40:
        return []
    trades, _, _ = mp.run_backtest(ds, p)
    atr_at = ds['atr']                       # 按时间戳查进场ATR
    out = []
    for t in trades:
        a = atr_at.get(t['entry_time'])
        if a is None or not np.isfinite(a) or a <= 0:
            continue
        sign = 1 if t['type'] == 'LONG' else -1
        risk = p['ATR_SL'] * a
        R = sign * (t['exit_price'] - t['entry_price']) / risk
        out.append(R)
    return out


def side_stats(direction, start=None, end=None):
    """direction: 'long' or 'short'。返回逐币与汇总。"""
    p = dict(BASE, ENABLE_LONG=(direction == 'long'), ENABLE_SHORT=(direction == 'short'))
    all_pnl, per = [], []
    for s in SYMS:
        pt = path(s)
        if not pt:
            continue
        df = mp.cc.load_data(pt)
        if start is not None:
            df = df[(df.index >= start) & (df.index < end)]
        pnls = norm_trades(df, p)
        if pnls:
            arr = np.array(pnls, float)
            per.append(dict(币=s, 笔数=len(arr), 归一总盈亏=round(arr.sum(), 4),
                            胜率=round((arr > 0).mean()*100, 1)))
            all_pnl += pnls
    arr = np.array(all_pnl, float)
    n = len(arr)
    t = arr.mean()/(arr.std(ddof=1)/np.sqrt(n)) if n > 1 and arr.std(ddof=1) > 0 else 0
    win_syms = sum(1 for r in per if r['归一总盈亏'] > 0)
    return dict(dir=direction, n=n, sum=arr.sum(), mean=arr.mean() if n else 0,
                win=(arr > 0).mean()*100 if n else 0, t=t,
                win_syms=win_syms, tot_syms=len(per)), per


def full_span():
    """取所有币的公共时间范围中点, 做样本内/外切分。"""
    lo, hi = None, None
    for s in SYMS:
        pt = path(s)
        if not pt:
            continue
        df = mp.cc.load_data(pt)
        lo = df.index[0] if lo is None else min(lo, df.index[0])
        hi = df.index[-1] if hi is None else max(hi, df.index[-1])
    mid = lo + (hi - lo) / 2
    return lo, mid, hi


def show(tag, st):
    print(f'  [{tag}] 笔数={st["n"]}  归一总盈亏={st["sum"]:.3f}  平均单笔={st["mean"]:.5f}  '
          f'胜率={st["win"]:.1f}%  t={st["t"]:.2f}  盈利币={st["win_syms"]}/{st["tot_syms"]}')


def main():
    pd.set_option('display.width', 200)
    pd.set_option('display.unicode.east_asian_width', True)
    lo, mid, hi = full_span()
    print('='*88)
    print(f'做多腿 vs 做空腿 实证对比 (全{len(SYMS)}币 4h, CROSS_DELAY=6, TOUCH=0.5ATR, 出场ATR TP4/SL2)')
    print(f'数据范围: {lo.date()} ~ {hi.date()}   样本内<{mid.date()}<=样本外')
    print('='*88)

    for direction in ('long', 'short'):
        title = '做多腿(金叉回踩)' if direction == 'long' else '做空腿(死叉反弹)'
        print(f'\n### {title}')
        full, per = side_stats(direction)
        ins, _ = side_stats(direction, lo, mid)
        oos, _ = side_stats(direction, mid, hi + pd.Timedelta(hours=1))
        print(pd.DataFrame(per).to_string(index=False))
        show('全样本', full)
        show('样本内', ins)
        show('样本外', oos)


if __name__ == '__main__':
    main()
