"""牛熊判定器横评: BTC单价 vs 市场宽度。

判定器好不好, 标准不是"看起来像不像牛市", 而是**它标出的牛市段里做空腿是否真的更差**。
所以直接用 分段做空均R 的差值 (熊市段均R − 牛市段均R) 当打分: 差值越大, 分辨力越强。
同时看两段(2023-24 牛 / 2025-26 熊)是否一致 —— 只在一段有效的判定器不可用。

用法:
  UNIVERSE=universe_live.txt MA_SLOW=30 MAX_BARS=40 python test_regime.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import importlib
import analyze_long as al
import ma_pullback as mp

SPANS = [('20230101_20250101', '2023-24牛'), ('20250101_20260801', '2025-26熊')]
MA_SLOW = int(os.environ.get('MA_SLOW', '30'))
MAX_BARS = int(os.environ.get('MAX_BARS', '40'))


def syms():
    u = os.environ.get('UNIVERSE', 'universe_live.txt')
    return [x.strip() for x in open(os.path.join(os.path.dirname(__file__), u),
                                    encoding='utf-8').read().split(',') if x.strip()]


def short_trades(span, ss):
    """返回 DataFrame(entry_date, R): 做空腿逐笔 R 倍数 + 入场日期。"""
    os.environ['SPAN'] = span
    importlib.reload(al)
    p = dict(al.BASE, ENABLE_LONG=False, ENABLE_SHORT=True,
             MA_SLOW=MA_SLOW, MAX_BARS=MAX_BARS)
    rows = []
    for s in ss:
        pa = al.path(s)
        if not pa:
            continue
        try:
            df = mp.cc.load_data(pa)
            di = mp.calc_indicators(df, p)
            ds = mp.gen_signals(di, p)
            ds = ds.iloc[p['MA_SLOW'] + 2:]
            trades, _, _ = mp.run_backtest(ds, p)
            atrs = ds['atr']
            for t in trades:
                a = atrs.get(t['entry_time'], np.nan)
                if not np.isfinite(a) or a <= 0:
                    continue
                # R = 盈亏 / 单位风险; 用价格口径避免依赖张数/面值
                r = -(t['exit_price'] - t['entry_price']) / (p['ATR_SL'] * a)
                rows.append((t['entry_time'], r))
        except Exception:
            pass
    return pd.DataFrame(rows, columns=['t', 'R'])


def daily_close(span, s):
    os.environ['SPAN'] = span
    importlib.reload(al)
    pa = al.path(s)
    return mp.cc.load_data(pa)['close'].resample('1D').last().dropna() if pa else None


def btc_mask(span, ma_n, slope):
    b = daily_close(span, 'BTC')
    ma = b.rolling(ma_n).mean()
    return ((b > ma) & (ma > ma.shift(slope))).shift(1).fillna(False)


def breadth_raw(span, ss, ma_n):
    """宽度序列: 币池中 日收盘 > 自身MA(ma_n) 的比例 (逐币独立, 无前视)。"""
    os.environ['SPAN'] = span
    importlib.reload(al)
    flags = {}
    for s in ss:
        pa = al.path(s)
        if not pa:
            continue
        c = mp.cc.load_data(pa)['close'].resample('1D').last().dropna()
        if len(c) < ma_n + 5:
            continue
        ma = c.rolling(ma_n).mean()
        flags[s] = (c > ma).where(ma.notna())
    F = pd.DataFrame(flags)
    return (F.sum(axis=1) / F.notna().sum(axis=1).replace(0, np.nan)).dropna()


def score(tr, bull_daily):
    """按判定器把逐笔R分成牛/熊两组, 返回 (熊均R, 牛均R, 差值, 牛市段交易占比)。"""
    days = pd.Series(bull_daily.values, index=bull_daily.index.normalize())
    isbull = tr['t'].dt.normalize().map(days).fillna(False).astype(bool).values
    R = tr['R'].values
    if isbull.sum() < 30 or (~isbull).sum() < 30:
        return None
    rb, rl = R[~isbull].mean(), R[isbull].mean()
    return rb, rl, rb - rl, isbull.mean()


def main():
    ss = syms()
    # 逐段准备一次交易 + 日线数据, 后面所有判定器复用
    data = {}
    for span, lab in SPANS:
        tr = short_trades(span, ss)
        data[span] = dict(tr=tr, lab=lab,
                          br={n: breadth_raw(span, ss, n) for n in (20, 50, 100, 200)})
        print(f'{lab}: 做空 {len(tr)} 笔, 全样本均R {tr.R.mean():+.4f}')

    cands = []
    for ma_n in (20, 35, 50, 100):
        for sl in (5, 10, 20):
            cands.append((f'BTC MA{ma_n}/斜率{sl}', 'btc', (ma_n, sl)))
    for ma_n in (20, 50, 100, 200):
        for th in (0.3, 0.4, 0.5, 0.6, 0.7):
            cands.append((f'宽度 MA{ma_n}>{th:.0%}', 'br', (ma_n, th)))

    print(f'\n{"判定器":20s}' + ''.join(f'{lab:>34s}' for _, lab in SPANS) + f'{"两段最小差":>12s}')
    print(f'{"":20s}' + ''.join(f'{"熊均R":>9s}{"牛均R":>9s}{"差值":>9s}{"牛占比":>7s}' for _ in SPANS))
    print('-' * 116)
    rows = []
    for name, kind, args in cands:
        cells, diffs = [], []
        for span, _ in SPANS:
            d = data[span]
            if kind == 'btc':
                m = btc_mask(span, *args)
            else:
                ma_n, th = args
                m = (d['br'][ma_n] > th).shift(1).fillna(False)
            sc = score(d['tr'], m)
            if sc is None:
                cells.append(f'{"样本不足":>34s}')
                diffs.append(-9)
                continue
            rb, rl, df_, frac = sc
            cells.append(f'{rb:+9.3f}{rl:+9.3f}{df_:+9.3f}{frac:7.0%}')
            diffs.append(df_)
        mn = min(diffs)
        rows.append((mn, name, ''.join(cells)))
        print(f'{name:20s}' + ''.join(cells) + f'{mn:+12.3f}')

    print('\n按"两段最小差值"排序 (两段都要有分辨力才算好):')
    for mn, name, _ in sorted(rows, reverse=True)[:6]:
        print(f'  {name:20s} 最小差 {mn:+.3f}')


if __name__ == '__main__':
    main()
