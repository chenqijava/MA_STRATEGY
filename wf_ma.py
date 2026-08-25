"""4h MA5/MA24 金叉回踩 —— Walk-forward 滚动验证。

数据连续(2025.1~2026.7, 约19个月), 适合真正的滚动:
  训练窗(TRAIN月)选参 → 紧接测试窗(TEST月)盲测 → 步进TEST月, 滚动前进。
拼接所有测试窗结果, 才是"选参不靠后见之明"时的真实表现。

参数网格刻意小(仅出场已固定为破MA24; 搜 TOUCH_TH × 斜率过滤), 网格越大越易过拟合。
选参目标: 训练窗夏普最高, 且要求训练窗至少 MIN_TRADES 笔(避免选到偶然的少样本高分)。
"""
import os
import numpy as np
import pandas as pd
import ma_pullback as mp

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
FILE = 'ETH-USDT-SWAP_4h_20250101_20260801.csv'

GRID = [dict(TOUCH_TH=th, SLOPE_FILTER=f)
        for th in (10, 20, 40) for f in (True, False)]

TRAIN_MONTHS = 6
TEST_MONTHS = 2
MIN_TRADES = 8          # 训练窗最少交易数, 否则该参数不参与选择


def backtest_slice(full, p, start, end):
    di = mp.calc_indicators(full, p)
    ds = mp.gen_signals(di, p)
    ds = ds[(ds.index >= start) & (ds.index < end)]
    if len(ds) < 30:
        return None
    trades, equity, curve = mp.run_backtest(ds, p)
    days = (ds.index[-1] - ds.index[0]).days or 1
    return mp.cc.metrics(trades, curve, ds.index, days)


def main():
    full = mp.cc.load_data(os.path.join(DATA, FILE))
    months = pd.date_range(full.index[0].normalize().replace(day=1),
                           full.index[-1], freq='MS')
    oos = []
    i = 0
    while i + TRAIN_MONTHS < len(months):
        tr_start = months[i]
        tr_end = months[i + TRAIN_MONTHS]
        te_start = tr_end
        te_end_i = i + TRAIN_MONTHS + TEST_MONTHS
        te_end = months[te_end_i] if te_end_i < len(months) else full.index[-1] + pd.Timedelta(hours=1)

        # 训练：选训练窗夏普最高(且交易数达标)的参数
        best_p, best_sh, best_m = None, -1e9, None
        for g in GRID:
            p = dict(mp.DEFAULTS, **g)
            m = backtest_slice(full, p, tr_start, tr_end)
            if m and m['trades'] >= MIN_TRADES and m['sharpe'] > best_sh:
                best_sh, best_p, best_m = m['sharpe'], p, m
        if best_p is None:
            i += TEST_MONTHS
            continue
        # 测试：用选中参数在紧接测试窗盲测
        mt = backtest_slice(full, best_p, te_start, te_end)
        if mt:
            oos.append(dict(
                测试窗=f'{str(te_start.date())[:7]}~{str((te_end - pd.Timedelta(days=1)).date())[:7]}',
                选中TH=best_p['TOUCH_TH'], 斜率=('开' if best_p['SLOPE_FILTER'] else '关'),
                训练夏普=round(best_sh, 2),
                测试交易=mt['trades'], 测试收益=mt['ret'], 测试夏普=mt['sharpe'],
                测试胜率=mt['win'], 测试盈亏比=(mt['pf'] if mt['pf'] != float('inf') else 99)))
        i += TEST_MONTHS

    if not oos:
        print('无有效折')
        return
    d = pd.DataFrame(oos)
    show = d.copy()
    show['训练夏普'] = show['训练夏普'].round(2)
    show['测试收益'] = (show['测试收益'] * 100).round(1).astype(str) + '%'
    show['测试夏普'] = show['测试夏普'].round(2)
    show['测试胜率'] = (show['测试胜率'] * 100).round(1).astype(str) + '%'
    show['测试盈亏比'] = show['测试盈亏比'].round(2)
    pd.set_option('display.width', 220)
    pd.set_option('display.unicode.east_asian_width', True)
    print('=' * 90)
    print(f'4h MA金叉回踩 Walk-forward (train {TRAIN_MONTHS}月/test {TEST_MONTHS}月, 出场破MA24)')
    print('=' * 90)
    print(show.to_string(index=False))

    # 拼接样本外统计
    tot_trades = d['测试交易'].sum()
    pos = (d['测试夏普'] > 0).mean()
    cum = (np.prod(1 + d['测试收益']) - 1) * 100
    print(f'\n样本外汇总: 折数={len(d)}  总交易={tot_trades}  '
          f'平均测试夏普={d["测试夏普"].mean():.2f}  正夏普折占比={pos*100:.0f}%')
    print(f'          平均每折收益={d["测试收益"].mean()*100:.2f}%  拼接累计收益≈{cum:.1f}%')
    if pos > 0.5 and d['测试收益'].mean() > 0 and tot_trades >= 40:
        v = '稳健: 滚动选参下样本外仍正, 且交易数尚可 → 微弱但可能真实的edge'
    elif pos > 0.5 and d['测试收益'].mean() > 0:
        v = '存疑: 样本外为正但总交易过少, 统计不显著'
    else:
        v = '证伪: 滚动选参下样本外不成立, 之前的正夏普是样本内运气'
    print(f'判定: {v}')


if __name__ == '__main__':
    main()
