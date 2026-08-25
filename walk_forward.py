"""Walk-forward 滚动窗口验证 —— 纯趋势腿。

目的：避免"整段样本内挑最优"的过拟合。做法：
  训练窗(3个月)选参 → 紧接测试窗(1个月)记录样本外表现 → 步进1个月, 滚动前进。
拼接所有测试窗的样本外结果, 才是对真实 edge 的诚实估计。

数据有缺口(2025.1-7 与 2026.1-7), 故在每年 7 个月内各自滚动。
参数网格刻意小(只扫已确认影响最大的出场维度), 网格越大每折越易过拟合。
"""
import os, sys, io, itertools
import numpy as np
import pandas as pd
import combo_core as cc
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
YEARS = {
    '2025': {'5m': 'ETH-USDT-SWAP_5m_20250101_20250801.csv',
             '15m': 'ETH-USDT-SWAP_15m_20250101_20250801.csv'},
    '2026': {'5m': 'ETH-USDT-SWAP_5m_20260101_20260801.csv',
             '15m': 'ETH-USDT-SWAP_15m_20260101_20260801.csv'},
}

# 小参数网格：只扫出场(已确认是主因)。触发999=关移动止损。
GRID = [
    dict(TRAIL_TRIGGER1=999, ATR_TP_MULT=3.0),
    dict(TRAIL_TRIGGER1=999, ATR_TP_MULT=5.0),
    dict(TRAIL_TRIGGER1=3.5, TRAIL_ATR_MULT=2.0, ATR_TP_MULT=5.0),
]

TRAIN_MONTHS = 3
TEST_MONTHS = 1


def backtest_slice(df_ind, p, start, end):
    """在已算好指标的df上, 取[start,end)时段回测, 返回metrics。"""
    df_s = cc.gen_signals(df_ind.copy(), p)
    df_s = df_s[(df_s.index >= start) & (df_s.index < end)]
    if len(df_s) < 50:
        return None
    trades, equity, curve = cc.run_backtest(df_s, p)
    days = (df_s.index[-1] - df_s.index[0]).days or 1
    return cc.metrics(trades, curve, df_s.index, days)


def month_starts(idx):
    ms = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq='MS')
    return [m for m in ms if m >= idx[0]]


def walk(tf):
    oos_metrics = []          # 每个测试月的样本外metric
    chosen = []               # 每折选中的参数
    for yr, files in YEARS.items():
        df = cc.load_data(os.path.join(DATA, files[tf]))
        # 每个参数组只算一次指标(出场参数不影响指标, 但止盈影响止损→仍在回测层, 指标层相同)
        # 指标层与出场参数无关, 用默认算一次即可
        base = dict(cc.DEFAULTS, TREND_ENTRY='breakout')
        df_ind = cc.calc_indicators(df, base)
        months = month_starts(df.index)
        # 滚动：train=[i, i+3), test=[i+3, i+4)
        for i in range(len(months) - TRAIN_MONTHS):
            tr_start = months[i]
            tr_end = months[i + TRAIN_MONTHS]
            te_end_idx = i + TRAIN_MONTHS + TEST_MONTHS
            if te_end_idx > len(months):
                te_end = df.index[-1] + pd.Timedelta(seconds=1)
            else:
                te_end = months[i + TRAIN_MONTHS + TEST_MONTHS - 1] + pd.offsets.MonthBegin(1)
            te_start = tr_end
            # 训练：在train窗选夏普最高的参数
            best_p, best_sh = None, -1e9
            for g in GRID:
                p = dict(base, **g)
                m = backtest_slice(df_ind, p, tr_start, tr_end)
                if m and m['trades'] >= 5 and m['sharpe'] > best_sh:
                    best_sh, best_p = m['sharpe'], p
            if best_p is None:
                continue
            # 测试：用选中参数在test窗跑样本外
            mt = backtest_slice(df_ind, best_p, te_start, te_end)
            if mt is None:
                continue
            g_used = {k: best_p[k] for k in ('TRAIL_TRIGGER1', 'ATR_TP_MULT')}
            chosen.append(g_used)
            oos_metrics.append(dict(年=yr, 测试月=str(te_start.date())[:7],
                                    交易=mt['trades'], 收益=mt['ret'],
                                    夏普=mt['sharpe'], 回撤=mt['mdd'],
                                    胜率=mt['win'], 选参止盈=g_used['ATR_TP_MULT'],
                                    移动止损=('关' if g_used['TRAIL_TRIGGER1'] == 999 else '开')))
    return oos_metrics


def summarize(tf, oos):
    if not oos:
        print(f'[{tf}] 无有效折')
        return
    d = pd.DataFrame(oos)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 200)
    print(f'\n{"="*80}\n[{tf}] Walk-forward 样本外逐月结果 (train {TRAIN_MONTHS}月/test {TEST_MONTHS}月)\n{"="*80}')
    show = d.copy()
    show['收益'] = (show['收益'] * 100).round(1).astype(str) + '%'
    show['夏普'] = show['夏普'].round(2)
    show['回撤'] = (show['回撤'] * 100).round(1).astype(str) + '%'
    show['胜率'] = (show['胜率'] * 100).round(1).astype(str) + '%'
    print(show.to_string(index=False))
    # 拼接样本外统计
    pos = (d['夏普'] > 0).mean()
    print(f'\n  样本外月数={len(d)}  平均夏普={d["夏普"].mean():.2f}  '
          f'正夏普月占比={pos*100:.0f}%  平均月收益={d["收益"].mean()*100:.2f}%  '
          f'累计收益≈{(np.prod(1+d["收益"])-1)*100:.1f}%')
    verdict = '稳健(样本外正收益且过半月为正)' if pos > 0.5 and d['收益'].mean() > 0 \
        else '不稳健(样本外无一致 edge)'
    print(f'  判定: {verdict}')


def main():
    for tf in ['5m', '15m']:
        summarize(tf, walk(tf))


if __name__ == '__main__':
    main()
