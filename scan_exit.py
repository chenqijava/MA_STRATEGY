"""针对性扫描出场参数 —— 根因是移动止损触发过早/跟踪过松, 利润跑不起来。

维度：
  - 移动止损触发阈值 TRAIL_TRIGGER1
  - 移动止损跟踪宽度 TRAIL_ATR_MULT
  - 目标止盈倍数 ATR_TP_MULT
  - 特例：关掉移动止损(纯目标止盈+EMA止损)
只跑趋势-突破单腿(最干净的基线), 定位出场机制。
"""
import os, sys, io, itertools
import pandas as pd
import combo_core as cc
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def run(tf, overrides):
    df = cc.load_data(os.path.join(DATA, f'ETH-USDT-SWAP_{tf}_20250101_20250801.csv'))
    p = dict(cc.DEFAULTS, ENABLE_TREND=True, ENABLE_MR=False, TREND_ENTRY='breakout', **overrides)
    di = cc.calc_indicators(df, p)
    m, trades, curve = cc.evaluate(di, p)
    return m


def main():
    triggers = [1.5, 2.5, 3.5]
    widths = [2.0, 3.0, 4.0]
    tps = [3.0, 5.0, 8.0]
    for tf in ['5m', '15m']:
        rows = []
        # 特例：关掉移动止损(用极大触发阈值使其永不激活)
        for tp in tps:
            m = run(tf, dict(TRAIL_TRIGGER1=999, ATR_TP_MULT=tp))
            rows.append(('无移动止损', 999, '-', tp, m))
        for trig, w, tp in itertools.product(triggers, widths, tps):
            m = run(tf, dict(TRAIL_TRIGGER1=trig, TRAIL_ATR_MULT=w, ATR_TP_MULT=tp))
            rows.append(('移动止损', trig, w, tp, m))
        res = pd.DataFrame([dict(
            触发=trig, 跟踪宽=w, 止盈=tp, 交易=m['trades'],
            年化=round(m['ann_ret'] * 100, 1), 夏普=round(m['sharpe'], 2),
            回撤=round(m['mdd'] * 100, 1), 胜率=round(m['win'] * 100, 1),
            盈亏比=round(m['pf'], 2) if m['pf'] != float('inf') else 99,
        ) for _, trig, w, tp, m in rows])
        res = res.sort_values('夏普', ascending=False)
        pd.set_option('display.unicode.east_asian_width', True)
        pd.set_option('display.width', 200)
        print(f'\n===== {tf} 出场参数扫描 (按夏普降序, 趋势-突破单腿) =====')
        print(res.head(12).to_string(index=False))


if __name__ == '__main__':
    main()
