"""在 1h / 4h 上跑 MA5/MA24 金叉回踩策略, 样本内(2025)/样本外(2026)切分对比。

大周期价位波动绝对值更大, TOUCH_TH(回踩容差)需相应放大 —— 做成按周期缩放:
  用 ATR 或价格比例更稳健, 这里先按周期给一组候选 TOUCH_TH 对比。
"""
import os
import pandas as pd
import ma_pullback as mp

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
FILE = {'1h': 'ETH-USDT-SWAP_1h_20250101_20260801.csv',
        '4h': 'ETH-USDT-SWAP_4h_20250101_20260801.csv'}

# 回踩容差候选(绝对价差, ETH约1800-4000). 大周期波动大, 给多档对比。
TOUCH_CANDS = [10, 20, 40, 80]


def split_year(df, yr):
    lo = pd.Timestamp(f'{yr}-01-01'); hi = pd.Timestamp(f'{yr+1}-01-01')
    return df[(df.index >= lo) & (df.index < hi)]


def main():
    for tf in ['1h', '4h']:
        path = os.path.join(DATA, FILE[tf])
        if not os.path.exists(path):
            print(f'[跳过] 缺 {path}')
            continue
        full = mp.cc.load_data(path)
        rows = []
        for th in TOUCH_CANDS:
            for filt in [True, False]:
                for yr in [2025, 2026]:
                    df = split_year(full, yr)
                    if len(df) < 100:
                        continue
                    p = dict(mp.DEFAULTS, TOUCH_TH=th, SLOPE_FILTER=filt)
                    m, trades, curve = mp.evaluate(df, p)
                    rows.append(dict(
                        TH=th, 过滤=('开' if filt else '关'), 年=yr,
                        交易=m['trades'], 年化=round(m['ann_ret'] * 100, 1),
                        夏普=round(m['sharpe'], 2), 回撤=round(m['mdd'] * 100, 1),
                        胜率=round(m['win'] * 100, 1),
                        盈亏比=round(m['pf'], 2) if m['pf'] != float('inf') else 99))
        r = pd.DataFrame(rows)
        pd.set_option('display.width', 220)
        pd.set_option('display.unicode.east_asian_width', True)
        print(f'\n{"="*80}\n[{tf}] MA5/MA24 金叉回踩 (出场破MA24)\n{"="*80}')
        print(r.to_string(index=False))
        # 挑样本内外都为正的稳健配置
        print(f'\n  样本内外均为正夏普的配置:')
        piv = r.pivot_table(index=['TH', '过滤'], columns='年', values='夏普')
        good = piv[(piv.get(2025, -9) > 0) & (piv.get(2026, -9) > 0)]
        print(good if len(good) else '    无(样本内外一致为正的配置不存在)')


if __name__ == '__main__':
    main()
