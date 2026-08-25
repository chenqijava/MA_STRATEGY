"""跨币种显著性验证：同一套固定SMA参数, 跑多个币种的4h, 看正收益是普遍还是ETH特例。

关键: 参数固定不再按币种优化 —— 这才是诚实的样本外/跨标的检验。
用之前"两年都正"里最稳的一档: SMA, TOUCH_TH=10, 斜率过滤关, 出场破MA24。
汇总所有(币种×年份)交易, 用配对统计判断是否显著异于0。
"""
import os
import glob
import numpy as np
import pandas as pd
import ma_pullback as mp

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']
# 固定参数(不按币种调)
FIXED = dict(mp.DEFAULTS, MA_TYPE='sma', TOUCH_TH=10, SLOPE_FILTER=False)


def find_4h(sym):
    hits = glob.glob(os.path.join(DATA, f'{sym}-USDT-SWAP_4h_*.csv'))
    return sorted(hits)[-1] if hits else None


def yr(df, y):
    return df[(df.index >= pd.Timestamp(f'{y}-01-01')) & (df.index < pd.Timestamp(f'{y+1}-01-01'))]


def main():
    rows = []
    all_trade_pnls = []          # 汇总所有单笔收益率(相对入场名义), 用于显著性
    for sym in SYMBOLS:
        path = find_4h(sym)
        if not path:
            print(f'[{sym}] 无数据, 跳过')
            continue
        full = mp.cc.load_data(path)
        for y in [2025, 2026]:
            df = yr(full, y)
            if len(df) < 100:
                continue
            m, trades, curve = mp.evaluate(df, dict(FIXED), label=f'{sym}-{y}')
            rows.append(dict(币种=sym, 年=y, 交易=m['trades'],
                             年化=round(m['ann_ret'] * 100, 1), 夏普=round(m['sharpe'], 2),
                             回撤=round(m['mdd'] * 100, 1), 胜率=round(m['win'] * 100, 1),
                             盈亏比=round(m['pf'], 2) if m['pf'] != float('inf') else 99,
                             总盈亏=round(sum(t['pnl'] for t in trades), 1)))
            # 单笔收益率(pnl / 名义), 近似等权, 用于跨样本显著性
            for t in trades:
                notional = abs(t['entry_price']) * mp.CONTRACT_VALUE
                all_trade_pnls.append(t['pnl'])

    r = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    pd.set_option('display.unicode.east_asian_width', True)
    print('=' * 80)
    print('跨币种 4h SMA金叉回踩 (固定参数 TH=10/斜率关/出场破MA24, 未按币种优化)')
    print('=' * 80)
    print(r.to_string(index=False))

    # 汇总显著性
    pnls = np.array(all_trade_pnls, dtype=float)
    n = len(pnls)
    print(f'\n{"-"*80}\n跨币种汇总:')
    print(f'  样本×年份组合数={len(r)}  正夏普占比={(r["夏普"]>0).mean()*100:.0f}%  '
          f'正年化占比={(r["年化"]>0).mean()*100:.0f}%')
    print(f'  总交易笔数={n}  总盈亏={pnls.sum():.1f}  平均单笔={pnls.mean():.2f}  胜率={ (pnls>0).mean()*100:.1f}%')
    if n > 1 and pnls.std(ddof=1) > 0:
        t_stat = pnls.mean() / (pnls.std(ddof=1) / np.sqrt(n))
        print(f'  单笔盈亏 t检验: t={t_stat:.2f}  (|t|>1.96 才算95%显著异于0)')
        sig = '显著为正 → edge可能真实存在' if t_stat > 1.96 else \
              ('显著为负 → 策略系统性亏损' if t_stat < -1.96 else '不显著 → 无法证明有edge(与随机不可区分)')
        print(f'  判定: {sig}')
    # 每币种两年是否一致为正
    print(f'\n  各币种一致性(两年是否都正年化):')
    for sym in SYMBOLS:
        s = r[r['币种'] == sym]
        if len(s) == 2:
            both = (s['年化'] > 0).all()
            print(f'    {sym}: 2025={s[s.年==2025]["年化"].iloc[0]}%  '
                  f'2026={s[s.年==2026]["年化"].iloc[0]}%  {"都正" if both else ""}')


if __name__ == '__main__':
    main()
