"""最终验证：固化最优出场参数 → 样本内(2025)结合对比 → 样本外(2026)验证。

根因已定位(见 scan_exit)：原移动止损触发过早/跟踪过松, 利润跑不起来。
最优出场：关闭移动止损(TRAIL_TRIGGER1=999), 纯目标止盈 + EMA反向 + 时间止损。
按周期固化止盈倍数：5m→3ATR, 15m→5ATR。
"""
import os, sys, io
import pandas as pd
import combo_core as cc
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')

# 固化的最优出场(按周期)
BEST_EXIT = {
    '5m': dict(TRAIL_TRIGGER1=999, ATR_TP_MULT=3.0),
    '15m': dict(TRAIL_TRIGGER1=999, ATR_TP_MULT=5.0),
}
FILES = {
    'in':  {'5m': 'ETH-USDT-SWAP_5m_20250101_20250801.csv',
            '15m': 'ETH-USDT-SWAP_15m_20250101_20250801.csv'},
    'out': {'5m': 'ETH-USDT-SWAP_5m_20260101_20260801.csv',
            '15m': 'ETH-USDT-SWAP_15m_20260101_20260801.csv'},
}


def cfg(tf, name):
    base = dict(cc.DEFAULTS, TREND_ENTRY='breakout', **BEST_EXIT[tf])
    if name == '趋势单腿':
        return dict(base, ENABLE_TREND=True, ENABLE_MR=False)
    if name == '回归单腿':
        return dict(base, ENABLE_TREND=False, ENABLE_MR=True, MR_SIGNAL='bbands')
    if name == '结合':
        return dict(base, ENABLE_TREND=True, ENABLE_MR=True, MR_SIGNAL='bbands')


def run(sample, tf, name):
    path = os.path.join(DATA, FILES[sample][tf])
    if not os.path.exists(path):
        return None
    df = cc.load_data(path)
    p = cfg(tf, name)
    di = cc.calc_indicators(df, p)
    m, _, _ = cc.evaluate(di, p)
    return m


def table(sample, title):
    rows = []
    for tf in ['5m', '15m']:
        for name in ['趋势单腿', '回归单腿', '结合']:
            m = run(sample, tf, name)
            if m is None:
                continue
            rows.append(dict(
                周期=tf, 策略=name, 交易=m['trades'],
                年化=f"{m['ann_ret']*100:.1f}%", 夏普=round(m['sharpe'], 2),
                回撤=f"{m['mdd']*100:.1f}%", 胜率=f"{m['win']*100:.1f}%",
                盈亏比=round(m['pf'], 2) if m['pf'] != float('inf') else 'inf',
                趋势腿盈亏=round(m['tr_pnl'], 1), 回归腿盈亏=round(m['mr_pnl'], 1),
            ))
    res = pd.DataFrame(rows)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 200)
    print(f'\n{"="*90}\n{title}\n{"="*90}')
    print(res.to_string(index=False))
    return res


def main():
    r_in = table('in', '样本内 (2025-01~08)  固化最优出场')
    r_out = table('out', '样本外 (2026-01~08)  同参数, 未再优化')

    print(f'\n{"-"*90}\n结论:')
    for tf in ['5m', '15m']:
        def get(res, name, col='夏普'):
            s = res[(res['周期'] == tf) & (res['策略'] == name)][col]
            return s.iloc[0] if len(s) else None
        ti, to = get(r_in, '趋势单腿'), get(r_out, '趋势单腿')
        ci, co = get(r_in, '结合'), get(r_out, '结合')
        print(f'  [{tf}] 趋势单腿夏普 样本内={ti} 样本外={to}')
        print(f'        结合夏普     样本内={ci} 样本外={co}')
        if ci is not None and ti is not None:
            print(f'        结合是否优于趋势单腿(样本内): {"是" if ci > ti else "否"}')
        if to is not None:
            print(f'        样本外是否仍为正夏普: {"是, edge较稳" if to > 0 else "否, 疑似样本内过拟合"}')


if __name__ == '__main__':
    main()
