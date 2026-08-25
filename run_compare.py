"""批量对比：5m/15m × {纯趋势基线, 纯回归基线, 结合} × 回归信号{bbands, zscore}。

验收标准(见 设计方案.md §8)：结合策略的夏普/回撤应优于任一单腿, 否则"结合"无正贡献。
用法：python run_compare.py
"""
import os
import sys
import io
import pandas as pd
import combo_core as cc

# Windows 控制台默认 GBK, 强制 UTF-8 输出
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
# 用 2025 全年样本(样本内)。样本外验证可换 2026 文件。
FILES = {
    '5m': 'ETH-USDT-SWAP_5m_20250101_20250801.csv',
    '15m': 'ETH-USDT-SWAP_15m_20250101_20250801.csv',
}


def variants():
    base = cc.DEFAULTS
    out = []
    # 纯趋势基线
    out.append(('趋势基线', dict(base, ENABLE_TREND=True, ENABLE_MR=False)))
    # 纯回归基线(两种信号)
    out.append(('回归基线-bb', dict(base, ENABLE_TREND=False, ENABLE_MR=True, MR_SIGNAL='bbands')))
    out.append(('回归基线-z', dict(base, ENABLE_TREND=False, ENABLE_MR=True, MR_SIGNAL='zscore')))
    # 趋势腿：回踩 vs 突破 对照(单腿)
    out.append(('趋势-回踩', dict(base, ENABLE_TREND=True, ENABLE_MR=False, TREND_ENTRY='pullback')))
    out.append(('趋势-突破', dict(base, ENABLE_TREND=True, ENABLE_MR=False, TREND_ENTRY='breakout')))
    # 结合(回踩趋势 + bbands回归)
    out.append(('结合-回踩', dict(base, ENABLE_TREND=True, ENABLE_MR=True,
                                 TREND_ENTRY='pullback', MR_SIGNAL='bbands')))
    return out


def main():
    rows = []
    for tf, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f'[跳过] 找不到 {path}')
            continue
        df = cc.load_data(path)
        for name, p in variants():
            df_ind = cc.calc_indicators(df, p)
            m, trades, curve = cc.evaluate(df_ind, p, label=f'{tf}|{name}')
            rows.append(dict(
                周期=tf, 策略=name, 交易数=m['trades'],
                年化=f"{m['ann_ret']*100:.1f}%", 夏普=round(m['sharpe'], 2),
                最大回撤=f"{m['mdd']*100:.1f}%", 胜率=f"{m['win']*100:.1f}%",
                盈亏比=round(m['pf'], 2) if m['pf'] != float('inf') else 'inf',
                趋势腿笔数=m['tr_trades'], 回归腿笔数=m['mr_trades'],
                趋势腿盈亏=round(m['tr_pnl'], 1), 回归腿盈亏=round(m['mr_pnl'], 1),
                期末权益=round(m['final'], 1),
            ))

    res = pd.DataFrame(rows)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 200)
    print('\n' + '=' * 100)
    print('趋势 + 均值回归结合策略 — 回测对比 (ETH-USDT-SWAP, 2025样本内)')
    print('=' * 100)
    print(res.to_string(index=False))

    outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, 'compare.csv')
    res.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n结果已保存: {csv_path}')

    # 验收判定
    print('\n' + '-' * 100)
    print('验收判定(结合 vs 单腿基线, 同周期):')
    for tf in FILES:
        sub = res[res['周期'] == tf]
        if sub.empty:
            continue
        def sharpe_of(name):
            r = sub[sub['策略'] == name]['夏普']
            return r.iloc[0] if len(r) else None
        pb = sharpe_of('趋势-回踩')
        bo = sharpe_of('趋势-突破')
        combo = sharpe_of('结合-回踩')
        print(f'  [{tf}] 趋势-回踩夏普={pb}  趋势-突破夏普={bo}')
        if pb is not None and bo is not None:
            print(f'        回踩 vs 突破: {"回踩更优" if pb > bo else "突破更优"}')
        if combo is not None and pb is not None:
            v = '通过[PASS]' if combo > pb else '未通过[FAIL] (结合未超回踩单腿)'
            print(f'        结合-回踩夏普={combo}  → {v}')


if __name__ == '__main__':
    main()
