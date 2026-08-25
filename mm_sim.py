"""高频做市回测 (秒级K线)。

【必须先明确这类回测的上限】1s K线只有 OHLCV, 没有盘口。因此有三件事无法建模:
  1. 真实买卖价差 —— 只能假设一个固定 spread, 而真实 spread 随时间变化
  2. 排队位置与成交概率 —— 只能假设"价格触及挂单价即成交"。真实做市是排在队列里,
     价格仅仅"触及"你的价位时通常**不会**成交(前面还有单), 必须价格穿过才有较高概率。
  3. 逆向选择 —— 你的单被成交, 往往正是因为市场朝不利方向走(有信息的对手方吃你)。
     OHLCV 回测天然低估这一项, 而它是做市盈亏的决定性因素。
所以下面的结果只能用来**排除明显不可行的参数**, 不能当收益预期。乐观偏差是系统性的。

模型:
  每秒在 mid×(1∓spread/2) 挂买卖单, 单笔 QTY。
  成交判定: 该秒 low <= 买价 则买单成交; high >= 卖价 则卖单成交 (可用 FILL_PROB 打折)
  库存管理: 净头寸超过 MAX_INV 则停止同向挂单(防单边累积)
  盈亏 = 已实现价差收益 − 手续费 + 库存浮动盈亏

用法:
  python mm_sim.py                                 # 默认参数
  SPREAD_BP=5 FEE_BP=1 python mm_sim.py            # 自定义
"""
import os
import sys
import numpy as np
import pandas as pd

if sys.stdout.encoding and not sys.stdout.encoding.lower().startswith('utf'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache', 'BTC-USDT_1s_1day.csv')
SPREAD_BP = float(os.environ.get('SPREAD_BP', '5'))      # 挂单价差(bp, 双边总和)
FEE_BP = float(os.environ.get('FEE_BP', '1'))            # 单边手续费(bp)。maker费通常1-2bp
QTY = float(os.environ.get('QTY', '0.01'))               # 每单数量(BTC)
MAX_INV = float(os.environ.get('MAX_INV', '0.1'))        # 净库存上限(BTC)
FILL_PROB = float(os.environ.get('FILL_PROB', '1.0'))    # 触及即成交的概率(1=乐观上限)
INIT = 10000.0


def load():
    d = pd.read_csv(DATA, parse_dates=['timestamp'])
    return d


def simulate(d, spread_bp=None, fee_bp=None, qty=None, max_inv=None, fill_prob=None, seed=42):
    sp = (spread_bp if spread_bp is not None else SPREAD_BP) / 1e4
    fee = (fee_bp if fee_bp is not None else FEE_BP) / 1e4
    q = qty if qty is not None else QTY
    mi = max_inv if max_inv is not None else MAX_INV
    fp = fill_prob if fill_prob is not None else FILL_PROB
    rng = np.random.default_rng(seed)

    o, h, l, c = d['open'].values, d['high'].values, d['low'].values, d['close'].values
    cash = 0.0          # 已实现现金流(含手续费)
    inv = 0.0           # 净库存(BTC)
    n_buy = n_sell = 0
    fees = 0.0
    eq = np.empty(len(d))
    for i in range(len(d)):
        mid = o[i]
        bid = mid * (1 - sp / 2)
        ask = mid * (1 + sp / 2)
        # 买单: 该秒最低价触及 bid; 库存已满则不挂
        if l[i] <= bid and inv < mi and (fp >= 1.0 or rng.random() < fp):
            cash -= bid * q
            f = bid * q * fee
            cash -= f
            fees += f
            inv += q
            n_buy += 1
        # 卖单: 该秒最高价触及 ask; 库存为负超限则不挂
        if h[i] >= ask and inv > -mi and (fp >= 1.0 or rng.random() < fp):
            cash += ask * q
            f = ask * q * fee
            cash -= f
            fees += f
            inv -= q
            n_sell += 1
        eq[i] = INIT + cash + inv * c[i]      # 盯市: 现金 + 库存市值
    return dict(curve=eq, cash=cash, inv=inv, n_buy=n_buy, n_sell=n_sell, fees=fees)


def report(d, r, label=''):
    eq = r['curve']
    pnl = eq[-1] - INIT
    hours = (d['timestamp'].iloc[-1] - d['timestamp'].iloc[0]).total_seconds() / 3600
    dd = np.max(1 - eq / np.maximum.accumulate(eq)) * 100
    n = r['n_buy'] + r['n_sell']
    return dict(label=label, pnl=pnl, ann=pnl / INIT * (365 * 24 / hours) * 100,
                dd=dd, n=n, fees=r['fees'], inv=r['inv'],
                per=pnl / n if n else 0)


def main():
    d = load()
    hours = (d['timestamp'].iloc[-1] - d['timestamp'].iloc[0]).total_seconds() / 3600
    px = d['close']
    print('=' * 96)
    print(f'BTC 1s 做市回测  {d.timestamp.iloc[0]} ~ {d.timestamp.iloc[-1]}  ({len(d)}根 / {hours:.1f}小时)')
    print(f'价格 {px.iloc[0]:.0f} -> {px.iloc[-1]:.0f} ({(px.iloc[-1]/px.iloc[0]-1)*100:+.2f}%)  '
          f'秒级波动 {px.pct_change().std()*1e4:.2f}bp')
    print('=' * 96)
    print('⚠ 无盘口数据, 三项无法建模: 真实spread / 排队成交概率 / 逆向选择。')
    print('  结果系统性偏乐观, 只能用于排除不可行参数, 不能当收益预期。')
    print()
    print(f'{"价差(bp)":>9s}{"手续费":>7s}{"成交数":>8s}{"盈亏(U)":>10s}{"折年化":>10s}'
          f'{"回撤":>7s}{"手续费合计":>10s}{"末库存":>8s}')
    print('-' * 96)
    for sp in (2, 5, 10, 20, 50):
        for fee in (1.0,):
            r = simulate(d, spread_bp=sp, fee_bp=fee)
            m = report(d, r)
            print(f'{sp:9.0f}{fee:7.1f}{m["n"]:8d}{m["pnl"]:+10.2f}{m["ann"]:+9.0f}%'
                  f'{m["dd"]:6.2f}%{m["fees"]:10.2f}{m["inv"]:+8.3f}')


if __name__ == '__main__':
    main()
