"""对冲腿实盘成本建模: 把"滚动beta对冲"的理论收益折成可信数字。

理论对冲收益 = r_strat − beta·r_btc, 但实盘要付三笔账:
  1. 资金费  对冲腿做多 BTC, 资金费率为正时多头付钱。用真实历史费率(每8h结算),
             不是假设值 —— 实测 2023-24 正费率占91%, 折年化 -9.9%。
  2. 调仓手续费  beta 变动就要调对冲腿名义, 每次按 |Δ名义| 收 taker 费。
             beta 抖得越厉害, 这笔越贵 —— 所以"对冲精度"和"调仓成本"是对立的。
  3. 保证金占用  对冲腿要占保证金, 挤占做空腿的可用敞口。建模为: 总敞口不变的前提下,
             做空腿被按比例缩小 (更贴近实盘: 交易所看总保证金, 不会白送额度)。

还建模了实盘必然存在的执行摩擦:
  - beta 用滚动窗口估计且 shift(1) (只用过去数据)
  - 加死区 REBAL_TH: |Δbeta| 小于阈值不调仓, 省手续费

用法:
  UNIVERSE=universe_live.txt TOTAL=1.5 MAX_BARS=40 MA_SLOW=30 SIZING=risk python test_hedge_cost.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import importlib
import analyze_long as al
import portfolio_sim as ps
import test_alpha_beta as tb

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
HEDGE_WIN = int(os.environ.get('HEDGE_WIN', '60'))     # 滚动beta窗口(日)
REBAL_TH = float(os.environ.get('REBAL_TH', '0.05'))   # beta死区: 变动小于此值不调仓
TAKER = ps.FEE                                          # 对冲腿调仓手续费(taker)


def funding_daily(span):
    """真实 BTC 资金费率 -> 日累计 (每8h一次, 一天3次)。正值=多头付。"""
    f = os.path.join(DATA, f'funding_BTC_{span}.csv')
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, parse_dates=['timestamp']).set_index('timestamp')['rate']
    return d.resample('1D').sum().fillna(0.0)


def hedged_returns(rs, rb, fund, win=HEDGE_WIN, th=REBAL_TH,
                   pay_funding=True, pay_rebal=True, share_margin=True):
    """返回对冲后日收益序列。逐日推进, 显式记账三笔成本。

    对冲腿名义(占权益) = -beta_est  (beta<0 => 正数 => 做多BTC)
    做空腿若共享保证金: 按 1/(1+对冲腿名义/总敞口) 缩放, 使总保证金占用不变。
    """
    df = pd.concat([rs.rename('s'), rb.rename('b')], axis=1).dropna()
    beta_raw = tb.rolling_beta(df['s'], df['b'], win)     # 已 shift(1), 无前视
    fd = fund.reindex(df.index).fillna(0.0) if fund is not None else pd.Series(0.0, index=df.index)

    out, h_prev, cost_f, cost_r = [], 0.0, 0.0, 0.0
    for t in df.index:
        b_est = beta_raw.get(t, np.nan)
        h = 0.0 if not np.isfinite(b_est) else -b_est     # 对冲腿名义(占权益), >0 = 做多BTC
        h = float(np.clip(h, 0.0, 2.0))                   # 不做空BTC, 且封顶2x防估计爆掉
        if abs(h - h_prev) < th:                          # 死区: 省调仓费
            h = h_prev
        # 保证金占用: 对冲腿也吃保证金, 总额度不变 => **两腿等比缩放**。
        # 只缩做空腿是错的 —— 那等于偷偷改变对冲比例, 会在亏损段伪装成"改善"。
        k = 1.0 / (1.0 + h / ps.TOTAL_NOTIONAL) if share_margin else 1.0
        h_eff = h * k                                     # 实际持有的对冲腿名义
        r = (df.at[t, 's'] + h * df.at[t, 'b']) * k       # 组合整体按 k 缩放
        if pay_funding:
            c = h_eff * fd.get(t, 0.0)                    # 多头付正费率
            r -= c
            cost_f += c
        if pay_rebal:
            c = abs(h - h_prev) * k * TAKER
            r -= c
            cost_r += c
        out.append(r)
        h_prev = h
    return (pd.Series(out, index=df.index), cost_f, cost_r,
            (-beta_raw.clip(upper=0)).mean())


def strat_series(span, sizing):
    """跑组合(无牛熊过滤, 因为对冲已处理beta)返回日收益。"""
    os.environ['SPAN'] = span
    os.environ['SIZING'] = sizing
    for k in ('REGIME', 'BTC_MA', 'SHORT_BULL', 'SHORT_BEAR', 'BREADTH_MA', 'BREADTH_TH'):
        os.environ.pop(k, None)
    importlib.reload(al)
    importlib.reload(ps)
    idx, C, H, L, A, S, _ = ps.build_panel()
    curve, _, _ = ps.simulate(36, idx, C, H, L, A, S, bull=None)
    eq = pd.Series(curve, index=idx).resample('1D').last().dropna()
    return eq.pct_change().dropna()


def main():
    sizing = os.environ.get('SIZING', 'risk')
    print('=' * 96)
    print(f'对冲腿实盘成本建模  (仓位={sizing}, beta窗口={HEDGE_WIN}日, 死区={REBAL_TH}, '
          f'taker={TAKER*100:.3f}%)')
    print('=' * 96)

    store = {}
    for span, lab in tb.SPANS:
        rb = tb.btc_daily(span)
        fund = funding_daily(span)
        rs = strat_series(span, sizing)
        store[span] = (rs, rb, fund, lab)
        fa = fund.mean() * 365 * 100 if fund is not None else float('nan')
        print(f'\n{lab}   BTC资金费 折年化 {fa:+.1f}% (正=多头付)')

    # --- 逐项剥离成本 ---
    print(f'\n{"成本项":34s}' + ''.join(f'{l[:9]:>26s}' for _, l in tb.SPANS) + f'{"两段最差":>10s}')
    print(f'{"":34s}' + ''.join(f'{"年化":>9s}{"夏普":>8s}{"回撤":>9s}' for _ in tb.SPANS))
    print('-' * 96)
    VARIANTS = [
        ('① 理论对冲(零成本)',        dict(pay_funding=False, pay_rebal=False, share_margin=False)),
        ('② +资金费',                dict(pay_funding=True,  pay_rebal=False, share_margin=False)),
        ('③ +调仓手续费',            dict(pay_funding=True,  pay_rebal=True,  share_margin=False)),
        ('④ +保证金占用 (全成本)',    dict(pay_funding=True,  pay_rebal=True,  share_margin=True)),
    ]
    detail = {}
    for nm, kw in VARIANTS:
        cells, anns = [], []
        for span, _ in tb.SPANS:
            rs, rb, fund, _ = store[span]
            h, cf, cr, hbar = hedged_returns(rs, rb, fund, **kw)
            days = (h.index[-1] - h.index[0]).days
            cum, ann, sh, md = tb.perf(h.values, days)
            cells.append(f'{ann:+8.1f}%{sh:8.2f}{md:8.1f}%')
            anns.append(ann)
            detail[(nm, span)] = (cf * 100, cr * 100, hbar, days)
        print(f'{nm:34s}' + ''.join(cells) + f'{min(anns):+9.1f}%')

    print('\n成本明细 (全成本档):')
    for span, lab in tb.SPANS:
        cf, cr, hbar, days = detail[('④ +保证金占用 (全成本)', span)]
        rs, rb, fund, _ = store[span]
        h_nc, _, _, _ = hedged_returns(rs, rb, fund, pay_funding=False, pay_rebal=False,
                                       share_margin=False)
        h_fc, _, _, _ = hedged_returns(rs, rb, fund, pay_funding=True, pay_rebal=True,
                                       share_margin=True)
        _, a0, _, _ = tb.perf(h_nc.values, days)
        _, a1, _, _ = tb.perf(h_fc.values, days)
        print(f'  {lab[:24]:26s} 对冲腿均名义 {hbar:.2f}x   资金费累计 {cf:.1f}%   '
              f'调仓费累计 {cr:.2f}%   成本合计吃掉年化 {a0-a1:.1f}pp')

    # --- 参数敏感性: 窗口 × 死区 (全成本) ---
    print(f'\n全成本下的稳健性 (年化, 两段最差):')
    hdr = '死区/窗口'
    print(f'{hdr:>10s}' + ''.join(f'{w:>10d}' for w in (30, 45, 60, 90, 120)))
    for th in (0.0, 0.05, 0.10, 0.20):
        row = []
        for w in (30, 45, 60, 90, 120):
            anns = []
            for span, _ in tb.SPANS:
                rs, rb, fund, _ = store[span]
                h, _, _, _ = hedged_returns(rs, rb, fund, win=w, th=th,
                                            pay_funding=True, pay_rebal=True, share_margin=True)
                days = (h.index[-1] - h.index[0]).days
                anns.append(tb.perf(h.values, days)[1])
            row.append(f'{min(anns):+9.1f}%')
        print(f'{th:10.2f}' + ''.join(row))


if __name__ == '__main__':
    main()
