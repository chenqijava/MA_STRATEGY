"""Alpha/Beta 拆解: 两段分别回归, 判断"牛市亏损"是 beta 失血还是 alpha 本身为负。

这是个分水叉诊断:
  若 2023-24 亏损主要来自 beta 而 alpha 仍正 -> 策略可作市场中性组合救活,
    固定一条 BTC 多头腿对冲即可, 且**不需要预测牛熊**(不猜, 只对冲)。
  若 alpha 在 2023-24 本身为负 -> 信号在牛市真无优势, 对冲只是搬运亏损。

口径: 日收益回归 r_strat = alpha + beta·r_btc + eps (OLS)。
  alpha 年化 = alpha_daily × 365; beta 贡献 = beta × BTC同期累计。
  对冲组合 = r_strat − beta·r_btc (事后 beta, 上界估计) 及
             r_strat − beta_roll·r_btc (滚动 beta, 可实盘复现)。

用法:
  UNIVERSE=universe_live.txt TOTAL=1.5 MAX_BARS=40 MA_SLOW=30 python test_alpha_beta.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import importlib
import analyze_long as al
import ma_pullback as mp
import portfolio_sim as ps

SPANS = [('20230101_20250101', '2023-01~2025-01 (BTC +466%)'),
         ('20250101_20260801', '2025-01~2026-08 (BTC -33%)')]
N = int(os.environ.get('N', '36'))

CASES = [
    ('无过滤(原始信号)',   dict(REGIME='btc'), 0),
    ('BTC MA35 牛市停空',  dict(REGIME='btc', BTC_MA=35, BTC_SLOPE=10,
                              SHORT_BULL=0, SHORT_BEAR=1), 1),
    ('宽度MA200>40% 停空', dict(REGIME='breadth', BREADTH_MA=200, BREADTH_TH=0.4,
                              SHORT_BULL=0, SHORT_BEAR=1), 1),
]
_RESET = ('REGIME', 'BTC_MA', 'BTC_SLOPE', 'SHORT_BULL', 'SHORT_BEAR',
          'BREADTH_MA', 'BREADTH_TH')


def ols(y, x):
    """返回 (alpha_daily, beta, r2)。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return a, b, 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def perf(r, days=None):
    """日收益序列 -> (累计%, 年化%, 夏普, 最大回撤%)。"""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return (np.nan,) * 4
    eq = np.cumprod(1 + r)
    cum = eq[-1] - 1
    d = days or len(r)
    ann = (1 + cum) ** (365 / d) - 1 if cum > -1 else -1
    sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
    mdd = np.max(1 - eq / np.maximum.accumulate(eq))
    return cum * 100, ann * 100, sh, mdd * 100


def btc_daily(span):
    os.environ['SPAN'] = span
    importlib.reload(al)
    c = mp.cc.load_data(al.path('BTC'))['close'].resample('1D').last().dropna()
    return c.pct_change().dropna()


def strat_daily(span, env, use_regime):
    """跑组合并返回日收益序列 (盯市权益重采样到日)。"""
    os.environ['SPAN'] = span
    for k in _RESET:
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = str(v)
    importlib.reload(al)
    importlib.reload(ps)
    idx, C, H, L, A, S, syms = ps.build_panel()
    bull = ps.regime_mask(idx) if use_regime else None
    curve, trades, _ = ps.simulate(N, idx, C, H, L, A, S, bull=bull)
    eq = pd.Series(curve, index=idx).resample('1D').last().dropna()
    return eq.pct_change().dropna()


def rolling_beta(y, x, win=60):
    """滚动 beta (只用过去数据, shift(1) 后可实盘复现)。"""
    df = pd.concat([y.rename('y'), x.rename('x')], axis=1).dropna()
    cov = df['y'].rolling(win).cov(df['x'])
    var = df['x'].rolling(win).var()
    return (cov / var).shift(1)


def main():
    for span, lab in SPANS:
        rb = btc_daily(span)
        print('=' * 100)
        print(f'{lab}')
        print('=' * 100)
        btc_cum = (np.prod(1 + rb.values) - 1) * 100
        print(f'BTC 同期: 累计 {btc_cum:+.1f}%   日波动 {rb.std()*100:.2f}%   天数 {len(rb)}\n')

        print(f'{"配置":20s}{"策略年化":>9s}{"Alpha年化":>10s}{"Beta":>7s}{"R²":>6s}'
              f'{"Beta贡献":>9s}{"Alpha贡献":>10s}')
        print('-' * 72)
        rows = []
        for name, env, reg in CASES:
            rs = strat_daily(span, env, reg)
            df = pd.concat([rs.rename('s'), rb.rename('b')], axis=1).dropna()
            a, b, r2 = ols(df['s'], df['b'])
            days = (df.index[-1] - df.index[0]).days or len(df)
            cum, ann, sh, mdd = perf(df['s'].values, days)
            # 分解: beta 部分 = beta·r_btc 的累计; alpha 部分 = 残差+alpha 的累计
            beta_leg = np.prod(1 + b * df['b'].values) - 1
            alpha_leg = np.prod(1 + (df['s'].values - b * df['b'].values)) - 1
            print(f'{name:20s}{ann:+8.1f}%{a*365*100:+9.1f}%{b:7.2f}{r2:6.2f}'
                  f'{beta_leg*100:+8.1f}%{alpha_leg*100:+9.1f}%')
            rows.append((name, df, a, b, days))

        # 对冲效果: 用无过滤(原始信号)那一档, 看去掉 beta 后剩什么
        name, df, a, b, days = rows[0]
        print(f'\n[对冲验证 — 基于"{name}"]')
        print(f'{"组合":26s}{"累计":>9s}{"年化":>9s}{"夏普":>7s}{"回撤":>8s}')
        print('-' * 60)
        variants = [('原始(未对冲)', df['s'].values)]
        variants.append((f'事后beta对冲 (β={b:.2f})', (df['s'] - b * df['b']).values))
        rbeta = rolling_beta(df['s'], df['b'], 60)
        hedged = (df['s'] - rbeta * df['b']).dropna()
        variants.append(('滚动beta对冲 (60日,可实盘)', hedged.values))
        for nm, r in variants:
            c_, an, sh, md = perf(r, days)
            print(f'{nm:26s}{c_:+8.1f}%{an:+8.1f}%{sh:7.2f}{md:7.1f}%')
        print()


if __name__ == '__main__':
    main()
