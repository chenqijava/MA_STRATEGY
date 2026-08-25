"""MA90 趋势跟随做多腿 — 与 ma_pullback 的做空腿是**不同结构**, 单独一套。

入场 (每根4h收盘判断):
  1. 收盘价 > MA90        —— 处于慢线上方 = 上升结构
  2. 收盘价 > 开盘价      —— 阳线, 当根买盘占优
出场 (取先触发者):
  a. ATR 止盈 +4·ATR / 止损 -2·ATR (盘中触发, 止损优先)
  b. 入场后第1根起, 收盘价 < MA90 立即平 —— 跌回慢线下方即上升结构已破

与被否决的"金叉回踩做多"的区别 (后者 alpha 两段都负、熊市段 t=-4.3 反向显著):
  回踩式是**逆势**: 等价格跌到慢线附近抄支撑, 赌支撑有效。而 4h 加密下跌由恐慌抛售
  驱动, 支撑常被直接击穿 —— 这是它失效的机制。
  本策略是**顺势**: 只在价格已站在慢线上方时买, 结构破了就走, 不预判支撑。
接口与 ma_pullback 对齐 (calc_indicators / gen_signals 同名同签名), 便于 portfolio_sim 复用。
"""
import numpy as np
import pandas as pd

DEFAULTS = dict(
    MA_SLOW=90,             # 慢线(与做空腿同值, 便于对照; 可单独扫)
    MA_TYPE='sma',
    ATR_LEN=14,
    ATR_TP=4.0,
    ATR_SL=2.0,
    MAX_BARS=0,             # 时间止损(根); 0=关。做空腿用40, 这里先不设
    MIN_BODY_PCT=0.0,       # 阳线实体最小占比(|close-open|/close), 0=只要收>开
    MA_BUFFER=0.0,          # 入场需 close > MA×(1+此值); >0 可要求更明确站上慢线
)


def calc_indicators(df, p):
    """与 ma_pullback.calc_indicators 接口一致, 只算本策略需要的列。"""
    df = df.copy()
    if p.get('MA_TYPE', 'sma') == 'ema':
        df['ma_slow'] = df['close'].ewm(span=p['MA_SLOW'], adjust=False).mean()
    else:
        df['ma_slow'] = df['close'].rolling(p['MA_SLOW']).mean()
    # ma_fast 不参与本策略, 但 portfolio_sim 的 COOLDOWN_ONCE 会读它 —— 填 close 代替,
    # 使"穿越即重置"退化为按收盘价与慢线的关系判断窗口, 语义上正好对应本策略的结构破坏。
    df['ma_fast'] = df['close']
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum((df['high'] - df['close'].shift()).abs(),
                               (df['low'] - df['close'].shift()).abs()))
    df['atr'] = tr.rolling(p.get('ATR_LEN', 14)).mean()
    return df


def gen_signals(df, p):
    """entry_sig: +1 做多 / 0 无。收盘>MA90 且 收盘>开盘。"""
    df = df.copy()
    ma = df['ma_slow'].values
    c = df['close'].values
    o = df['open'].values
    buf = p.get('MA_BUFFER', 0.0)
    minb = p.get('MIN_BODY_PCT', 0.0)
    above = np.isfinite(ma) & (c > ma * (1.0 + buf))
    bull = c > o
    if minb > 0:
        bull &= ((c - o) / np.where(c > 0, c, np.nan)) >= minb
    df['entry_sig'] = np.where(above & bull, 1, 0).astype(int)
    return df
