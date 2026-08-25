"""MA5/MA24 金叉/死叉回踩策略 (多空对称)。

做多(金叉):
  1. MA5 上穿 MA24 (金叉)
  2. 金叉后第 3 根起, MA5 仍在 MA24 上方期间, 回踩触发:
       0 <= (最低价 - MA24) <= TOUCH_TH  且  收盘 > 开盘(阳线)  → 买入
  3. 收盘 < MA24  → 平多

做空(死叉, 与做多镜像):
  1. MA5 下穿 MA24 (死叉)
  2. 死叉后第 3 根起, MA5 仍在 MA24 下方期间, 反弹触发:
       0 <= (MA24 - 最高价) <= TOUCH_TH  且  收盘 < 开盘(阴线)  → 卖出开空
  3. 收盘 > MA24  → 平空

ENABLE_LONG/ENABLE_SHORT 可单独开关, 便于对比做多腿/做空腿/双向。
成本/绩效沿用 combo_core (taker 0.05% + 滑点 0.02%, 逐笔单边扣费)。
"""
import os, sys, io
import numpy as np
import pandas as pd
import combo_core as cc
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

INITIAL_CAPITAL = cc.INITIAL_CAPITAL
CONTRACT_VALUE = cc.CONTRACT_VALUE
TAKER_FEE = cc.TAKER_FEE
SLIPPAGE = cc.SLIPPAGE

DEFAULTS = dict(
    MA_FAST=5, MA_SLOW=24,
    MA_TYPE='sma',          # 'sma' 简单均线(默认) | 'ema' 指数均线(反应更快)
    TOUCH_MODE='atr',       # 'abs' 绝对价距(旧) | 'pct' 相对MA24百分比 | 'atr' ATR倍数(波动率归一, 默认; 4h做空t=2.99/样本外3.61)
    TOUCH_TH=10.0,          # abs模式: 0 <= (最低价 - MA24) <= 此值 (绝对价, 跨币种不可比)
    TOUCH_PCT=0.001,        # pct模式: 0 <= (最低价 - MA24)/MA24 <= 此值 (0.1%, 跨币种一致; 4h样本外t=2.28)
    TOUCH_ATR=0.5,          # atr模式: 0 <= (最低价 - MA24) <= 此值·ATR (波动率归一, 跨币种&跨时间可比)
    CROSS_DELAY=6,          # 金叉后间隔 N 根起才允许触发(且期间MA5须持续>MA24)
    # 出场模式:
    #   'ma_slow'  收盘破MA24平仓(默认)
    #   'ma_fast'  收盘破MA5平仓(更快)
    #   'atr'      ATR目标止盈 + ATR止损 (让利润跑、不早砍)
    #   'atr_trail' 晚触发移动止损 (浮盈到TRAIL_TRIG·ATR才启动, 之后TRAIL_W·ATR跟踪) + ATR止损
    EXIT_MODE='ma_slow',
    ATR_LEN=14, ATR_SL=2.0, ATR_TP=4.0,   # atr模式: 止损/止盈 倍数
    MAX_BARS=0,             # 时间止损: 持仓超过N根按收盘平; 0=不启用 (做空扫描最优≈40根/6.7天)
    TRAIL_TRIG=3.0, TRAIL_W=3.0,          # atr_trail模式: 晚触发/跟踪宽度(ATR倍)
    SLOPE_FILTER=True,      # 做多要MA24向上/做空要MA24向下, 过滤走弱的假回踩
    SLOPE_LOOKBACK=5,       # MA24斜率窗口
    ENABLE_LONG=True,       # 做多腿(金叉回踩)
    ENABLE_SHORT=True,      # 做空腿(死叉反弹)
    # ---- 做空腿的K线形态加强 (可选, 0/False=关闭, 只作用于做空) ----
    UPPER_SHADOW=0,         # >0: 要求 上影线 > 实体×此倍数 (长上影=冲高受阻); 2=影线是实体2倍
    BREAK_PREV_LOW=False,   # True: 要求 收盘 < 上一根最低价 (跌破前低, 比阴线更强的确认)
    RISK_PER_TRADE=0.01,
    MAX_NOTIONAL_PCT=0.2,
)


def calc_indicators(df, p):
    df = df.copy()
    if p.get('MA_TYPE', 'sma') == 'ema':
        df['ma_fast'] = df['close'].ewm(span=p['MA_FAST'], adjust=False).mean()
        df['ma_slow'] = df['close'].ewm(span=p['MA_SLOW'], adjust=False).mean()
    else:
        df['ma_fast'] = df['close'].rolling(p['MA_FAST']).mean()
        df['ma_slow'] = df['close'].rolling(p['MA_SLOW']).mean()
    # 金叉：上一根 fast<=slow, 当根 fast>slow；死叉反之
    df['golden_cross'] = (df['ma_fast'] > df['ma_slow']) & (df['ma_fast'].shift(1) <= df['ma_slow'].shift(1))
    df['death_cross'] = (df['ma_fast'] < df['ma_slow']) & (df['ma_fast'].shift(1) >= df['ma_slow'].shift(1))
    # MA24 斜率
    df['slow_rising'] = df['ma_slow'] > df['ma_slow'].shift(p['SLOPE_LOOKBACK'])
    df['slow_falling'] = df['ma_slow'] < df['ma_slow'].shift(p['SLOPE_LOOKBACK'])
    # ATR (供 atr / atr_trail 出场)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum((df['high'] - df['close'].shift()).abs(),
                               (df['low'] - df['close'].shift()).abs()))
    df['atr'] = tr.rolling(p.get('ATR_LEN', 14)).mean()
    return df


def gen_signals(df, p):
    """entry_sig: +1 金叉回踩做多 / -1 死叉反弹做空 / 0 无。多空对称。"""
    df = df.copy()
    n = len(df)
    ma_fast = df['ma_fast'].values
    ma_slow = df['ma_slow'].values
    gcross = df['golden_cross'].values
    dcross = df['death_cross'].values
    low = df['low'].values
    high = df['high'].values
    close = df['close'].values
    open_ = df['open'].values
    slow_rising = df['slow_rising'].values
    slow_falling = df['slow_falling'].values
    atr = df['atr'].values

    entry_sig = np.zeros(n, dtype=int)
    bars_since_g = -1     # 金叉窗口计数, -1=无效
    bars_since_d = -1     # 死叉窗口计数
    for i in range(n):
        if not (np.isfinite(ma_fast[i]) and np.isfinite(ma_slow[i])):
            continue
        # 窗口维护：fast 回到 slow 另一侧则对应窗口失效
        if ma_fast[i] <= ma_slow[i]:
            bars_since_g = -1
        if ma_fast[i] >= ma_slow[i]:
            bars_since_d = -1
        if gcross[i]:
            bars_since_g = 0
        elif bars_since_g >= 0:
            bars_since_g += 1
        if dcross[i]:
            bars_since_d = 0
        elif bars_since_d >= 0:
            bars_since_d += 1

        # 做多：金叉后回踩MA24支撑(贴近但不跌破) + 阳线 + (MA24向上)
        if p['ENABLE_LONG'] and bars_since_g >= p['CROSS_DELAY'] and ma_fast[i] > ma_slow[i]:
            gap = low[i] - ma_slow[i]
            mode = p.get('TOUCH_MODE', 'pct')
            if mode == 'pct':
                th = p['TOUCH_PCT'] * ma_slow[i]
            elif mode == 'atr':
                th = p['TOUCH_ATR'] * atr[i] if np.isfinite(atr[i]) else -1
            else:
                th = p['TOUCH_TH']
            touch = (gap >= 0) and (gap <= th)
            bull = close[i] > open_[i]
            slope_ok = (not p['SLOPE_FILTER']) or bool(slow_rising[i])
            if touch and bull and slope_ok:
                entry_sig[i] = 1
        # 做空：死叉后反弹到MA24阻力(贴近但不上破) + 阴线 + (MA24向下)
        if p['ENABLE_SHORT'] and entry_sig[i] == 0 \
           and bars_since_d >= p['CROSS_DELAY'] and ma_fast[i] < ma_slow[i]:
            gap = ma_slow[i] - high[i]
            mode = p.get('TOUCH_MODE', 'pct')
            if mode == 'pct':
                th = p['TOUCH_PCT'] * ma_slow[i]
            elif mode == 'atr':
                th = p['TOUCH_ATR'] * atr[i] if np.isfinite(atr[i]) else -1
            else:
                th = p['TOUCH_TH']
            touch = (gap >= 0) and (gap <= th)
            bear = close[i] < open_[i]
            slope_ok = (not p['SLOPE_FILTER']) or bool(slow_falling[i])
            # ---- K线形态加强 (可选, 默认关) ----
            # 现有 bear 只要求"收盘<开盘"。两个可选的加强:
            #   UPPER_SHADOW: 上影线 > 实体 × 倍数。语义是"冲高被打回来" —— 触碰 MA90
            #   遇阻的力度写在影线里, 实体小而影线长说明卖压在高位就接手了。
            #   注意"最高价接近MA90"这个条件已由 touch 实现(gap = MA90 - high 在 0~0.5ATR),
            #   无需重复; 这里只补影线/实体的比例。
            #   BREAK_PREV_LOW: 收盘 < 上一根最低价。要求反弹当根就已跌回前一根的区间下方,
            #   是比"阴线"更强的下行确认。
            shape_ok = True
            us_mult = p.get('UPPER_SHADOW', 0)
            if us_mult and shape_ok:
                body = abs(close[i] - open_[i])
                upper = high[i] - max(close[i], open_[i])
                # 实体为 0 时(十字星)算通过: 影线相对实体无穷长, 符合"冲高受阻"的语义
                shape_ok = upper > us_mult * body if body > 0 else upper > 0
            if p.get('BREAK_PREV_LOW') and shape_ok:
                shape_ok = i > 0 and close[i] < low[i - 1]
            if touch and bear and slope_ok and shape_ok:
                entry_sig[i] = -1
    df['entry_sig'] = entry_sig
    return df


def run_backtest(df, p):
    trades = []
    position = 0
    entry_price = 0.0
    entry_time = None
    contract = 0
    equity = INITIAL_CAPITAL
    equity_curve = []

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    ma_fast = df['ma_fast'].values
    ma_slow = df['ma_slow'].values
    atrs = df['atr'].values
    entry_sig = df['entry_sig'].values
    idx = df.index
    mode = p['EXIT_MODE']
    ma_exit = mode in ('ma_slow', 'ma_fast')
    exit_ma = ma_slow if mode == 'ma_slow' else ma_fast
    exit_tag = 'MA24' if mode == 'ma_slow' else 'MA5'

    # ATR出场用的持仓状态
    entry_atr = stop = target = 0.0
    trail_active = False
    ext = 0.0                      # 持仓期间的最有利极值(多头看最高, 空头看最低)
    bars_held = 0                  # 持仓已过K线数, 供时间止损 MAX_BARS
    max_bars = p.get('MAX_BARS', 0)  # 0=不启用时间止损

    def close_pos(exit_price, ts, reason):
        nonlocal equity, position, trail_active
        pnl = position * (exit_price - entry_price) * CONTRACT_VALUE * contract
        fee = abs(exit_price) * CONTRACT_VALUE * contract * TAKER_FEE
        equity += pnl - fee
        trades.append(dict(entry_time=entry_time, exit_time=ts,
                           type='LONG' if position == 1 else 'SHORT',
                           leg='MA', entry_price=entry_price, exit_price=exit_price,
                           pnl=pnl - fee, exit_reason=reason))
        position = 0
        trail_active = False

    for i in range(len(df)):
        ts = idx[i]
        close = closes[i]
        high = highs[i]
        low = lows[i]
        # ---------- 出场 ----------
        if position != 0:
            bars_held += 1
            # 时间止损: 持仓超过 MAX_BARS 根仍未止盈/止损, 按收盘平 (砍长尾, 释放资金/省funding)
            if max_bars and bars_held >= max_bars:
                close_pos(close, ts, '时间止损')
            elif ma_exit:
                if position == 1 and np.isfinite(exit_ma[i]) and close < exit_ma[i]:
                    close_pos(close, ts, f'收盘破{exit_tag}')
                elif position == -1 and np.isfinite(exit_ma[i]) and close > exit_ma[i]:
                    close_pos(close, ts, f'收盘升破{exit_tag}')
            elif mode == 'atr':
                if position == 1:
                    if low <= stop:
                        close_pos(stop, ts, 'ATR止损')
                    elif high >= target:
                        close_pos(target, ts, 'ATR止盈')
                else:
                    if high >= stop:
                        close_pos(stop, ts, 'ATR止损')
                    elif low <= target:
                        close_pos(target, ts, 'ATR止盈')
            elif mode == 'atr_trail':
                if position == 1:
                    ext = max(ext, high)
                    if not trail_active and (high - entry_price) >= p['TRAIL_TRIG'] * entry_atr:
                        trail_active = True
                    dyn = max(stop, ext - p['TRAIL_W'] * entry_atr) if trail_active else stop
                    if low <= dyn:
                        close_pos(dyn, ts, '移动止损' if trail_active else 'ATR止损')
                else:
                    ext = min(ext, low)
                    if not trail_active and (entry_price - low) >= p['TRAIL_TRIG'] * entry_atr:
                        trail_active = True
                    dyn = min(stop, ext + p['TRAIL_W'] * entry_atr) if trail_active else stop
                    if high >= dyn:
                        close_pos(dyn, ts, '移动止损' if trail_active else 'ATR止损')
        # ---------- 开仓 ----------
        if position == 0 and entry_sig[i] != 0 and np.isfinite(atrs[i]) and atrs[i] > 0:
            sig = entry_sig[i]
            entry_price = close * (1 + SLIPPAGE) if sig == 1 else close * (1 - SLIPPAGE)
            max_notional = equity * p['MAX_NOTIONAL_PCT']
            contract = max(1, int(max_notional / (close * CONTRACT_VALUE)))
            entry_time = ts
            position = sig
            entry_atr = atrs[i]
            trail_active = False
            ext = entry_price
            bars_held = 0
            if sig == 1:
                stop = entry_price - p['ATR_SL'] * entry_atr
                target = entry_price + p['ATR_TP'] * entry_atr
            else:
                stop = entry_price + p['ATR_SL'] * entry_atr
                target = entry_price - p['ATR_TP'] * entry_atr
            fee = entry_price * CONTRACT_VALUE * contract * TAKER_FEE
            equity -= fee
        # 盯市权益
        mtm = equity + (position * (close - entry_price) * CONTRACT_VALUE * contract if position != 0 else 0)
        equity_curve.append(mtm)

    if position != 0:
        close_pos(closes[-1], idx[-1], '数据结束平仓')
        equity_curve[-1] = equity
    return trades, equity, equity_curve


def evaluate(df, p, label=''):
    di = calc_indicators(df, p)
    ds = gen_signals(di, p)
    warm = p['MA_SLOW'] + 2
    ds = ds.iloc[warm:]
    trades, equity, curve = run_backtest(ds, p)
    days = (ds.index[-1] - ds.index[0]).days or 1
    m = cc.metrics(trades, curve, ds.index, days)
    m['label'] = label
    return m, trades, curve
