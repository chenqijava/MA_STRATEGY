"""趋势 + 均值回归结合策略 — 回测核心。

设计见 设计方案.md。架构：
  regime 状态判别 → 趋势市走突破腿 / 震荡市走回归腿 → 单仓状态机(分腿出场) → 统一风控。

在 py/backtest_core.py 的账户/成本/风控框架上扩展，新增：
  - 布林带 / RSI / z-score / regime 指标
  - 回归腿信号(bbands 主 / zscore 对照)
  - 状态机区分两腿出场逻辑(趋势腿宽止损让利润跑, 回归腿快进快出硬止损)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------- 账户/成本(与 backtest_core 一致) ----------
INITIAL_CAPITAL = 10000
CONTRACT_VALUE = 0.01
TAKER_FEE = 0.0005
SLIPPAGE = 0.0002

DEFAULTS = dict(
    # 趋势腿(沿用 backtest_core)
    EMA_LEN=50, ATR_LEN=14, ADX_LEN=14,
    CHANNEL_LEN=10, VOL_MULT=2.0, ATR_SL_MULT=2.0, ATR_TP_MULT=5.0,
    TRAIL_TRIGGER1=1.5, TRAIL_ATR_MULT=2.0, MAX_BARS=36,
    VOL_RATIO_MIN=0.005, USE_1H_TREND=True, CLOSE_CONFIRM=True,
    REQUIRE_ADX_MIN=25, ADX_RISING=True,
    TREND_ENTRY='pullback',          # 'pullback' 回踩企稳 | 'breakout' 突破追高(对照)
    PB_TOL=0.004,                    # 回踩到 EMA 的容差(±0.4%算触及)
    PB_LOOKBACK=4,                   # 回踩观察窗口(根)
    PB_SL_BUF=0.5,                   # 回踩腿止损: swing低点下方留 0.5·ATR 缓冲
    # 状态判别(双阈值缓冲带)
    ADX_TREND=25, ADX_RANGE=20,
    # 回归腿
    BB_LEN=20, BB_MULT=2.0, RSI_LEN=14, RSI_OS=30, RSI_OB=70,
    Z_TH=2.0, MR_SL_ATR=1.0, MR_MAX_BARS=12,
    MR_SIGNAL='bbands',              # 'bbands' | 'zscore'
    # 开关：回归腿经样本外验证为负贡献(ETH趋势性过强), 默认关闭。见 复盘.md
    ENABLE_TREND=True, ENABLE_MR=False,
    # 统一风控(沿用)
    MAX_DAILY_LOSS=0.05, MAX_NOTIONAL_PCT=0.2, RISK_PER_TRADE=0.01,
)


def load_data(path):
    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
    if 'vol' in df.columns and 'volume' not in df.columns:
        df = df.rename(columns={'vol': 'volume'})
    return df


def _rsi(close, n):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_indicators(df, p):
    df = df.copy()
    # --- 趋势腿指标 ---
    df['ema'] = df['close'].ewm(span=p['EMA_LEN'], adjust=False).mean()
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift()),
                                     abs(df['low'] - df['close'].shift())))
    df['atr'] = df['tr'].rolling(p['ATR_LEN']).mean()
    df['up'] = df['high'].diff()
    df['down'] = -df['low'].diff()
    df['plus_dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['minus_dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
    atr_adx = df['tr'].rolling(p['ADX_LEN']).mean()
    df['plus_di'] = 100 * (pd.Series(df['plus_dm']).rolling(p['ADX_LEN']).mean() / atr_adx)
    df['minus_di'] = 100 * (pd.Series(df['minus_dm']).rolling(p['ADX_LEN']).mean() / atr_adx)
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(p['ADX_LEN']).mean()
    df['hh_channel'] = df['high'].rolling(p['CHANNEL_LEN']).max().shift(1)
    df['ll_channel'] = df['low'].rolling(p['CHANNEL_LEN']).min().shift(1)
    df['vol_prev'] = df['volume'].shift(1)
    # 1H趋势(shift避免前视)
    h = df['close'].resample('1h').ohlc()
    h['ema50'] = h['close'].ewm(span=50, adjust=False).mean()
    df['close_1h'] = h['close'].shift(1).reindex(df.index, method='ffill')
    df['ema50_1h'] = h['ema50'].shift(1).reindex(df.index, method='ffill')
    # --- 回归腿指标 ---
    mid = df['close'].rolling(p['BB_LEN']).mean()
    sd = df['close'].rolling(p['BB_LEN']).std()
    df['bb_mid'] = mid
    df['bb_up'] = mid + p['BB_MULT'] * sd
    df['bb_low'] = mid - p['BB_MULT'] * sd
    df['zscore'] = (df['close'] - mid) / sd.replace(0, np.nan)
    df['rsi'] = _rsi(df['close'], p['RSI_LEN'])
    # 回踩用：近 PB_LOOKBACK 根内是否触及过 EMA(±容差)
    dist_ema = (df['low'] - df['ema']) / df['ema']       # 多头看低点是否探到EMA
    dist_ema_up = (df['ema'] - df['high']) / df['ema']   # 空头看高点是否反弹到EMA
    touched_lo = (dist_ema.abs() <= p['PB_TOL']) | (df['low'] <= df['ema'])
    touched_hi = (dist_ema_up.abs() <= p['PB_TOL']) | (df['high'] >= df['ema'])
    df['pb_touch_long'] = touched_lo.rolling(p['PB_LOOKBACK']).max().shift(1).fillna(0)
    df['pb_touch_short'] = touched_hi.rolling(p['PB_LOOKBACK']).max().shift(1).fillna(0)
    # 回踩腿止损参考：近窗口 swing 低/高点
    df['swing_low'] = df['low'].rolling(p['PB_LOOKBACK']).min()
    df['swing_high'] = df['high'].rolling(p['PB_LOOKBACK']).max()
    # --- 状态判别 ---
    # TREND=2, NEUTRAL=1, RANGE=0
    regime = np.full(len(df), 1, dtype=int)
    adx_rising = df['adx'] > df['adx'].shift(1)
    is_trend = (df['adx'] > p['ADX_TREND']) & adx_rising
    is_range = (df['adx'] < p['ADX_RANGE'])
    regime = np.where(is_trend, 2, np.where(is_range, 0, 1))
    df['regime'] = regime
    return df


def gen_signals(df, p):
    """生成分腿信号。signal: +1做多/-1做空/0无; leg: 1=趋势腿 2=回归腿 0=无。"""
    df['signal'] = 0
    df['leg'] = 0

    # ---------- 趋势腿(仅趋势市) ----------
    if p['ENABLE_TREND']:
        in_trend = df['regime'] == 2
        trend_up = df['close'] > df['ema']
        trend_down = df['close'] < df['ema']
        vol_ok = df['volume'] > df['vol_prev'] * p['VOL_MULT']
        adx_ok = df['adx'] > p['REQUIRE_ADX_MIN']
        if p['ADX_RISING']:
            adx_ok = adx_ok & (df['adx'] > df['adx'].shift(1))
        volatile = (df['atr'] / df['close']) > p['VOL_RATIO_MIN']
        if p['TREND_ENTRY'] == 'breakout':
            # 对照：通道突破追高
            if p['CLOSE_CONFIRM']:
                trig_up = df['close'] > df['hh_channel']
                trig_dn = df['close'] < df['ll_channel']
            else:
                trig_up = df['high'] > df['hh_channel']
                trig_dn = df['low'] < df['ll_channel']
        else:
            # 主：回踩企稳。先前几根探到过EMA(回踩)，当根收盘重新站回EMA同侧且收阳/阴(企稳)
            bull_bar = df['close'] > df['open'] if 'open' in df else df['close'] > df['close'].shift(1)
            bear_bar = df['close'] < df['open'] if 'open' in df else df['close'] < df['close'].shift(1)
            trig_up = (df['pb_touch_long'] > 0) & (df['close'] > df['ema']) & bull_bar
            trig_dn = (df['pb_touch_short'] > 0) & (df['close'] < df['ema']) & bear_bar
        long_tr = in_trend & trig_up & adx_ok & trend_up & volatile
        short_tr = in_trend & trig_dn & adx_ok & trend_down & volatile
        # 回踩不要求放量(突破才要)；突破模式保留放量过滤
        if p['TREND_ENTRY'] == 'breakout':
            long_tr = long_tr & vol_ok
            short_tr = short_tr & vol_ok
        if p['USE_1H_TREND']:
            long_tr = long_tr & (df['close_1h'] > df['ema50_1h'])
            short_tr = short_tr & (df['close_1h'] < df['ema50_1h'])
        df.loc[long_tr, ['signal', 'leg']] = [1, 1]
        df.loc[short_tr, ['signal', 'leg']] = [-1, 1]

    # ---------- 回归腿(仅震荡市) ----------
    if p['ENABLE_MR']:
        in_range = df['regime'] == 0
        if p['MR_SIGNAL'] == 'bbands':
            long_mr = in_range & (df['low'] <= df['bb_low']) & (df['rsi'] < p['RSI_OS'])
            short_mr = in_range & (df['high'] >= df['bb_up']) & (df['rsi'] > p['RSI_OB'])
        else:  # zscore
            long_mr = in_range & (df['zscore'] < -p['Z_TH'])
            short_mr = in_range & (df['zscore'] > p['Z_TH'])
        # 趋势腿未占用的K线才允许回归腿开(避免同一K线两腿抢)
        free = df['leg'] == 0
        df.loc[long_mr & free, ['signal', 'leg']] = [1, 2]
        df.loc[short_mr & free, ['signal', 'leg']] = [-1, 2]

    return df


def run_backtest(df, p):
    """单仓状态机。趋势腿(leg=1)与回归腿(leg=2)共用一个仓位槽, 出场逻辑分开。"""
    trades = []
    position = 0            # 0/1/-1
    cur_leg = 0             # 1=趋势 2=回归
    entry_price = entry_atr = 0.0
    entry_time = None
    bars_held = 0
    stop_loss = take_profit = bb_mid_target = 0.0
    trail_active = False
    trail_price = 0.0
    contract = 0
    hi_since = lo_since = 0.0
    equity = INITIAL_CAPITAL
    equity_curve = []
    daily_start = equity
    cur_date = None

    ema = df['ema'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['atr'].values
    sigs = df['signal'].values
    legs = df['leg'].values
    regimes = df['regime'].values
    bb_mids = df['bb_mid'].values
    swing_lows = df['swing_low'].values
    swing_highs = df['swing_high'].values
    idx = df.index

    def close_pos(exit_price, ts, reason):
        nonlocal equity, position, cur_leg, bars_held, trail_active
        pnl = position * (exit_price - entry_price) * CONTRACT_VALUE * contract
        fee = abs(exit_price) * CONTRACT_VALUE * contract * TAKER_FEE
        equity += pnl - fee
        trades.append(dict(entry_time=entry_time, exit_time=ts,
                           leg='TREND' if cur_leg == 1 else 'MR',
                           type='LONG' if position == 1 else 'SHORT',
                           entry_price=entry_price, exit_price=exit_price,
                           pnl=pnl - fee, exit_reason=reason))
        position = 0
        cur_leg = 0
        bars_held = 0
        trail_active = False

    for i in range(len(df)):
        ts = idx[i]
        close = closes[i]
        high = highs[i]
        low = lows[i]
        atr_now = atrs[i]
        signal = sigs[i]
        leg = legs[i]

        d = ts.date()
        if cur_date is None or d != cur_date:
            cur_date = d
            daily_start = equity
        allow_new = equity >= daily_start * (1 - p['MAX_DAILY_LOSS'])

        # ============ 持仓管理(分腿出场) ============
        if position != 0:
            bars_held += 1
            if cur_leg == 1:
                # --- 趋势腿：宽止损 + 移动止损 + 目标止盈 + EMA反向 + 时间 ---
                if bars_held > p['MAX_BARS']:
                    close_pos(close, ts, '趋势-时间出场')
                elif (position == 1 and close < ema[i]) or (position == -1 and close > ema[i]):
                    close_pos(close, ts, '趋势-EMA止损')
                elif position == 1:
                    hi_since = max(hi_since, high)
                    profit_atr = (high - entry_price) / entry_atr
                    if not trail_active and profit_atr >= p['TRAIL_TRIGGER1']:
                        trail_active = True
                        trail_price = entry_price
                    if trail_active:
                        trail_price = max(trail_price, hi_since - p['TRAIL_ATR_MULT'] * entry_atr)
                    dynamic_sl = trail_price if trail_active else stop_loss
                    if low <= dynamic_sl:
                        close_pos(dynamic_sl, ts, '趋势-移动止损')
                    elif high >= take_profit:
                        close_pos(take_profit, ts, '趋势-目标止盈')
                elif position == -1:
                    lo_since = min(lo_since, low)
                    profit_atr = (entry_price - low) / entry_atr
                    if not trail_active and profit_atr >= p['TRAIL_TRIGGER1']:
                        trail_active = True
                        trail_price = entry_price
                    if trail_active:
                        trail_price = min(trail_price, lo_since + p['TRAIL_ATR_MULT'] * entry_atr)
                    dynamic_sl = trail_price if trail_active else stop_loss
                    if high >= dynamic_sl:
                        close_pos(dynamic_sl, ts, '趋势-移动止损')
                    elif low <= take_profit:
                        close_pos(take_profit, ts, '趋势-目标止盈')
            else:
                # --- 回归腿：中轨止盈 + 硬止损 + 时间 + 状态止损 ---
                if regimes[i] == 2:
                    close_pos(close, ts, '回归-状态转趋势')          # 前提假设破坏
                elif bars_held > p['MR_MAX_BARS']:
                    close_pos(close, ts, '回归-时间出场')
                elif position == 1:
                    if low <= stop_loss:
                        close_pos(stop_loss, ts, '回归-硬止损')
                    elif high >= bb_mid_target:
                        close_pos(bb_mid_target, ts, '回归-中轨止盈')
                elif position == -1:
                    if high >= stop_loss:
                        close_pos(stop_loss, ts, '回归-硬止损')
                    elif low <= bb_mid_target:
                        close_pos(bb_mid_target, ts, '回归-中轨止盈')

        # ============ 开仓 ============
        if position == 0 and signal != 0 and allow_new \
           and np.isfinite(atr_now) and atr_now > 0:
            entry_atr = atr_now
            entry_time = ts
            bars_held = 0
            trail_active = False
            cur_leg = leg
            entry_price = close * (1 + SLIPPAGE) if signal == 1 else close * (1 - SLIPPAGE)
            # 止损：回踩腿用结构化(swing低点±ATR缓冲)，突破/回归腿用ATR倍数
            if leg == 1 and p['TREND_ENTRY'] == 'pullback':
                if signal == 1:
                    stop_loss = swing_lows[i] - p['PB_SL_BUF'] * entry_atr
                else:
                    stop_loss = swing_highs[i] + p['PB_SL_BUF'] * entry_atr
            else:
                sl_mult = p['ATR_SL_MULT'] if leg == 1 else p['MR_SL_ATR']
                stop_loss = (entry_price - sl_mult * entry_atr) if signal == 1 \
                    else (entry_price + sl_mult * entry_atr)
            # 按止损距离定仓位(风险平价)
            sl_dist = abs(entry_price - stop_loss)
            if not np.isfinite(sl_dist) or sl_dist <= 0:
                sl_dist = p['ATR_SL_MULT'] * entry_atr
            risk_amount = equity * p['RISK_PER_TRADE']
            raw = risk_amount / (sl_dist * CONTRACT_VALUE)
            max_notional = equity * p['MAX_NOTIONAL_PCT']
            max_by_notional = max_notional / (close * CONTRACT_VALUE)
            contract = max(1, int(min(raw, max_by_notional)))
            if signal == 1:
                position = 1
                hi_since = entry_price
            else:
                position = -1
                lo_since = entry_price
            if leg == 1:
                take_profit = entry_price + (p['ATR_TP_MULT'] * entry_atr if signal == 1
                                             else -p['ATR_TP_MULT'] * entry_atr)
            else:
                bb_mid_target = bb_mids[i]        # 回归目标=当前中轨
            fee = entry_price * CONTRACT_VALUE * contract * TAKER_FEE
            equity -= fee

        # 盯市权益(等间隔序列, 供夏普)
        if position != 0:
            mtm = equity + position * (close - entry_price) * CONTRACT_VALUE * contract
        else:
            mtm = equity
        equity_curve.append(mtm)

    if position != 0:
        close_pos(closes[-1], idx[-1], '数据结束平仓')
        equity_curve[-1] = equity

    return trades, equity, equity_curve


def metrics(trades, equity_curve, index, days):
    if not trades or len(equity_curve) < 2:
        return dict(trades=0, ret=0, ann_ret=0, sharpe=0, mdd=0, win=0, pf=0,
                    final=INITIAL_CAPITAL, tr_trades=0, mr_trades=0, tr_pnl=0, mr_pnl=0)
    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity_curve, index=index)
    final = equity_curve[-1]
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = days / 365.0
    ann_ret = (final / INITIAL_CAPITAL) ** (1 / years) - 1 if years > 0 and final > 0 else -1
    daily_eq = eq.resample('1D').last().dropna()
    drets = daily_eq.pct_change().dropna()
    sharpe = (drets.mean() / drets.std()) * np.sqrt(365) if drets.std() > 0 else 0
    mdd = ((eq.cummax() - eq) / eq.cummax()).max()
    wins = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] <= 0]
    win = len(wins) / len(tdf)
    pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) and losses['pnl'].sum() != 0 else np.inf
    tr = tdf[tdf['leg'] == 'TREND']
    mr = tdf[tdf['leg'] == 'MR']
    return dict(trades=len(tdf), ret=total_ret, ann_ret=ann_ret, sharpe=sharpe,
                mdd=mdd, win=win, pf=pf, final=final,
                tr_trades=len(tr), mr_trades=len(mr),
                tr_pnl=tr['pnl'].sum(), mr_pnl=mr['pnl'].sum())


def evaluate(df_ind, p, label=''):
    df_s = gen_signals(df_ind.copy(), p)
    warm = max(p['EMA_LEN'], p['BB_LEN']) + p['CHANNEL_LEN'] + p['ATR_LEN'] + p['ADX_LEN']
    df_s = df_s.iloc[warm:]
    trades, equity, curve = run_backtest(df_s, p)
    days = (df_s.index[-1] - df_s.index[0]).days or 1
    m = metrics(trades, curve, df_s.index, days)
    m['label'] = label
    return m, trades, curve
