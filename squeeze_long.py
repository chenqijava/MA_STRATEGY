"""均线收敛后初次发散 + MACD金叉 做多腿。

四个要件, 每个可独立开关, 便于归因(测出哪些真起作用, 哪些只是复杂度):

【1. 收敛后初次发散】不等"完美排列", 抓启动点而非加速末端
   收敛: 过去 SQZ_LOOK 根内, 四线极差/价格 曾低于 SQZ_TH (粘合/缠绕)
   发散: 当根 四线多头排列 (MA5>MA20>MA50>MA100) 且 排列刚成立(前一根不成立)
   不追高: (MA5/MA100 - 1) <= EXT_MAX (默认20%)
【2. 波动率确认】避开慢熊反弹里"无力的多头排列"
   ADX > ADX_MIN 且 ADX 上升;  或  布林带宽处于近期高位 (BBW 分位 > BBW_Q)
   放量: 当根成交量 > 过去 VOL_LOOK 根均量 × VOL_MULT
【3. MACD金叉】DIF 上穿 DEA (标准 12/26/9)
【4. 出场完全独立】不用均线空头排列/死叉平仓
   a. 移动止损: 初始 入场价 − TRAIL_ATR×ATR, 随价格新高上移, 永不回撤
   b. 单线跟踪: 收盘 < MA_TRAIL(默认MA20) 即走
   c. 分批: 浮盈达 SCALE_ATR×ATR 平一半, 余仓改用 TRAIL_ATR2 更宽止损博大趋势

设计取舍: 这些条件都是"与"关系, 单独看每个都会大幅减少信号。4h 上 49 币的信号本就
稀疏(做空腿中位0/根), 全开可能几乎不触发 —— 故回测时逐个加, 观察信号数与收益的权衡。
"""
import numpy as np
import pandas as pd

DEFAULTS = dict(
    MA1=5, MA2=20, MA3=50, MA4=100,
    # 1. 收敛后发散
    USE_SQUEEZE=True,
    SQZ_LOOK=20,          # 回看多少根内曾出现收敛
    SQZ_TH=0.03,          # 四线极差/价格 < 此值算收敛(3%)
    REQUIRE_FRESH=True,   # 要求多头排列"刚成立"(前一根不成立)
    EXT_MAX=0.20,         # (MA5/MA100-1) 上限, 防追高
    # 2. 波动率确认
    USE_ADX=True,
    ADX_LEN=14, ADX_MIN=20.0, ADX_RISING=True,
    USE_BBW=False,        # 布林带宽分位(与ADX二选一或并用)
    BBW_LEN=20, BBW_Q=0.6,
    USE_VOL=True,
    VOL_LOOK=20, VOL_MULT=1.2,
    # 3. MACD
    USE_MACD=True,
    MACD_FAST=12, MACD_SLOW=26, MACD_SIG=9,
    # 出场参数在 squeeze_sim 里用
    ATR_LEN=14,
)


def _adx(df, n):
    """标准 ADX (Wilder)。返回 adx 序列。"""
    h, l, c = df['high'], df['low'], df['close']
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = pd.Series(tr, index=df.index).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def calc_indicators(df, p):
    df = df.copy()
    for k, n in (('ma1', p['MA1']), ('ma2', p['MA2']), ('ma3', p['MA3']), ('ma4', p['MA4'])):
        df[k] = df['close'].rolling(n).mean()
    # 四线极差 / 价格 —— 收敛度量
    mas = df[['ma1', 'ma2', 'ma3', 'ma4']]
    df['ma_spread'] = (mas.max(axis=1) - mas.min(axis=1)) / df['close']
    # 多头排列
    df['stacked'] = (df['ma1'] > df['ma2']) & (df['ma2'] > df['ma3']) & (df['ma3'] > df['ma4'])
    # 伸展度: 防追高
    df['ext'] = df['ma1'] / df['ma4'] - 1.0
    # MACD
    ef = df['close'].ewm(span=p['MACD_FAST'], adjust=False).mean()
    es = df['close'].ewm(span=p['MACD_SLOW'], adjust=False).mean()
    df['dif'] = ef - es
    df['dea'] = df['dif'].ewm(span=p['MACD_SIG'], adjust=False).mean()
    df['macd_cross'] = (df['dif'] > df['dea']) & (df['dif'].shift(1) <= df['dea'].shift(1))
    # ADX
    df['adx'] = _adx(df, p['ADX_LEN'])
    # 布林带宽 + 其分位
    m = df['close'].rolling(p['BBW_LEN']).mean()
    sd = df['close'].rolling(p['BBW_LEN']).std()
    df['bbw'] = (4 * sd) / m
    df['bbw_q'] = df['bbw'].rolling(100).rank(pct=True)
    # 量
    # 成交量列名: combo_core.load_data 重命名为 volume, 原始 csv 是 vol —— 两种都兼容
    vcol = 'volume' if 'volume' in df.columns else ('vol' if 'vol' in df.columns else None)
    if vcol:
        df['vol'] = df[vcol]
    else:
        df['vol'] = np.nan          # 无量数据时 USE_VOL 会因 NaN 比较而不通过
    df['vol_ma'] = df['vol'].rolling(p['VOL_LOOK']).mean()
    # ATR (出场用)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum((df['high'] - df['close'].shift()).abs(),
                               (df['low'] - df['close'].shift()).abs()))
    df['atr'] = tr.rolling(p['ATR_LEN']).mean()
    df['ma_trail'] = df['ma2']        # 单线跟踪默认用 MA20
    return df


def gen_signals(df, p):
    """entry_sig: 1 = 满足全部启用的条件; 0 = 无。各条件按开关叠加, 便于归因。"""
    df = df.copy()
    n = len(df)
    ok = np.ones(n, dtype=bool)

    # --- 基础: 四线多头排列 ---
    st = df['stacked'].values
    ok &= st
    # 刚成立 (抓启动点, 而非已经排好很久的加速末端)
    if p.get('REQUIRE_FRESH', True):
        prev = np.concatenate([[False], st[:-1]])
        ok &= ~prev

    # --- 1. 收敛后初次发散 ---
    if p.get('USE_SQUEEZE', True):
        sp = df['ma_spread'].values
        look = p['SQZ_LOOK']
        # 过去 look 根内曾出现粘合。用 rolling min 判断"曾经收敛过"
        was = pd.Series(sp).rolling(look, min_periods=2).min().values <= p['SQZ_TH']
        ok &= np.nan_to_num(was, nan=0).astype(bool)
        # 不追高
        ok &= df['ext'].values <= p['EXT_MAX']

    # --- 2. 波动率确认 ---
    if p.get('USE_ADX', True):
        adx = df['adx'].values
        ok &= np.nan_to_num(adx, nan=0) > p['ADX_MIN']
        if p.get('ADX_RISING', True):
            rise = np.concatenate([[False], adx[1:] > adx[:-1]])
            ok &= rise
    if p.get('USE_BBW', False):
        ok &= np.nan_to_num(df['bbw_q'].values, nan=0) > p['BBW_Q']
    if p.get('USE_VOL', True):
        v, vm = df['vol'].values, df['vol_ma'].values
        ok &= np.nan_to_num(v, nan=0) > np.nan_to_num(vm, nan=1e18) * p['VOL_MULT']
        ok &= df['close'].values > df['open'].values        # 放量**阳**线

    # --- 3. MACD 金叉 (允许最近若干根内发生过, 否则与"排列刚成立"同根的概率极低) ---
    if p.get('USE_MACD', True):
        mc = df['macd_cross'].values
        w = int(p.get('MACD_WINDOW', 3))
        recent = pd.Series(mc.astype(float)).rolling(w, min_periods=1).max().values > 0
        ok &= recent

    df['entry_sig'] = np.where(ok, 1, 0).astype(int)
    return df
