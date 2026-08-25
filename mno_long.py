"""MNO 双路径做多 (MOU突破路径 + KAKU回调路径)。

按给定描述原样实现, 每条件可单独开关以便归因。所有阈值走参数, 便于按 4h 尺度调整
(原描述针对日线, 文中已注明 4h 需把 MACD 零轴阈值从 0.2 降到 0.1/0.05)。

共同前提: EMA5 > EMA13 > EMA26 (三重黄金排列)

MOU 突破路径 (两种之一):
  A. 阻力突破后回调: 曾突破前期阻力, 随后回调 PB_MIN~PB_MAX (5%~15%), 现企稳
  B. 直接突破: 收盘 > 前期阻力×(1+BRK_TH) 且 K线实体 > 过去20根均实体×(1+BODY_TH)
  两者都要求 量比 ∈ [VOL_MIN, VOL_MAX] (1.3~3.0), 且 MACD 在零轴附近金叉

KAKU 回调路径 (严格版, 8基础 + 3终极):
  基础: 三重排列/价在EMA26上/EMA26上行/回调至EMA13附近/未破EMA26/
        趋势已持续N根/ADX>阈值/未过度伸展
  终极: 针脚K线(下影≥2倍实体 且 收盘≥开盘) + MACD零轴**上方**金叉 + 量比≥1.5

风控: 止盈 TP_PCT(2%) / 止损 SL_PCT(1%) / 最长持仓 MAX_BARS(30根)

【实现前的判断】文中"KAKU胜率>75%"与"2:1盈亏比"不可能同时成立 —— 那意味着每笔期望
+1.25%、利润因子6.0。这里如实实现规则, 由三段数据检验, 不预设结论。
"""
import numpy as np
import pandas as pd

DEFAULTS = dict(
    EMA1=5, EMA2=13, EMA3=26,
    # 共同
    RES_LOOK=20,          # 前期阻力回看根数
    VOL_LOOK=20,
    ADX_LEN=14,
    ATR_LEN=14,
    MACD_FAST=12, MACD_SLOW=26, MACD_SIG=9,
    # MOU
    USE_MOU=True,
    BRK_TH=0.003,         # 突破需超阻力 0.3%
    BODY_TH=0.20,         # 实体需超20根均实体 20%
    VOL_MIN=1.3, VOL_MAX=3.0,
    PB_MIN=0.05, PB_MAX=0.15,   # 回调幅度 5%~15%
    ZERO_TH=0.001,        # MACD"零轴附近": |dif|/价格 < 此值 (4h 用小值, 文中建议降低)
    # KAKU
    USE_KAKU=True,
    KAKU_TREND_BARS=10,   # 三重排列已持续根数
    KAKU_ADX_MIN=25.0,
    KAKU_NEAR_EMA2=0.02,  # 回调到 EMA13 附近 2% 内
    KAKU_EXT_MAX=0.15,    # 未过度伸展 (EMA5/EMA26-1)
    PIN_RATIO=2.0,        # 针脚: 下影 ≥ 实体×此倍数
    KAKU_VOL_MIN=1.5,
    KAKU_ONLY=False,      # 仅KAKU模式
)


def _adx(df, n):
    h, l, c = df['high'], df['low'], df['close']
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = pd.Series(tr, index=df.index).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def calc_indicators(df, p):
    df = df.copy()
    for k, n in (('e1', p['EMA1']), ('e2', p['EMA2']), ('e3', p['EMA3'])):
        df[k] = df['close'].ewm(span=n, adjust=False).mean()
    df['stacked'] = (df['e1'] > df['e2']) & (df['e2'] > df['e3'])
    df['stack_run'] = df['stacked'].groupby((~df['stacked']).cumsum()).cumcount() + 1
    df['stack_run'] = df['stack_run'].where(df['stacked'], 0)
    df['e3_up'] = df['e3'] > df['e3'].shift(1)
    df['ext'] = df['e1'] / df['e3'] - 1.0
    # 前期阻力(不含当根)
    df['res'] = df['high'].rolling(p['RES_LOOK']).max().shift(1)
    # 实体 与 20根均实体
    df['body'] = (df['close'] - df['open']).abs()
    df['body_ma'] = df['body'].rolling(p['VOL_LOOK']).mean()
    # 量比
    vcol = 'volume' if 'volume' in df.columns else ('vol' if 'vol' in df.columns else None)
    df['v'] = df[vcol] if vcol else np.nan
    df['vr'] = df['v'] / df['v'].rolling(p['VOL_LOOK']).mean()
    # MACD
    ef = df['close'].ewm(span=p['MACD_FAST'], adjust=False).mean()
    es = df['close'].ewm(span=p['MACD_SLOW'], adjust=False).mean()
    df['dif'] = ef - es
    df['dea'] = df['dif'].ewm(span=p['MACD_SIG'], adjust=False).mean()
    df['mx'] = (df['dif'] > df['dea']) & (df['dif'].shift(1) <= df['dea'].shift(1))
    df['dif_r'] = df['dif'] / df['close']       # 归一化, 跨币可比
    df['adx'] = _adx(df, p['ADX_LEN'])
    # 针脚K线: 下影 ≥ 实体×PIN_RATIO 且 收盘 ≥ 开盘
    lower = np.minimum(df['open'], df['close']) - df['low']
    body = (df['close'] - df['open']).abs()
    df['pin'] = (lower >= body * p['PIN_RATIO']) & (df['close'] >= df['open'])
    # 回调幅度: 从近期高点回撤多少
    df['pb'] = 1.0 - df['close'] / df['high'].rolling(p['RES_LOOK']).max()
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum((df['high'] - df['close'].shift()).abs(),
                               (df['low'] - df['close'].shift()).abs()))
    df['atr'] = tr.rolling(p['ATR_LEN']).mean()
    df['ma_trail'] = df['e2']
    return df


def gen_signals(df, p):
    """entry_sig: 1=MOU / 2=KAKU (同时触发取KAKU, 按"优先KAKU"); 0=无。"""
    df = df.copy()
    v = lambda k: df[k].values                      # noqa: E731
    nz = lambda a, d=0.0: np.nan_to_num(a, nan=d)   # noqa: E731

    stacked = v('stacked')
    vr = nz(v('vr'))
    mx = v('mx')
    difr = nz(v('dif_r'))

    # ---------- MOU 突破路径 ----------
    mou = np.zeros(len(df), dtype=bool)
    if p.get('USE_MOU', True):
        vol_ok = (vr >= p['VOL_MIN']) & (vr <= p['VOL_MAX'])
        near_zero = np.abs(difr) < p['ZERO_TH']
        macd_ok = mx & near_zero                    # 零轴附近金叉
        # A. 阻力突破后回调企稳: 回调幅度在区间内 + 当根阳线
        pb = nz(v('pb'), 9)
        a = (pb >= p['PB_MIN']) & (pb <= p['PB_MAX']) & (v('close') > v('open'))
        # B. 直接突破: 收盘>阻力×(1+BRK_TH) 且 实体>均实体×(1+BODY_TH)
        res = v('res')
        b = (np.isfinite(res) & (v('close') > res * (1 + p['BRK_TH']))
             & (nz(v('body')) > nz(v('body_ma'), 1e18) * (1 + p['BODY_TH'])))
        mou = stacked & vol_ok & macd_ok & (a | b)

    # ---------- KAKU 回调路径 ----------
    kaku = np.zeros(len(df), dtype=bool)
    if p.get('USE_KAKU', True):
        base = (
            stacked                                                    # 1 三重排列
            & (v('close') > v('e3'))                                   # 2 价在EMA26上方
            & v('e3_up')                                               # 3 EMA26 上行
            & (np.abs(v('close') / v('e2') - 1.0) <= p['KAKU_NEAR_EMA2'])  # 4 回调至EMA13附近
            & (v('low') > v('e3'))                                     # 5 未破 EMA26
            & (nz(v('stack_run')) >= p['KAKU_TREND_BARS'])              # 6 趋势已持续
            & (nz(v('adx')) > p['KAKU_ADX_MIN'])                        # 7 ADX 足够
            & (nz(v('ext'), 9) <= p['KAKU_EXT_MAX'])                   # 8 未过度伸展
        )
        ultimate = v('pin') & (mx & (difr > 0)) & (vr >= p['KAKU_VOL_MIN'])
        kaku = base & ultimate

    if p.get('KAKU_ONLY', False):
        mou = np.zeros(len(df), dtype=bool)
    df['entry_sig'] = np.where(kaku, 2, np.where(mou, 1, 0)).astype(int)
    return df
