"""海龟交易系统 — 做多腿 (唐奇安通道突破)。

原版是日线, 这里按 4h 实现, 通道 20/10 根 (≈3.3天/1.7天)。

系统1 (主):
  入场   收盘 > 过去 ENTRY_N 根最高价 (不含当根)
  止损   入场价 − STOP_N × N     (N = ATR_LEN 根 ATR, 原版 N=20日ATR)
  出场   收盘 < 过去 EXIT_N 根最低价
系统2 (备用, LONG_ENTRY_N): 更长通道, 不受"上一次突破是否盈利"过滤影响
  原版规则: 系统1 的上次突破若盈利则跳过本次信号, 用系统2 兜底。
  这条过滤是海龟最反直觉也最关键的部件之一 —— 它避免在同一段趋势里反复追高。

仓位 (原版核心, 与"等名义"完全不同):
  1 单位 = RISK_PER_UNIT × 权益 / (STOP_N × N / 价格)
  即每单位的**风险**固定占权益 RISK_PER_UNIT(原版1%), 波动大的币自动缩仓。
  这与本项目做空腿的 SIZING=risk 是同一思想, 但海龟按 N 而非 ATR/价格中位数归一。

加仓金字塔:
  价格每涨 ADD_STEP_N × N 加 1 单位, 最多 MAX_UNITS 单位。
  每次加仓后, **全部仓位**的止损上移到"最新入场价 − STOP_N×N"(原版做法)。
"""
import numpy as np
import pandas as pd

DEFAULTS = dict(
    ENTRY_N=20,          # 入场通道(根): 收盘>过去N根最高
    EXIT_N=10,           # 出场通道(根): 收盘<过去N根最低
    LONG_ENTRY_N=55,     # 系统2 通道
    ATR_LEN=20,          # N 值 = 此长度的 ATR (原版20)
    STOP_N=2.0,          # 止损 = 入场价 − 此倍数 × N
    ADD_STEP_N=0.5,      # 每涨此倍数×N 加一单位
    MAX_UNITS=4,         # 单币最多单位数
    RISK_PER_UNIT=0.01,  # 每单位风险占权益比例 (原版1%)
    USE_FILTER=True,     # 系统1 的"上次突破盈利则跳过"过滤
)


def calc_indicators(df, p):
    df = df.copy()
    # 通道用**不含当根**的历史极值, 否则当根自己参与比较, 突破条件恒真
    df['hh_entry'] = df['high'].rolling(p['ENTRY_N']).max().shift(1)
    df['ll_exit'] = df['low'].rolling(p['EXIT_N']).min().shift(1)
    df['hh_long'] = df['high'].rolling(p['LONG_ENTRY_N']).max().shift(1)
    # N 值: 原版用 True Range 的指数式平均, 这里用简单平均(差异极小)
    tr = np.maximum(df['high'] - df['low'],
                    np.maximum((df['high'] - df['close'].shift()).abs(),
                               (df['low'] - df['close'].shift()).abs()))
    df['atr'] = tr.rolling(p['ATR_LEN']).mean()
    df['ma_slow'] = df['close']      # 占位, 供复用 portfolio_sim 的接口
    df['ma_fast'] = df['close']
    return df


def gen_signals(df, p):
    """entry_sig: 1=系统1突破 / 2=系统2突破 / 0=无。过滤在回测循环里做(需知上次盈亏)。"""
    df = df.copy()
    c = df['close'].values
    s1 = np.isfinite(df['hh_entry'].values) & (c > df['hh_entry'].values)
    s2 = np.isfinite(df['hh_long'].values) & (c > df['hh_long'].values)
    df['entry_sig'] = np.where(s1, 1, np.where(s2, 2, 0)).astype(int)
    df['exit_sig'] = (np.isfinite(df['ll_exit'].values) & (c < df['ll_exit'].values)).astype(int)
    return df
