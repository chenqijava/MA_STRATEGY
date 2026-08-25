"""组合层模拟: 固定总敞口, 变同时持仓上限 N, 看夏普/回撤/收益如何变化。

关键设计:
  单币名义 = 权益 × TOTAL_NOTIONAL / N  →  总敞口不随 N 变, 纯粹对比"分散度"。
  (若单币名义固定不变而加 N, 那是加杠杆, 不是分散, 不是本实验要测的。)

模型: 4h, 多空双腿。信号收盘开仓, ATR止盈止损盘中触发(止损优先), 超MAX_BARS根时间止损。
盈亏 = 名义 × 方向·(出-进)/进, 跨币用价格百分比, 不依赖合约面值。成本 taker+滑点沿用回测。

牛熊判断 (REGIME): 用 BTC 日线相对 MA(BTC_MA) 的位置+斜率判牛熊, 按下面的 4 格系数
缩放各腿敞口 —— 牛市重做多轻做空, 熊市反之。系数=0 即该腿在该状态完全不开仓。
多空槽位共享 N: 总敞口始终受 TOTAL_NOTIONAL 约束, 牛市砍空头等于净降风险。
"""
import os, sys
import numpy as np
import pandas as pd
import ma_pullback as mp
from analyze_long import BASE, path

if sys.stdout.encoding and not sys.stdout.encoding.lower().startswith('utf'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

_UNI = os.environ.get('UNIVERSE', 'universe_pass.txt')   # 可切换币种名单文件
SYMS = open(os.path.join(os.path.dirname(__file__), _UNI), encoding='utf-8').read().strip().split(',')
INIT = 10000.0
TOTAL_NOTIONAL = float(os.environ.get('TOTAL', '3.0'))   # 满仓总名义/权益 (3.0 = 你当前 5币×60%)
FEE = mp.TAKER_FEE
SLIP = mp.SLIPPAGE                                        # 入场滑点(沿用回测)
# 实盘出场损耗建模(回测忽略了这些, 会乐观高估):
EXIT_SLIP = float(os.environ.get('EXIT_SLIP', '0.0005'))  # 出场市价单滑点 (默认0.05%)
SLIP_THRU = float(os.environ.get('SLIP_THRU', '0.3'))     # 止损击穿系数: 成交价在stop与当根极值间取此比例(0=理想封顶, 1=最坏成交在极值)
FUNDING_BP = float(os.environ.get('FUNDING_BP', '0'))     # 每根K线资金费(bp, 占名义), 做空付为正; 默认0(未建模)
MAX_BARS = int(os.environ.get('MAX_BARS', '40'))          # 时间止损: 持仓超N根按收盘平; 0=关
# ---- 牛市专用出场 (可选): 牛市里空头优势可能只存在于头几根, 拿久了被反弹吃掉 ----
# 未设置则沿用全局 ATR_TP/ATR_SL/MAX_BARS (即不区分牛熊)。出场参数在**开仓时**按当时
# 的牛熊状态固定, 与实盘一致(挂单即定, 不随后续状态漂移)。
# ---- MA 出场 (可选): 做空持仓若收盘回到慢线上方, 立即市价平 ----
# 逻辑依据: 死叉反弹做空的前提是"价格被慢线压制"。收盘站上慢线 = 前提已破,
# 继续持有就是在赌一个不再成立的结构。比等 ATR 止损(2·ATR)更早离场。
# 与 ATR 止损的关系: 二者取先触发者; 止损仍在, MA 出场只是多一道更早的闸。
MA_EXIT = os.environ.get('MA_EXIT', '0') == '1'
MA_EXIT_BULL_ONLY = os.environ.get('MA_EXIT_BULL_ONLY', '0') == '1'   # 只在牛市启用
# ---- 重入冷却 (可选) ----
# 平仓后同一币在 COOLDOWN 根内不再开新仓。原逻辑允许"平了下一根又开": 死叉窗口
# 只要仍有效、价格再触碰一次慢线就重新入场, 时间止损/MA出场后尤其容易来回打。
# COOLDOWN_ONCE=1 则更严: 同一次死叉窗口只做一次 (窗口失效后才允许再做)。
COOLDOWN = int(os.environ.get('COOLDOWN', '0'))
COOLDOWN_ONCE = os.environ.get('COOLDOWN_ONCE', '0') == '1'
TP_BULL = os.environ.get('TP_BULL')                       # 牛市止盈 ATR 倍数
SL_BULL = os.environ.get('SL_BULL')                       # 牛市止损 ATR 倍数
BARS_BULL = os.environ.get('BARS_BULL')                   # 牛市时间止损根数
# ---- 仓位分配方式 ----
# 'notional' 等名义: 每仓名义 = 权益·TOTAL/N。但止损是 ATR_SL·ATR, 所以单笔风险
#            = 名义·ATR_SL·(ATR/价格), 而 ATR/价格 在池内差 3~4 倍(BTC~1%, 小市值~4%)
#            => 高波动币的每笔风险是 BTC 的数倍, 盈亏被它们主导。
# 'risk'     风险等分: 每笔风险占权益固定比例, 名义 ∝ 1/(ATR_SL·ATR/价格)。
#            为了与等名义在**同等总敞口**下可比, 风险预算由 TOTAL 反推(见下), 而非另设参数。
SIZING = os.environ.get('SIZING', 'notional')
RISK_CAP = float(os.environ.get('RISK_CAP', '3.0'))       # risk模式单仓名义上限(倍等名义), 防低波动币开出天量仓位
# ---- 组合层动态风控(可选) ----
DD_HALT = float(os.environ.get('DD_HALT', '0'))           # 回撤熔断: 组合回撤超此%则暂停开新仓; 0=关
DD_RESUME = float(os.environ.get('DD_RESUME', '0'))       # 回撤收敛到此%以下恢复开仓 (默认=DD_HALT的一半)
VOL_TARGET = float(os.environ.get('VOL_TARGET', '0'))     # 波动率目标(日波动%): 按组合近期已实现波动缩放敞口; 0=关
VOL_LOOKBACK = int(os.environ.get('VOL_LOOKBACK', '20'))  # 已实现波动率回看根数
VOL_CAP = float(os.environ.get('VOL_CAP', '1.5'))         # 波动率择时的敞口缩放上限(倍)
# ---- 腿开关 ----
LONG_ON = os.environ.get('LONG', '0') == '1'              # 做多腿(金叉回踩); 默认关(向后兼容旧的纯做空口径)
SHORT_ON = os.environ.get('SHORT', '1') == '1'            # 做空腿(死叉反弹)
# ---- BTC 牛熊判断: 4 格敞口系数 ----
# 旧口径 BTC_FILTER 仍生效(= 牛市做空系数), 保留是为了不打断已有的对比脚本。
BTC_MA = int(os.environ.get('BTC_MA', '30'))              # BTC 判牛熊的均线天数(日线)
BTC_SLOPE = int(os.environ.get('BTC_SLOPE', '10'))        # 判斜率的回看天数(MA 相对 N 天前是否上行)
BTC_FILTER = float(os.environ.get('BTC_FILTER', '0'))     # 兼容旧口径: 牛市做空系数(0=牛市停空)
SHORT_BULL = float(os.environ.get('SHORT_BULL', str(BTC_FILTER)))   # 牛市里做空的敞口系数
SHORT_BEAR = float(os.environ.get('SHORT_BEAR', '1'))               # 熊市里做空的敞口系数
LONG_BULL = float(os.environ.get('LONG_BULL', '1'))                 # 牛市里做多的敞口系数
LONG_BEAR = float(os.environ.get('LONG_BEAR', '0'))                 # 熊市里做多的敞口系数
# 是否分牛熊由调用方决定: simulate(bull=None) 即全程系数 1.0, 不需要额外开关。
# ---- 牛熊判定器选择 ----
# 'btc'     : BTC 日线 > MA(BTC_MA) 且 MA 上行 (原口径, 单一标的)
# 'breadth' : 市场宽度 —— 币池中收盘价在自身 MA(BREADTH_MA) 上方的比例 > BREADTH_TH 判牛。
#             宽度用全池信息, 对单币噪声不敏感; 且"多少币在涨"本身就是做空腿的对立面,
#             比 BTC 单价更贴近"整个池子能不能做空"这个问题。
REGIME = os.environ.get('REGIME', 'btc')
# ---- 逐币日线趋势过滤 (与上面的全市场 REGIME 不同: 这是每个币看自己的日线) ----
# 用当日之前已收盘的日线判断(shift(1)), 无前视。D_MODE 选规则:
#   'ma'   : 日线 MA(D_MA) 向下 且 日收盘在 MA 下方        —— 看"位置"
#   'adx'  : 日线 ADX(D_ADX_LEN) > D_ADX_TH 且 -DI > +DI   —— 看"下跌趋势的力度"
#   'both' : 两者同时满足
D_FILTER = os.environ.get('D_FILTER', '0') == '1'
D_MODE = os.environ.get('D_MODE', 'ma')
D_MA = int(os.environ.get('D_MA', '20'))
D_SLOPE = int(os.environ.get('D_SLOPE', '1'))     # MA 相对 N 日前更低才算"向下"
D_ADX_LEN = int(os.environ.get('D_ADX_LEN', '14'))
D_ADX_TH = float(os.environ.get('D_ADX_TH', '25'))
# ---- 慢线自身斜率过滤 (4h 周期上的 MA_SLOW, 不是日线) ----
# MS_SLOPE='up'   : 要求 ma_slow[MS_L1] > ma_slow[MS_L2]  (慢线仍在上行)
# MS_SLOPE='down' : 要求 ma_slow[MS_L1] < ma_slow[MS_L2]  (慢线已在下行)
# 索引口径与 Pine 一致: [1]=上一根, [6]=六根前, 都是已收盘的值, 无前视。
# 'up' 的逻辑: 死叉反弹做空最好的位置可能是"长期趋势还没坏、价格先跌破慢线"的早期,
# 此时空头对手盘(还在做多的人)最多; 慢线已经掉头向下反而说明行情走完了大半。
MS_SLOPE = os.environ.get('MS_SLOPE', '0')
MS_L1 = int(os.environ.get('MS_L1', '1'))
MS_L2 = int(os.environ.get('MS_L2', '6'))
# ---- 波动放大过滤 ----
# 只在 ATR(14) > 过去 VE_LEN 根 ATR 均值 × VE_MULT 时开仓, 即波动正在放大。
# 逻辑: 低波动横盘时死叉回踩信号密集但幅度小, 手续费+滑点占比高; 波动放大才有行情。
# 与前面几个过滤器的维度不同 —— 那些看趋势方向, 这个看波动状态, 不是重复条件。
# 均值用 shift(1) 排除当根, 是严格的"过去N根", 无前视 (ATR[i] 本身在收盘时已知)。
VOL_EXP = os.environ.get('VOL_EXP', '0') == '1'
VE_LEN = int(os.environ.get('VE_LEN', '20'))
VE_MULT = float(os.environ.get('VE_MULT', '1.0'))
VE_INVERT = os.environ.get('VE_INVERT', '0') == '1'   # 反过来只做低波动那批(验证被砍的赚不赚)
# ---- 移动止盈 / 保本止损 (可选, 四个独立部件, 可组合) ----
# BE_ATR    >0: 浮盈达此倍数ATR后, 把止损移到成本价(保本)。防"回吐已有浮盈"。
#               注意这会把"本可继续跑的赢家"在回撤到成本时砍掉 —— 保本不是免费的。
# TRAIL_ATR >0: 移动止损 = 持仓期间最低价(做空) + 此倍数·ATR, 只朝有利方向移动。
# TRAIL_PCT >0: 移动止损 = 持仓期最优价(做空最低/做多最高) 回撤此比例即平, 如 0.05=5%。
#               与 TRAIL_ATR 的区别: 止损间距按**价格百分比**而非ATR倍数, 不受各币
#               ATR/价格比差异影响(高波动币在 TRAIL_ATR 下实际止损空间更宽松)。
# TRAIL_MA  >0: 追踪均线出场 —— 收盘上穿 MA(此长度) 即平 (做空)。比 MA90 快得多。
# TRAIL_ARM >0: 移动止损/追踪均线的启动门槛(浮盈达此倍ATR后才启用), 0=开仓即启用。
#               先赚够再收紧, 避免开仓初期的正常波动就把仓打掉。
# ---- 分批止盈 (可选) ----
# PT_LEVELS: 逗号分隔的 ATR 倍数, 到达即平掉一部分 (如 '3,6' = 浮盈3倍ATR平一批, 6倍再平一批)
# PT_FRACS:  对应比例, 占**原始**名义 (如 '0.5,0.25' = 先平半仓, 再平原仓的四分之一)
# PT_BE=1:   第一批止盈后把止损移到成本价 (经典的"分批止盈+保本"组合)
# 口径说明: 一笔交易的盈亏 = 各批次之和, 在最终清仓时作为**一笔**计入 trades。
#   若每批各记一笔, 笔数和胜率都会被稀释(分批本身就会造出一堆小赢单), 无法与基线比。
PT_LEVELS = os.environ.get('PT_LEVELS', '')
PT_FRACS = os.environ.get('PT_FRACS', '')
PT_BE = os.environ.get('PT_BE', '0') == '1'
_PT = []
if PT_LEVELS and PT_FRACS:
    _lv = [float(x) for x in PT_LEVELS.split(',') if x.strip()]
    _fr = [float(x) for x in PT_FRACS.split(',') if x.strip()]
    if len(_lv) != len(_fr):
        raise SystemExit(f'PT_LEVELS({len(_lv)}) 与 PT_FRACS({len(_fr)}) 数量不一致')
    if sum(_fr) > 1.0 + 1e-9:
        raise SystemExit(f'PT_FRACS 合计 {sum(_fr)} > 1, 会平掉超过原仓位')
    _PT = sorted(zip(_lv, _fr))          # 按触发倍数升序, 保证先近后远
BE_ATR = float(os.environ.get('BE_ATR', '0'))
TRAIL_ATR = float(os.environ.get('TRAIL_ATR', '0'))
TRAIL_PCT = float(os.environ.get('TRAIL_PCT', '0'))       # 移动止损: 从最优价回撤此比例(如0.05=5%)即平
TRAIL_MA = int(os.environ.get('TRAIL_MA', '0'))
TRAIL_ARM = float(os.environ.get('TRAIL_ARM', '0'))
# ---- 长多填仓（叠加层, 不占 N 槽位, 用闲置容量做多）- FILLS 注册制 ----
# 当短空仓位不满时, 用闲置资金做多。每项一个 prefix 环境变量开启:
#   <PFX>=1 时注册一条 fill, 参数可用 <PFX>_MA_F/<PFX>_MA_S/<PFX>_MA_ONCE/
#   <PFX>_TRAIL_ATR/<PFX>_TRAIL_PCT/<PFX>_TRAIL_ARM/<PFX>_ATR_LEN/<PFX>_SRC 覆盖。
# 入场条件: 标的 4h MA(MA_F) > MA(MA_S) (金叉);
#   ONCE=1: 只在金叉**当根**入场一次, 窗口内(快线仍>慢线)不重复入场,
#           等快线回落到慢线下方(窗口结束)后的下一次新金叉才再允许。
#           =0: 水平条件(快线>慢线 全程开多), 熊市反弹期反复进出被打止损。
# SRC: close/high —— MA 计算用的价格源 (high 对应标的单腿回测口径)。
# 出场: 追踪止损二选一 —— TRAIL_ATR>0 用 ATR 倍数(×ATR(ATR_LEN)),
#       否则用价格百分比 TRAIL_PCT (从持仓期最高价回撤)。
FILLS = []   # 每项: dict(prefix,sym,ma_f,ma_s,once,trail_atr,trail_pct,arm,atr_len,src)

def _reg_fill(prefix, sym, ma_f, ma_s, once, trail_atr, trail_pct, arm, atr_len, src):
    if os.environ.get(prefix, '0') != '1':
        return
    FILLS.append(dict(
        prefix=prefix, sym=sym,
        ma_f=int(os.environ.get(prefix + '_MA_F', ma_f)),
        ma_s=int(os.environ.get(prefix + '_MA_S', ma_s)),
        once=os.environ.get(prefix + '_ONCE', '1' if once else '0') == '1',
        trail_atr=float(os.environ.get(prefix + '_TRAIL_ATR', trail_atr)),
        trail_pct=float(os.environ.get(prefix + '_TRAIL_PCT', trail_pct)),
        arm=float(os.environ.get(prefix + '_TRAIL_ARM', arm)),
        atr_len=int(os.environ.get(prefix + '_ATR_LEN', atr_len)),
        src=os.environ.get(prefix + '_SRC', src)))

_reg_fill('BTC_FILL', 'BTC', 10, 120, True, 3.0, 0, 0, 20, 'close')   # 与旧 BTC_FILL 行为一致
_reg_fill('ETH_FILL', 'ETH', 20, 85,  True, 4.0, 0, 0, 18, 'high')    # ETH 单腿回测参数
# D_INVERT=1: 反过来只做被过滤掉的那批信号。用来验证"过滤掉的是不是赚钱的单":
# 若被丢掉的那批 PF 也 >1, 那这个过滤器就是在扔钱, 而不是在挡亏损。
D_INVERT = os.environ.get('D_INVERT', '0') == '1'
BREADTH_MA = int(os.environ.get('BREADTH_MA', '50'))      # 宽度用的日线均线长度
BREADTH_TH = float(os.environ.get('BREADTH_TH', '0.5'))   # 判牛阈值(比例)
BREADTH_SM = int(os.environ.get('BREADTH_SM', '1'))       # 宽度序列平滑天数(1=不平滑)

CFG = dict(BASE, ENABLE_LONG=LONG_ON, ENABLE_SHORT=SHORT_ON,
           MA_SLOW=int(os.environ.get('MA_SLOW', '30')),   # 与实盘配置一致
           # ATR 止盈止损也走环境变量: 实盘用 10/10 (极端保险), 而 BASE 里写死的是 4/2。
           # 默认值保持 BASE 的 4/2, 不改动已有脚本的口径。
           ATR_TP=float(os.environ.get('ATR_TP', BASE['ATR_TP'])),
           ATR_SL=float(os.environ.get('ATR_SL', BASE['ATR_SL'])),
           # K线形态加强 (见 ma_pullback.gen_signals): 0/空 = 关闭
           UPPER_SHADOW=float(os.environ.get('UPPER_SHADOW', '0')),
           BREAK_PREV_LOW=os.environ.get('BREAK_PREV_LOW', '0') == '1')


def build_panel():
    cols = {}
    for key in ('close', 'high', 'low', 'atr', 'entry_sig', 'ma_slow', 'ma_fast'):
        cols[key] = {}
    for s in SYMS:
        p = path(s)
        if not p:
            continue
        ds = mp.gen_signals(mp.calc_indicators(mp.cc.load_data(p), CFG), CFG)
        ds = ds.iloc[CFG['MA_SLOW'] + 2:]
        for key in cols:
            cols[key][s] = ds[key]
    C = pd.DataFrame(cols['close']).sort_index()
    idx = C.index
    H = pd.DataFrame(cols['high']).reindex(idx)
    L = pd.DataFrame(cols['low']).reindex(idx)
    A = pd.DataFrame(cols['atr']).reindex(idx)
    S = pd.DataFrame(cols['entry_sig']).reindex(idx)
    # 慢线走模块级变量而非返回值: build_panel 的返回被多个脚本按位置解包, 加一项会全破。
    global _MA_SLOW_PANEL, _MA_FAST_PANEL, _DTREND_PANEL, _TRAIL_MA_PANEL
    _MA_SLOW_PANEL = pd.DataFrame(cols['ma_slow']).reindex(idx)
    _MA_FAST_PANEL = pd.DataFrame(cols['ma_fast']).reindex(idx)
    # 追踪均线面板 (TRAIL_MA): 现有面板只有 MA5/MA90, MA10/MA20 需另算。
    # 直接由收盘价 rolling 求, 与 calc_indicators 的 sma 口径一致(当根含在内, 收盘时已知)。
    _TRAIL_MA_PANEL = C.rolling(TRAIL_MA).mean() if TRAIL_MA else None
    # 逐币日线趋势: 由 4h 数据重采样成日线(不需要额外拉数据), 用上一日已收盘的状态判断
    _DTREND_PANEL = None
    if D_FILTER:
        cols_d = {}
        for s in C.columns:
            c1d = C[s].resample('1D').last().dropna()
            ok = pd.Series(True, index=c1d.index)
            if D_MODE in ('ma', 'both'):
                ma = c1d.rolling(D_MA).mean()
                ok &= (ma < ma.shift(D_SLOPE)) & (c1d < ma)   # MA向下 且 收盘在MA下方
            if D_MODE in ('adx', 'both'):
                h1d = H[s].resample('1D').max().reindex(c1d.index)
                l1d = L[s].resample('1D').min().reindex(c1d.index)
                adx, pdi, mdi = _adx(h1d, l1d, c1d, D_ADX_LEN)
                ok &= (adx > D_ADX_TH) & (mdi > pdi)          # 趋势够强 且 方向向下
            ok = ok.shift(1).fillna(False)                    # 用上一日状态, 无前视
            days = pd.Series(ok.values, index=ok.index.normalize())
            cols_d[s] = pd.Series([bool(days.get(t.normalize(), False)) for t in idx], index=idx)
        _DTREND_PANEL = pd.DataFrame(cols_d).reindex(index=idx, columns=C.columns)
    return idx, C, H, L, A, S, list(C.columns)


def _adx(high, low, close, n):
    """Wilder 原版 ADX / +DI / -DI。返回 (adx, plus_di, minus_di), 索引与输入一致。

    只用当根及之前的数据(diff/shift 都朝过去取), 无前视。Wilder 平滑用
    ewm(alpha=1/n) 等价于原版的递推 smooth = prev - prev/n + cur。
    """
    up = high.diff()
    dn = -low.diff()
    # +DM / -DM: 只有"单边超出更多"的那侧计数, 另一侧记 0 (Wilder 定义)
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx, pdi, mdi


_MA_SLOW_PANEL = None
_MA_FAST_PANEL = None
_DTREND_PANEL = None    # D_FILTER=1 时: 逐币日线趋势允许做空的布尔面板
_TRAIL_MA_PANEL = None  # TRAIL_MA>0 时: 追踪用的中期均线面板 (MA10/MA20 等)
_HELD_TRACE = None      # simulate() 每次运行后写入: 逐根持仓币数
# 每笔交易的**相对**收益率 (pnl / 开仓时权益)。绝对金额不能直接用于 bootstrap:
# 名义 = 权益×TOTAL/N 随权益复利变化, 后期的笔天然金额更大, 重排会把顺序信息
# 混进幅度里。相对收益率是尺度无关的, 才能重抽。见 monte_carlo.py。
_TRADE_TRACE = None


def btc_bull_mask(idx):
    """对齐 panel 索引的"BTC牛市"布尔数组。牛市判定: 日线收盘>MA 且 MA上行。
    只用当日及之前数据(shift对齐无未来函数): 每根4h bar 用其所在日"上一日已收盘"的状态。"""
    btc = mp.cc.load_data(path('BTC'))['close'].resample('1D').last().dropna()
    ma = btc.rolling(BTC_MA).mean()
    bull_daily = (btc > ma) & (ma > ma.shift(BTC_SLOPE))
    bull_daily = bull_daily.shift(1).fillna(False)     # 用上一日状态, 避免当日未收盘的前视
    # 映射到每个 4h bar: 取该 bar 日期对应的日线状态
    return _daily_to_bars(bull_daily, idx)


def _daily_to_bars(bull_daily, idx):
    """把日线布尔状态映射到每根 4h bar (取该 bar 所在日期的状态)。"""
    days = pd.Series(bull_daily.values, index=bull_daily.index.normalize())
    return np.array([bool(days.get(t.normalize(), False)) for t in idx])


def breadth_series(syms=None):
    """市场宽度日线序列: 币池中"日收盘 > 自身 MA(BREADTH_MA)"的比例。

    对每个币独立算 MA, 再按日横截面取比例 —— 只用各币自身历史, 无前视。
    新上线的币在其数据开始前不计入分母(dropna 后按当日有效币数归一)。
    """
    syms = syms or SYMS
    flags = {}
    for s in syms:
        p = path(s)
        if not p:
            continue
        c = mp.cc.load_data(p)['close'].resample('1D').last().dropna()
        if len(c) < BREADTH_MA + 5:
            continue
        ma = c.rolling(BREADTH_MA).mean()
        flags[s] = (c > ma).where(ma.notna())      # MA 未成型的日子记 NaN, 不计入分母
    F = pd.DataFrame(flags)
    br = F.sum(axis=1) / F.notna().sum(axis=1).replace(0, np.nan)
    return br.dropna()


def breadth_bull_mask(idx, syms=None):
    """宽度判牛: 宽度 > BREADTH_TH。用上一日状态避免当日未收盘的前视。"""
    br = breadth_series(syms)
    if BREADTH_SM > 1:
        br = br.rolling(BREADTH_SM).mean()
    bull_daily = (br > BREADTH_TH).shift(1).fillna(False)
    return _daily_to_bars(bull_daily, idx)


def regime_mask(idx, syms=None):
    """按 REGIME 选判定器。"""
    return breadth_bull_mask(idx, syms) if REGIME == 'breadth' else btc_bull_mask(idx)


def simulate(N, idx, C, H, L, A, S, bull=None):
    Cv, Hv, Lv, Av, Sv = C.values, H.values, L.values, A.values, S.values
    # 慢线面板 (MA出场用); 列序与 C 对齐后取值。
    # 显式报错而非静默跳过: 模块 reload 会清空这个全局, 若此时复用缓存的面板,
    # MA出场会悄悄失效并给出与"未开启"完全相同的结果 —— 那是最难发现的一类错。
    MAv = None
    if MA_EXIT:
        if _MA_SLOW_PANEL is None:
            raise RuntimeError('MA_EXIT=1 但慢线面板为空; 请在本次 import 后调用 build_panel()')
        MAv = _MA_SLOW_PANEL.reindex(index=C.index, columns=C.columns).values
    # 快线面板 (COOLDOWN_ONCE 判死叉窗口是否已失效)
    MFv = None
    if COOLDOWN_ONCE:
        if _MA_FAST_PANEL is None or _MA_SLOW_PANEL is None:
            raise RuntimeError('COOLDOWN_ONCE=1 但均线面板为空; 请在本次 import 后调用 build_panel()')
        MFv = _MA_FAST_PANEL.reindex(index=C.index, columns=C.columns).values
        if MAv is None:
            MAv = _MA_SLOW_PANEL.reindex(index=C.index, columns=C.columns).values
    # 逐币日线趋势面板 (同上: reload 会清空, 必须显式报错而不是静默放行所有入场)
    DTv = None
    if D_FILTER:
        if _DTREND_PANEL is None:
            raise RuntimeError('D_FILTER=1 但日线趋势面板为空; 请在本次 import 后调用 build_panel()')
        DTv = _DTREND_PANEL.reindex(index=C.index, columns=C.columns).values
        if D_INVERT:
            DTv = ~DTv.astype(bool)
    # 慢线斜率面板: ma_slow[MS_L1] vs ma_slow[MS_L2] (4h 周期, 都是已收盘值)
    MSv = None
    if MS_SLOPE in ('up', 'down'):
        if _MA_SLOW_PANEL is None:
            raise RuntimeError('MS_SLOPE 已设但慢线面板为空; 请在本次 import 后调用 build_panel()')
        _ms = _MA_SLOW_PANEL.reindex(index=C.index, columns=C.columns)
        a, b = _ms.shift(MS_L1), _ms.shift(MS_L2)
        MSv = ((a > b) if MS_SLOPE == 'up' else (a < b)).values
    # 波动放大面板: ATR(14) > 过去 VE_LEN 根 ATR 均值 × VE_MULT
    VEv = None
    if VOL_EXP:
        _a = A.reindex(index=C.index, columns=C.columns)
        base = _a.shift(1).rolling(VE_LEN).mean()        # shift(1): 严格"过去N根", 不含当根
        VEv = (_a > base * VE_MULT).fillna(False).values
        if VE_INVERT:
            VEv = ~VEv.astype(bool)
    # 追踪均线面板 (同上: reload 会清空, 显式报错而不是静默失效)
    TMv = None
    if TRAIL_MA:
        if _TRAIL_MA_PANEL is None:
            raise RuntimeError('TRAIL_MA 已设但均线面板为空; 请在本次 import 后调用 build_panel()')
        TMv = _TRAIL_MA_PANEL.reindex(index=C.index, columns=C.columns).values
    T, nS = Cv.shape
    global _HELD_TRACE, _TRADE_TRACE
    _HELD_TRACE = np.zeros(T, dtype=int)  # 每根K线的持仓币数(0 = 空仓)
    _TRADE_TRACE = []                     # (相对收益率, 持仓根数, 出场类型)
    cool = np.zeros(nS, dtype=int)        # 各币剩余冷却根数
    done = np.zeros(nS, dtype=bool)       # COOLDOWN_ONCE: 本次窗口是否已做过
    # ---- 长多填仓预计算: 每标的一段 MA/ATR/金叉 (FILLS 注册制) ----
    # fill_states[i]: dict(cfg, j, ma_f, ma_s, atr, gold, pos=None, done=False)
    fill_states = []
    for f in FILLS:
        j = next((j for j, s in enumerate(C.columns) if s == f['sym']), None)
        if j is None:
            continue
        _bs = (Hv if f['src'] == 'high' else Cv)[:, j]   # src 决定 MA 价格源 (BTC: close, ETH: high)
        if not np.any(np.isfinite(_bs)):
            continue
        ma_f = pd.Series(_bs).rolling(f['ma_f']).mean().values
        ma_s = pd.Series(_bs).rolling(f['ma_s']).mean().values
        # 金叉当根: 本根 MA_F > MA_S 且 上一根 MA_F <= MA_S (用 shift 排除当根, 无前视)
        g = pd.Series(ma_f > ma_s) & (pd.Series(ma_f).shift(1) <= pd.Series(ma_s).shift(1))
        _bh, _bl = Hv[:, j], Lv[:, j]
        _bc = Cv[:, j]
        tr = np.maximum(_bh - _bl,
                        np.maximum(np.abs(_bh - np.roll(_bc, 1)),
                                   np.abs(_bl - np.roll(_bc, 1))))
        tr[0] = np.nan
        fill_states.append(dict(cfg=f, j=j, ma_f=ma_f, ma_s=ma_s,
                                atr=pd.Series(tr).rolling(f['atr_len']).mean().values,
                                gold=g.fillna(False).values,
                                pos=None, done=False))
    equity = INIT
    held = {}
    curve = np.empty(T)
    trades = []
    held_sum = 0
    peak = INIT                          # 组合权益峰值(回撤熔断用)
    halted = False                       # 是否处于熔断暂停状态
    dd_resume = DD_RESUME or (DD_HALT / 2 if DD_HALT else 0)
    BARS_PER_DAY = 6                     # 4h 周期 = 每日6根 (供波动率换算)
    # risk 模式的风险预算: 取全样本 ATR/价格 的中位数, 令"中位波动币"拿到与等名义
    # 完全相同的仓位。这样两种模式的总敞口口径一致, 差异纯粹来自跨币再分配, 而不是
    # 偷偷加/减杠杆 —— 否则比的是杠杆不是分配方式。
    ref_vol = np.nan
    if SIZING == 'risk':
        v = (Av / Cv).ravel()
        v = v[np.isfinite(v) & (v > 0)]
        ref_vol = np.median(v) if len(v) else np.nan
    for i in range(T):
        # ---- 冷却递减 / 死叉窗口失效则解除"已做过"标记 ----
        if COOLDOWN:
            np.subtract(cool, 1, out=cool, where=cool > 0)
        if COOLDOWN_ONCE and MFv is not None:
            # 窗口结束 = 快线回到慢线另一侧。做空(死叉窗口)看 快线>=慢线; 做多(金叉窗口)
            # 反之。两腿同开时任一侧回归都算窗口翻转, 用异或形式覆盖: 只要快慢线关系
            # 与开仓时相反即解除标记。这里按当前启用的腿取对应条件。
            ok = np.isfinite(MFv[i]) & np.isfinite(MAv[i])
            if SHORT_ON and not LONG_ON:
                reopened = MFv[i] >= MAv[i]
            elif LONG_ON and not SHORT_ON:
                reopened = MFv[i] <= MAv[i]
            else:
                # 双腿: 记录开仓时的方向才能判翻转, 简化为"穿越即重置"(与上一根比符号变化)
                prev = MFv[i - 1] - MAv[i - 1] if i > 0 else MFv[i] - MAv[i]
                reopened = np.sign(MFv[i] - MAv[i]) != np.sign(prev)
            done[ok & reopened] = False
        # ---- 出场: 已持仓检查 ATR 止盈止损 (止损优先), 建模实盘市价出场损耗 ----
        for j in list(held.keys()):
            h, l = Hv[i, j], Lv[i, j]
            if not np.isfinite(h):
                continue
            st = held[j]
            st['bars'] += 1
            d = st['dir']                  # +1 多 / -1 空
            # 出场滑点方向: 平多是卖出(成交价更低=更差), 平空是买回(成交价更高=更差)
            slip_sign = -d
            # ---- 保本止损 / 移动止损: 在本根的止损判定之前更新 stop ----
            # 顺序很关键: 必须先按上一根收盘时已知的信息把 stop 收紧, 再用本根的 high/low
            # 判是否触发。反过来(先判触发再收紧)等于用本根的极值去决定本根的止损位, 是前视。
            if (BE_ATR or TRAIL_ATR or TRAIL_PCT) and st['atr'] > 0:
                # 浮盈(按有利方向的极值算, 单位=ATR倍数): 做空看本根之前的最低价
                ext_fav = st['ext']            # 持仓期间最有利的价格(空=最低, 多=最高)
                # 浮盈 = 方向 × (有利极值 - 入场) / ATR。做空 d=-1 且 ext<entry, 两个负号
                # 相乘得正 —— 写成 (entry-ext) 会符号反掉, 导致条件永不成立(静默失效)。
                gain_atr = d * (ext_fav - st['entry']) / st['atr']
                if BE_ATR and gain_atr >= BE_ATR:
                    # 保本: 止损移到成本价 (只朝有利方向移, 不放宽)
                    st['stop'] = (min(st['stop'], st['entry']) if d < 0
                                  else max(st['stop'], st['entry']))
                if TRAIL_ATR and gain_atr >= TRAIL_ARM:
                    # 移动止损 = 有利极值 + TRAIL_ATR·ATR (空头在上方, 多头在下方)
                    cand = ext_fav - d * TRAIL_ATR * st['atr']
                    st['stop'] = (min(st['stop'], cand) if d < 0
                                  else max(st['stop'], cand))
                if TRAIL_PCT and gain_atr >= TRAIL_ARM:
                    # 移动止损 = 有利极值 × (1 ∓ TRAIL_PCT) (空头在上方, 多头在下方);
                    # 按价格百分比而非ATR倍数, 各币止损间距与自身波动率无关。
                    cand = ext_fav * (1 - d * TRAIL_PCT)
                    st['stop'] = (min(st['stop'], cand) if d < 0
                                  else max(st['stop'], cand))
            if (d < 0 and h >= st['stop']) or (d > 0 and l <= st['stop']):
                # 止损: 市价成交并建模"击穿"—— 急动时价格穿过stop, 真实成交介于 stop
                # 与当根极值之间(SLIP_THRU 比例), 亏损可能远超理论 ATR_SL·ATR。
                # 空头 ext=high>stop → thru 上移; 多头 ext=low<stop → thru 下移, 同式两边通用。
                ext = h if d < 0 else l
                thru = st['stop'] + SLIP_THRU * (ext - st['stop'])
                exit_px = thru * (1 + slip_sign * EXIT_SLIP)
                st['why'] = 'stop'
            elif (d < 0 and l <= st['target']) or (d > 0 and h >= st['target']):
                # 止盈: 市价成交价比理论 target 略差
                exit_px = st['target'] * (1 + slip_sign * EXIT_SLIP)
                st['why'] = 'target'
            elif st['max_bars'] and st['bars'] >= st['max_bars'] and np.isfinite(Cv[i, j]):
                # 时间止损: 超MAX_BARS根按收盘市价平(叠出场滑点)
                exit_px = Cv[i, j] * (1 + slip_sign * EXIT_SLIP)
                st['why'] = 'time_stop'
            elif st['ma_exit'] and MAv is not None and np.isfinite(MAv[i, j]) \
                    and np.isfinite(Cv[i, j]) and d * (Cv[i, j] - MAv[i, j]) < 0:
                # MA出场: 做空收盘站上慢线(或做多跌破慢线) -> 压制结构已破, 收盘市价平。
                # 排在止损/止盈之后: 同一根内若已触及止损价, 实盘会先被止损单成交。
                exit_px = Cv[i, j] * (1 + slip_sign * EXIT_SLIP)
                st['why'] = 'ma_exit'
            elif TMv is not None and np.isfinite(TMv[i, j]) and np.isfinite(Cv[i, j]) \
                    and d * (Cv[i, j] - TMv[i, j]) < 0 \
                    and (not TRAIL_ARM or st['atr'] <= 0
                         or d * (st['ext'] - st['entry']) / st['atr'] >= TRAIL_ARM):
                # 追踪均线出场: 收盘上穿 MA(TRAIL_MA) 即平 (做空)。比 MA90 快得多,
                # 只在浮盈已达 TRAIL_ARM·ATR 后启用(否则开仓初期就会被打掉 —— MA10
                # 离价格很近, 入场当根之后几乎立刻会被穿)。
                exit_px = Cv[i, j] * (1 + slip_sign * EXIT_SLIP)
                st['why'] = 'trail_ma'
            else:
                exit_px = None
            # ---- 分批止盈: 无整仓出场时, 检查是否触及下一档 ----
            # 用本根极值判触发(与ATR止盈同口径: 盘中触发), 成交价按该档理论价叠出场滑点。
            if exit_px is None and _PT and st['atr'] > 0:
                while st['pt_i'] < len(_PT):
                    lv, fr = _PT[st['pt_i']]
                    px = st['entry'] + d * lv * st['atr']    # 空(d=-1): 目标在入场价下方
                    hit = (l <= px) if d < 0 else (h >= px)
                    if not (hit and np.isfinite(px)):
                        break
                    part = st['notional0'] * fr
                    part = min(part, st['notional'])       # 不能平超过剩余
                    if part <= 0:
                        st['pt_i'] += 1
                        continue
                    fill = px * (1 + slip_sign * EXIT_SLIP)
                    st['realized'] += part * d * (fill - st['entry']) / st['entry'] - part * FEE
                    st['notional'] -= part
                    st['pt_i'] += 1
                    if PT_BE:      # 第一批之后止损移成本价
                        st['stop'] = (min(st['stop'], st['entry']) if d < 0
                                      else max(st['stop'], st['entry']))
                if st['notional'] <= 1e-9:                # 已全部分批平完
                    equity += st['realized']
                    trades.append(st['realized'])
                    _TRADE_TRACE.append((st['realized'] / st['eq0'] if st['eq0'] > 0 else 0.0,
                                         st['bars'], 'partial'))
                    del held[j]
                    cool[j] = COOLDOWN
                    done[j] = True
                    continue
            # 更新有利极值 (供下一根的保本/移动止损用)。放在出场判定之后:
            # 本根的极值不能影响本根的止损位, 否则是前视。
            if exit_px is None and np.isfinite(l) and np.isfinite(h):
                st['ext'] = min(st['ext'], l) if d < 0 else max(st['ext'], h)
            if exit_px is not None:
                # 整仓盈亏 = 剩余仓位的盈亏 + 已分批实现的盈亏。合成一笔计入 trades,
                # 否则分批会造出一堆小赢单, 稀释胜率与笔数, 无法与基线比。
                pnl = st['notional'] * d * (exit_px - st['entry']) / st['entry'] + st['realized']
                equity += pnl - st['notional'] * FEE
                net = pnl - st['notional'] * FEE
                trades.append(net)
                _TRADE_TRACE.append((net / st['eq0'] if st['eq0'] > 0 else 0.0,
                                     st['bars'], st.get('why', '?')))
                del held[j]
                cool[j] = COOLDOWN             # 平仓起算冷却
                done[j] = True                 # 本次窗口已做过 (COOLDOWN_ONCE 用)
        # ---- 长多填仓出场: 追踪止损（ATR 倍数或价格%, 从持仓期最高价回撤） ----
        for f in fill_states:
            p = f['pos']
            if p is None:
                continue
            h, l = Hv[i, f['j']], Lv[i, f['j']]
            if np.isfinite(h):
                p['bars'] += 1
                # 用上一根已知的 ext/ATR 算 stop; 本根高点要等出场判定之后才更新极值,
                # 否则本根的极值决定了本根的止损位, 是前视 (与主持仓口径一致)。
                # ATR 用上一根(f['atr'][i-1])而非当根: 当根 ATR 含当根极值, 用它判当根
                # 触发同样有前视。入场时 i>=慢线, i-1 恒合法。
                if f['cfg']['trail_atr'] > 0 and np.isfinite(f['atr'][i - 1]):
                    gap = f['cfg']['trail_atr'] * f['atr'][i - 1]
                elif f['cfg']['trail_pct'] > 0:
                    gap = f['cfg']['trail_pct'] * p['ext']
                else:
                    gap = 0
                # 启动门槛: 浮盈达 TRAIL_ARM·ATR 前用初始固定止损(入场-间距),
                # 否则开仓初期正常波动就把追踪止损打掉 (与主持仓 TRAIL_ARM 同一问题)。
                if f['cfg']['arm'] > 0 and np.isfinite(f['atr'][i - 1]):
                    gain_atr = (p['ext'] - p['entry']) / f['atr'][i - 1]
                    armed = gain_atr >= f['cfg']['arm']
                else:
                    armed = True
                stop = max(p['stop0'], p['ext'] - gap) if armed else p['stop0']
                if l <= stop and np.isfinite(stop):
                    exit_px = stop * (1 - EXIT_SLIP)    # 平多卖出, 成交价更低
                    pnl = p['notional'] * (exit_px - p['entry']) / p['entry']
                    net = pnl - p['notional'] * FEE
                    equity += net
                    trades.append(net)
                    _TRADE_TRACE.append((net / p['eq0'] if p['eq0'] > 0 else 0.0,
                                         p['bars'], f"{f['cfg']['sym'].lower()}_fill"))
                    f['pos'] = None
                elif np.isfinite(h):
                    p['ext'] = max(p['ext'], h)
        # ---- 组合层动态风控: 决定本根是否/以多大敞口开新仓 ----
        # (1) 回撤熔断: 当前回撤超 DD_HALT 则暂停开新仓, 收敛到 dd_resume 以下才恢复
        # 用上一根盯市权益作当前值(本根mtm尚未算), 与回撤指标口径一致
        cur_eq = curve[i - 1] if i > 0 else equity
        cur_dd = (peak - cur_eq) / peak * 100 if peak > 0 else 0
        if DD_HALT:
            if not halted and cur_dd >= DD_HALT:
                halted = True
            elif halted and cur_dd <= dd_resume:
                halted = False
        # (2) 波动率择时: 用组合近期已实现日波动率, 缩放单币名义 (高波动缩仓)
        vol_scale = 1.0
        if VOL_TARGET and i >= VOL_LOOKBACK + BARS_PER_DAY:
            eqs = curve[i - VOL_LOOKBACK * BARS_PER_DAY:i] if i >= VOL_LOOKBACK * BARS_PER_DAY else curve[:i]
            if len(eqs) > BARS_PER_DAY:
                daily = eqs[::BARS_PER_DAY]
                rets = np.diff(daily) / daily[:-1]
                rv = np.std(rets) * 100 if len(rets) > 1 else 0
                if rv > 0:
                    vol_scale = min(VOL_CAP, VOL_TARGET / rv)   # 波动高于目标->缩, 低于->放(封顶VOL_CAP)

        # (3) BTC牛熊判断: 按 regime 分别缩放多/空腿敞口 (0 = 该腿在该状态不开新仓)
        if bull is None:
            long_scale = short_scale = 1.0          # 不做牛熊区分
        elif bull[i]:
            long_scale, short_scale = LONG_BULL, SHORT_BULL
        else:
            long_scale, short_scale = LONG_BEAR, SHORT_BEAR

        # ---- 入场: 多空信号分配给空槽 (按 universe 顺序先到先得, 两腿共享 N 个槽位) ----
        if not halted and (long_scale > 0 or short_scale > 0) and len(held) < N:
            for j in range(nS):
                if len(held) >= N:
                    break
                if j in held:
                    continue
                if cool[j] > 0 or (COOLDOWN_ONCE and done[j]):
                    continue                    # 冷却中 / 本次死叉窗口已做过
                sig, atr, c = Sv[i, j], Av[i, j], Cv[i, j]
                if sig == 0 or not (np.isfinite(atr) and atr > 0 and np.isfinite(c)):
                    continue
                scale = long_scale if sig == 1 else short_scale
                if scale <= 0:
                    continue
                # 逐币日线趋势过滤: 只在自己的日线 MA 向下且收盘在 MA 下方时允许做空
                if DTv is not None and sig == -1 and not DTv[i, j]:
                    continue
                # 慢线斜率过滤: ma_slow[MS_L1] vs ma_slow[MS_L2]
                if MSv is not None and sig == -1 and not MSv[i, j]:
                    continue
                # 波动放大过滤: 只在 ATR 高于近期均值时开仓
                if VEv is not None and sig == -1 and not VEv[i, j]:
                    continue
                entry = c * (1 + SLIP * sig)         # 入场滑点总是不利: 买贵/卖便宜
                # 该币被主策略做空时, 先平掉其填仓多单 (避免自我对冲, 与入场端 j not in held 对称)
                for f in fill_states:
                    if f['pos'] is not None and j == f['j']:
                        exit_px = Cv[i, j] * (1 + SLIP)      # 平多卖出, 市价叠滑点
                        pnl = f['pos']['notional'] * (exit_px - f['pos']['entry']) / f['pos']['entry']
                        net = pnl - f['pos']['notional'] * FEE
                        equity += net
                        trades.append(net)
                        _TRADE_TRACE.append((net / f['pos']['eq0'] if f['pos']['eq0'] > 0 else 0.0,
                                             f['pos']['bars'], f"{f['cfg']['sym'].lower()}_fill_hedge"))
                        f['pos'] = None
                # 出场参数按开仓时的牛熊状态固定 (与实盘一致: 挂单即定, 不随后续状态漂移)
                is_bull = bull is not None and bull[i]
                tp = float(TP_BULL) if (is_bull and TP_BULL) else CFG['ATR_TP']
                sl = float(SL_BULL) if (is_bull and SL_BULL) else CFG['ATR_SL']
                mb = int(BARS_BULL) if (is_bull and BARS_BULL) else MAX_BARS
                notional = equity * TOTAL_NOTIONAL / N * vol_scale * scale
                if SIZING == 'risk' and np.isfinite(ref_vol) and ref_vol > 0:
                    # 名义 ∝ 1/(ATR_SL·ATR/价格): 高波动币缩仓, 低波动币放仓, 每笔风险相等。
                    # 用实际 sl (牛市可能更近) 归一, 否则收窄止损会顺带缩小单笔风险,
                    # 那就混入了"降敞口"效应, 无法单独看出场位置的作用。
                    notional *= min(RISK_CAP, ref_vol / (atr / c) * CFG['ATR_SL'] / sl)
                equity -= notional * FEE
                held[j] = dict(dir=sig, entry=entry,
                               stop=entry - sig * sl * atr,
                               target=entry + sig * tp * atr,
                               notional=notional, bars=0, max_bars=mb,
                               atr=atr, ext=entry,   # ext: 持仓期最有利价, 初始=入场价
                               notional0=notional,   # 原始名义(分批比例的基数)
                               realized=0.0, pt_i=0,  # 已分批实现盈亏 / 下一个待触发档位
                               eq0=equity,           # 开仓时权益(把盈亏折成相对收益率用)
                               ma_exit=MA_EXIT and (is_bull or not MA_EXIT_BULL_ONLY))
        # ---- 长多填仓入场 (FILLS 注册制, 每标的独立窗口状态) ----
        # j not in held: 该币已被主策略做空时不开反向多单, 避免自我对冲。
        for f in fill_states:
            p0 = f['pos']
            if p0 is not None or f['j'] in held or halted \
                    or i < max(f['cfg']['ma_f'], f['cfg']['ma_s'], f['cfg']['atr_len']):
                continue
            # 窗口状态: 快线回落到慢线下方 => 金叉窗口结束, 解除"已做过"标记
            if f['cfg']['once'] and np.isfinite(f['ma_f'][i]) and np.isfinite(f['ma_s'][i]) \
                    and f['ma_f'][i] < f['ma_s'][i]:
                f['done'] = False
            ok_f = np.isfinite(Cv[i, f['j']]) and np.isfinite(f['ma_f'][i]) and np.isfinite(f['ma_s'][i])
            # 入场信号: ONCE=1 只认金叉当根 (gold); =0 用水平条件 (全程开多)
            enter = ok_f and ((f['gold'][i] if f['cfg']['once'] else f['ma_f'][i] > f['ma_s'][i]))
            if enter and not (f['cfg']['once'] and f['done']):
                used = sum(st['notional'] for st in held.values())
                total = equity * TOTAL_NOTIONAL
                idle = max(0.0, total - used)
                slot = equity * TOTAL_NOTIONAL / N
                notional = min(idle, slot)          # 不超过一个槽位的名义
                if notional > 1.0:
                    entry = Cv[i, f['j']] * (1 + SLIP)   # 做多入场滑点 = 买贵
                    equity -= notional * FEE
                    # 初始固定止损: 入场价 - 追踪间距 (宽松, 供未 arming 时用)
                    if f['cfg']['trail_atr'] > 0 and np.isfinite(f['atr'][i]):
                        stop0 = entry - f['cfg']['trail_atr'] * f['atr'][i]
                    elif f['cfg']['trail_pct'] > 0:
                        stop0 = entry * (1 - f['cfg']['trail_pct'])
                    else:
                        stop0 = 0
                    f['pos'] = dict(dir=1, entry=entry, notional=notional,
                                    bars=0, eq0=equity,
                                    ext=entry,           # 有利极值, 初始=入场价
                                    stop0=stop0)         # 初始固定止损(未arming时的底)
                    if f['cfg']['once']:
                        f['done'] = True           # 本窗口已做过, 等快线回落再重置
        # ---- 资金费: 每根K线对持仓按名义扣 FUNDING_BP (bp) ----
        if FUNDING_BP and held:
            equity -= sum(st['notional'] for st in held.values()) * FUNDING_BP / 1e4
        # ---- 盯市权益 ----
        mtm = equity
        for j, st in held.items():
            c = Cv[i, j]
            # realized: 分批止盈已落袋的部分。不加进盯市权益, 回撤/夏普会低估已实现收益。
            mtm += st.get('realized', 0.0)
            if np.isfinite(c):
                mtm += st['notional'] * st['dir'] * (c - st['entry']) / st['entry']
        for f in fill_states:
            if f['pos'] is not None:
                c = Cv[i, f['j']]
                if np.isfinite(c):
                    mtm += f['pos']['notional'] * (c - f['pos']['entry']) / f['pos']['entry']
        curve[i] = mtm
        if mtm > peak:                   # 更新峰值(盯市口径, 与回撤指标一致)
            peak = mtm
        held_sum += len(held)
        _HELD_TRACE[i] = len(held)          # 逐根持仓数, 供空仓期分析(见 analyze_idle.py)
    return curve, trades, held_sum / T


def metrics(curve, trades, idx):
    eq = pd.Series(curve, index=idx)
    final = curve[-1]
    days = (idx[-1] - idx[0]).days or 1
    ann = (final / INIT) ** (365 / days) - 1 if final > 0 else -1
    daily = eq.resample('1D').last().dropna()
    dr = daily.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = ((eq.cummax() - eq) / eq.cummax()).max()
    arr = np.array(trades)
    return dict(ret=(final - INIT) / INIT * 100, ann=ann * 100, sharpe=sharpe,
                mdd=mdd * 100, calmar=(ann * 100 / (mdd * 100)) if mdd > 0 else 0,
                trades=len(arr), win=(arr > 0).mean() * 100 if len(arr) else 0)


def main():
    pd.set_option('display.width', 200)
    idx, C, H, L, A, S, syms = build_panel()
    print('=' * 92)
    print(f'组合模拟: {len(syms)}币做空 4h, 总敞口固定={TOTAL_NOTIONAL:.0%}权益, 变持仓上限N')
    print(f'数据: {idx[0].date()} ~ {idx[-1].date()}  ({len(idx)}根)   单币名义 = 权益×{TOTAL_NOTIONAL:.0f}/N')
    print('=' * 92)
    rows = []
    for N in (3, 5, 8, 10, 15, 20, 30, 40, 61):
        curve, trades, avg_held = simulate(N, idx, C, H, L, A, S)
        m = metrics(curve, trades, idx)
        rows.append(dict(N=N, 单币名义=f'{TOTAL_NOTIONAL/N*100:.0f}%', 均持仓=round(avg_held, 1),
                         交易数=m['trades'], 总收益=round(m['ret'], 1), 年化=round(m['ann'], 1),
                         夏普=round(m['sharpe'], 2), 最大回撤=round(m['mdd'], 1),
                         Calmar=round(m['calmar'], 2), 胜率=round(m['win'], 1)))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
