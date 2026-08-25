"""MA5/MA24 死叉反弹做空 —— 实盘执行 (OKX 永续, 4h)。

策略来源: ma_pullback.py 做空腿, 已验证配置:
  周期4h / 做空 / CROSS_DELAY=6 / TOUCH_MODE=atr(0.5·ATR) / ATR止盈4·止损2
  全16币: 全样本t=2.99, 样本外t=3.61, 15/16币盈利 (ATR归一, 跨币种&跨时间可比)

执行模型 (与回测的关键差异):
  回测里ATR止盈/止损是"盘中触发"。实盘脚本不盯盘, 因此开空时
  在OKX同时挂"止盈止损委托(algo order)", 由交易所自动平仓。
  脚本只负责: 每根4h收盘后拉数据→算信号→无持仓且有做空信号则开空+挂TP/SL。

安全:
  - DRY_RUN 默认 True: 只算信号/打印意图, 绝不下真单。确认无误后手动改 False。
  - 凭证从环境变量读 (OKX_API_KEY / OKX_SECRET / OKX_PASSWORD), 不硬编码。
  - 本地持仓状态存 live_state.json, 每轮与交易所实际持仓核对, 不一致以交易所为准。
"""
import os, sys, json, time, io
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt
import ma_pullback as mp
import tg_notify as tg

def _utf8_stdout():
    # 仅在作为主程序运行时重设编码, 避免被 import 时重复包装 (会关闭共享 buffer)
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower().startswith('utf'):
            return
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

# ==================== .env 加载 (零依赖) ====================
def _load_dotenv(path=None):
    """把同目录 .env 读进 os.environ。不覆盖已存在的环境变量(命令行/外部传入优先)。
    docker-compose 用 env_file 已注入, 此处主要服务于「直接跑 python」的场景。"""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            v = v.strip()
            # 剥掉行尾注释 ( # 前需有空白, 以免误伤 SYMBOLS 之类值里的 # )。
            # 不剥的话 "0.027  # 注释" 会被当成数值 -> float() 直接抛错。
            if not (v.startswith('"') or v.startswith("'")):
                for i, ch in enumerate(v):
                    if ch == '#' and i > 0 and v[i - 1] in ' \t':
                        v = v[:i]
                        break
            k, v = k.strip(), v.strip().strip('"').strip("'")   # 去掉包裹的引号
            if k and k not in os.environ:      # 已存在则不覆盖
                os.environ[k] = v


_load_dotenv()


# ==================== 配置 (全部可由环境变量覆盖) ====================
def _env_bool(k, default):
    v = os.environ.get(k)
    return default if v is None else v.strip().lower() in ('1', 'true', 'yes', 'y', 'on')

def _env_float(k, default):
    v = os.environ.get(k)
    return default if v is None or v == '' else float(v)

def _env_int(k, default):
    v = os.environ.get(k)
    return default if v is None or v == '' else int(v)

def _env_str(k, default):
    v = os.environ.get(k)
    return default if v is None or v == '' else v

DRY_RUN = _env_bool('DRY_RUN', True)          # ★ True=只算信号不下单; 设 DRY_RUN=false 才会下真单
OKX_DEMO = _env_bool('OKX_DEMO', False)       # True=OKX模拟盘(demo trading), 用模拟盘专用key, 下单不花真钱
PROXY = _env_str('PROXY', '')                 # 空=直连 (容器内一般直连或用宿主代理, 见 docker-compose)
TF = _env_str('TF', '4h')
def _tf_ms(tf):
    """把 '4h'/'1d'/'15m' 之类周期转成毫秒 (供时间止损算持仓根数)。"""
    unit = tf[-1].lower()
    n = int(tf[:-1])
    mult = {'m': 60_000, 'h': 3_600_000, 'd': 86_400_000, 'w': 604_800_000}[unit]
    return n * mult
TF_MS = _tf_ms(TF)
LEVERAGE = _env_int('LEVERAGE', 3)            # 杠杆 (仅做空, 名义敞口由 MAX_NOTIONAL_PCT 控制)
POLL_SECONDS = _env_int('POLL_SECONDS', 60)  # 主循环每 N 秒检查一次是否到了新的收盘
# 心跳日志间隔(分钟)。本策略 86% 的K线无新开仓, 牛市最长可空仓23天(analyze_idle.py),
# 没有心跳就无法区分"策略本来就安静"和"进程已挂"。0=每轮都打。
HEARTBEAT_MIN = _env_int('HEARTBEAT_MIN', 30)
# ---- 网络重试 ----
# 拉K线/查持仓这类只读调用失败会让该币整轮跳过。虽然 last_bar 在 fetch 成功后才标记,
# 下一轮(POLL_SECONDS=60秒)会重试同一根, 但 4h 窗口内持续失败仍会丢掉信号, 且每轮刷一行
# error。指数退避重试把偶发抖动挡在本轮内。
# 每币之间的额外限速(ms)。ccxt 的 enableRateLimit 已按交易所规则排队, 这是额外保险。
# 原值300ms在49币下占轮次耗时15%, 直接推高入场延迟(参考价过期), 故降低。
PACE_MS = _env_int('PACE_MS', 50)
RETRY_TRIES = _env_int('RETRY_TRIES', 3)         # 总尝试次数(含首次)
RETRY_BASE_MS = _env_int('RETRY_BASE_MS', 500)   # 首次退避毫秒, 之后翻倍: 500/1000/2000
# ---- 入场新鲜度 ----
# 首次见到某币时, 若当前K线已开始超过此秒数, 不按"上一根收盘价"开新仓(参考价已过期)。
# 实测: 距收盘 13s/51s 的成交滑点 0.0/10.8bp, 而启动时补做已开始 3h48m 的 TIA 是 63.7bp
# —— 后者是延迟成本不是滑点, 且按过期价算出的止盈止损位也已失真。
# 900s=15分钟, 相当于 4h 周期的 6%。设 0 关闭该检查。
ENTRY_MAX_LAG_S = _env_int('ENTRY_MAX_LAG_S', 900)
# ---- Telegram 推送 (可选) ----
# 每轮把状态推到群, 便于离开电脑时确认策略还活着。凭据放 .env(已 gitignore), 不入代码。
# TG_CHAT: 群 id (形如 -1001234567890) 或 @频道名。获取方法: 把 bot 拉进群, 群里发一条
#   消息, 然后 python tg_notify.py --resolve
TG_TOKEN = _env_str('TG_TOKEN', '')
TG_CHAT = _env_str('TG_CHAT', '')
TG_MIN = _env_int('TG_MIN', 60)                # 无事件时的最小推送间隔(分钟), 0=每轮都推
DRY_EQUITY = _env_float('DRY_EQUITY', 10000)  # 干跑估算张数用的名义权益 (真单时以实际余额为准)
# ---- 组合总仓位控制 (多币同时出信号时的双重上限) ----
MAX_POSITIONS = _env_int('MAX_POSITIONS', 5)              # 同时最多持仓币数, 达上限不再开新仓
MAX_TOTAL_NOTIONAL_PCT = _env_float('MAX_TOTAL_NOTIONAL_PCT', 1.0)  # 总名义敞口上限(占 权益×杠杆 的比例)
# ---- BTC 趋势过滤: 牛市(BTC收盘>MA且MA上行)时做空敞口按此系数缩小 (验证: 减半样本外夏普1.71→1.79/回撤29→25%) ----
BTC_FILTER = _env_float('BTC_FILTER', 0.5)               # 牛市敞口系数; 1=关闭过滤, 0.5=减半, 0=牛市停做空
BTC_MA = _env_int('BTC_MA', 35)                          # BTC 判牛熊的日线均线天数 (敏感性: 20~40稳健)
# 状态文件: 容器内默认写 /data (挂卷持久化), 否则脚本同目录
STATE_FILE = _env_str('STATE_FILE', os.path.join(os.path.dirname(__file__), 'live_state.json'))

# 币种池: 市值>1亿的OKX USDT永续中, 做空腿验证通过者(样本外R>0且≥15笔)。
# 61币, 全样本t=5.79 / 样本外t=8.16 (原16币 3.47)。验证见 validate_universe.py / universe_pass.txt。
# 注: 原池的 DOT/TRX 因样本外做空R≤0 已剔除。
_DEFAULT_SYMS = ('BTC,ETH,BNB,XRP,SOL,DOGE,ADA,LINK,XLM,BCH,LTC,HBAR,SHIB,AVAX,SUI,UNI,NEAR,'
                 'TAO,AAVE,MORPHO,PEPE,ICP,ETC,POL,ALGO,ATOM,RENDER,FIL,ARB,APT,INJ,PENGU,TIA,'
                 'CRV,PYTH,ZRO,JTO,LDO,STX,BONK,IMX,XTZ,CFX,GRASS,OP,FLOKI,ENS,RAY,GRT,IOTA,'
                 'AXS,WIF,EIGEN,THETA,APE,CHZ,NEO,MANA,SNX,SAND,1INCH')
SYMBOLS = [s.strip().upper() for s in _env_str('SYMBOLS', _DEFAULT_SYMS).split(',') if s.strip()]

# 已验证的做空配置 (策略参数也可由环境变量覆盖, 便于调参而不改代码)
CFG = dict(mp.DEFAULTS, MA_TYPE='sma', SLOPE_FILTER=False,
           MA_FAST=_env_int('MA_FAST', 5),
           MA_SLOW=_env_int('MA_SLOW', 30),   # 慢线扫描(sweep_maslow.py): 26~32为稳健高原, 30居中最稳
           CROSS_DELAY=_env_int('CROSS_DELAY', 6),
           TOUCH_MODE=_env_str('TOUCH_MODE', 'atr'),
           TOUCH_ATR=_env_float('TOUCH_ATR', 0.5),
           TOUCH_PCT=_env_float('TOUCH_PCT', mp.DEFAULTS['TOUCH_PCT']),
           TOUCH_TH=_env_float('TOUCH_TH', mp.DEFAULTS['TOUCH_TH']),
           ENABLE_LONG=False, ENABLE_SHORT=True,
           EXIT_MODE='atr',
           ATR_TP=_env_float('ATR_TP', 4.0),
           ATR_SL=_env_float('ATR_SL', 2.0),
           MAX_BARS=_env_int('MAX_BARS', 40),   # 时间止损: 持仓超40根(6.7天)脚本主动平仓; 0=关
           MAX_NOTIONAL_PCT=_env_float('MAX_NOTIONAL_PCT', mp.DEFAULTS['MAX_NOTIONAL_PCT']))

# ==================== 配置 H (2023-24 样本外检验后确定, 详见 策略文档.md) ====================
# 三项改动, 均在 2023-24(BTC+463%) 与 2025-26(BTC-33%) 两段一致有效:
#   SIZING=risk     风险等分。等名义下单笔风险 = 名义·ATR_SL·(ATR/价格), 而 ATR/价格
#                   池内差 4.7 倍 => 高波动币主导盈亏。改为名义 ∝ 1/(ATR/价格) 后每笔
#                   风险相等, 牛市亏损 -27.1%->-19.0%, alpha 提纯。
#   MA_EXIT=1       收盘站上慢线立即平。做空前提是"被慢线压制", 站上即前提已破。
#                   单此一项让牛市段转正 (-19.0% -> +0.6%)。
#   COOLDOWN_ONCE=1 同一次死叉窗口只做一次。原逻辑平仓后下根可重开, 反复消耗同一结构;
#                   这些重复交易整体亏损。牛市 +0.6%->+4.5%, 交易数 -24%。
SIZING = _env_str('SIZING', 'notional')          # 'risk' = 风险等分; 'notional' = 等名义(旧)
RISK_REF_VOL = _env_float('RISK_REF_VOL', 0.027)  # 池内 ATR/价格 中位数(实测 2.69%), 风险等分基准
RISK_CAP = _env_float('RISK_CAP', 3.0)           # 单币名义上限(倍基准), 防低波动币开出天量仓位
MA_EXIT = _env_str('MA_EXIT', '0') == '1'        # 收盘价站上慢线则立即平仓
COOLDOWN_ONCE = _env_str('COOLDOWN_ONCE', '0') == '1'   # 同一次死叉窗口只开一次

# ---- 长多填仓 (同回测: BTC 10/120 close 3xATR20, ETH 20/85 high 4xATR18) ----
# 叠加层不占槽位, 用闲置容量做多。仅当主做空腿仓位不足时生效。
# 入场: 标的 4h MA(ma_f) > MA(ma_s) 金叉当根 (once=1 只做一次, 快线回落慢线下才解除);
#       src=high 时 MA 按最高价算(与单腿回测口径一致)。
# 出场: 追踪止损 — trail_atr×ATR(atr_len), stop = max(stop0, ext − trail·ATR_prev)。
# 开仓即挂 OKX algo 止损(入场−trail·ATR)兜底进程重启期间的裸奔风险。
FILLS_CFG = []

def _reg_fill_live(prefix, sym, ma_f, ma_s, once, trail_atr, atr_len, src):
    if _env_str(prefix, '0') != '1':
        return
    FILLS_CFG.append(dict(prefix=prefix, sym=sym,
                          ma_f=_env_int(prefix + '_MA_F', ma_f),
                          ma_s=_env_int(prefix + '_MA_S', ma_s),
                          once=_env_bool(prefix + '_ONCE', once),
                          trail_atr=_env_float(prefix + '_TRAIL_ATR', trail_atr),
                          atr_len=_env_int(prefix + '_ATR_LEN', atr_len),
                          src=_env_str(prefix + '_SRC', src)))

_reg_fill_live('BTC_FILL', 'BTC', 10, 120, True, 3.0, 20, 'close')
_reg_fill_live('ETH_FILL', 'ETH', 20, 85,  True, 4.0, 18, 'high')

# 单笔 fill 的名义上限 = 单币短空名义(等容量规则, 与回测 min(闲置,slot) 的 slot 对齐):
# 回测 slot = equity·TOTAL/N = 1.5/20 = 7.5% 权益, 实盘主空单币名义 = 0.025×3 = 7.5% 权益。
# 实际开仓取 min(单笔上限, 当时闲置容量)。
FILL_SLOT_NOTIONAL = _env_float('FILL_SLOT_NOTIONAL', CFG['MAX_NOTIONAL_PCT'] * LEVERAGE)

# ==================== 交易所 ====================
def make_exchange():
    key = os.environ.get('OKX_API_KEY')
    secret = os.environ.get('OKX_SECRET')
    pwd = os.environ.get('OKX_PASSWORD')
    # DRY_RUN 只需公开行情(拉K线), 无需凭证; 下真单前才强制校验
    if not DRY_RUN and not (key and secret and pwd):
        raise SystemExit('下真单需凭证: 请设置环境变量 OKX_API_KEY / OKX_SECRET / OKX_PASSWORD')
    conf = {
        'enableRateLimit': True, 'timeout': 15000,
        'options': {'defaultType': 'swap'},
    }
    if key and secret and pwd:
        conf.update(apiKey=key, secret=secret, password=pwd)
    if PROXY:
        # socks5h:// 走 socksProxy; http(s):// 走 httpProxy
        if PROXY.startswith('socks'):
            conf['socksProxy'] = PROXY
        else:
            conf['httpProxy'] = PROXY
    ex = ccxt.okx(conf)
    if OKX_DEMO:
        # 模拟盘: ccxt 给请求加 x-simulated-trading:1 头, 打到 OKX demo。
        # 注意凭证必须是「模拟盘」专门申请的 key, 与实盘 key 不通用。
        ex.set_sandbox_mode(True)
    # ccxt 4.x 的 set_sandbox_mode 会重置 session.proxies 为空 (实测验证) ——
    # 代理必须在它之后设置, 否则 sandbox 模式下代理静默失效, 直连解析 DNS 失败。
    if PROXY and not PROXY.startswith('socks'):
        ex.session.proxies = {'http': PROXY, 'https': PROXY}
    # OKX 的 fetch_currencies 是私有接口, 模拟盘/部分环境下不可用, 且策略不依赖 currencies,
    # 跳过以保 load_markets 不受阻。
    ex.fetch_currencies = lambda: {}
    # 启动时 load_markets 失败会直接崩溃 -> 容器 restart 循环。多给几次机会,
    # 且退避基数放大(启动期不着急, 网络刚起来时更可能需要等)。
    _retry(ex.load_markets, 'load_markets', tries=5, base_ms=2000)
    _build_symbol_map(ex)
    _detect_pos_mode(ex)
    return ex


# base 币 -> 真实 ccxt 永续符号 (load_markets 后建立, 避免硬编码 UNI/APT 之类映射不上)
_SYMBOL_MAP = {}
# 账户持仓模式: 'long_short_mode'(双向)需带 posSide; 'net_mode'(单向)不能带。None=未知/单向处理
_POS_MODE = None


def _build_symbol_map(ex):
    """从已加载的 markets 里解析每个 base 币的线性 USDT 永续符号。"""
    _SYMBOL_MAP.clear()
    for m in ex.markets.values():
        if (m.get('swap') and m.get('linear') and m.get('quote') == 'USDT'
                and m.get('settle') == 'USDT' and m.get('active', True)):
            _SYMBOL_MAP.setdefault(m['base'], m['symbol'])


def _detect_pos_mode(ex):
    """探测账户持仓模式。失败(无凭证/网络)则留 None, 下单按单向处理。"""
    global _POS_MODE
    if DRY_RUN or not ex.apiKey:
        return
    try:
        acc = ex.private_get_account_config()
        _POS_MODE = acc['data'][0].get('posMode')
        log(f'账户持仓模式 posMode={_POS_MODE}')
    except Exception as e:
        log(f'[warn] 探测持仓模式失败, 按单向(net)处理: {e}')


def _pos_side_params(side, pos='short'):
    """双向持仓模式下 OKX 要求带 posSide; 单向模式不能带。
    side: 'buy'/'sell' 为下单方向 (posSide 与交易方向无关, 由持仓方向指定)。
    pos: 'short'=空仓(主策略) / 'long'=多仓(长多填仓)。"""
    if _POS_MODE == 'long_short_mode':
        return {'posSide': pos}
    return {}


def inst(sym):
    """OKX 永续 ccxt 符号。优先用 load_markets 解析出的真实符号, 回退默认格式。"""
    return _SYMBOL_MAP.get(sym.upper(), f'{sym}/USDT:USDT')


# ==================== 状态持久化 ====================
def load_state():
    # positions:   sym->开仓信息; last_bar: sym->已处理的最后K线ts
    # done_window: sym->True 表示本次死叉窗口已做过一笔 (COOLDOWN_ONCE 用), 快线回到
    #              慢线上方即窗口结束并清除。必须持久化, 否则重启会重复入场。
    # pending:     sym->下单意图, 仅在"已下单但未落盘"的瞬间存在。重启后由
    #              sync_positions 按交易所实况接管或丢弃, 避免产生孤儿仓。
    # fills:       sym->长多填仓持仓 (含 sym_<x>_done 类窗口标记, 见 process_fills),
    #              与 positions 并存互不干扰。
    defaults = {'positions': {}, 'last_bar': {}, 'done_window': {}, 'pending': {}, 'fills': {}}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            st = json.load(f)
        # 旧状态文件缺 fills 等新增键: 合并默认值, 否则 process_fills/sync_positions 里
        # st['fills'] 直接 KeyError (实测: 集成 fill 前创建的状态文件没有 fills 键)。
        for k, v in defaults.items():
            st.setdefault(k, v)
        return st
    return defaults


def save_state(st):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def log(msg):
    print(f'{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}', flush=True)


# 成交记录: 预期价 vs 实际成交价, 供 analyze_slippage.py 校准回测滑点假设
FILLS_FILE = _env_str('FILLS_FILE', os.path.join(os.path.dirname(__file__), 'results', 'fills.csv'))


def log_fill(kind, sym, ref_px, fill_px, extra=''):
    """记录一笔成交滑点。kind: entry/tp/sl/time_stop/ma_exit。
    做空: 入场滑点=(参考-成交)/参考 (正=卖低了=不利); 出场(买回)滑点=(成交-参考)/参考 (正=买高了=不利)。

    同时记 lag_s = 成交时刻距所属4h K线开始的秒数。这一列是判读的前提:
      参考价是 K 线收盘价, lag 越大参考价越过期, 记出来的就越不是"滑点"而是"延迟成本"。
    实测(2026-08): DOGE lag=13s → 0.0bp, AVAX lag=51s → 10.8bp, TIA lag=6489s(启动时补做
    已开始的K线) → 63.7bp。两者混在一起会把滑点均值抬高一倍以上, 据此下调预期是错的。
    """
    if ref_px is None or fill_px is None or ref_px <= 0:
        return
    # 统一约定: 正 bp = 不利成交
    if kind == 'entry':
        slip_bp = (ref_px - fill_px) / ref_px * 1e4     # 卖出开空: 成交<参考(卖低了)=不利=正
    elif kind in ('fill_entry', 'fill_exit', 'fill_hedge'):
        # 填仓是多头, 方向与上面相反: 开多成交>参考(买高了)=不利; 平多成交<参考(卖低了)=不利
        slip_bp = ((fill_px - ref_px) if kind == 'fill_entry'
                   else (ref_px - fill_px)) / ref_px * 1e4
    else:
        slip_bp = (fill_px - ref_px) / ref_px * 1e4     # 买回平空: 成交>参考(买高了)=不利=正
    now = datetime.now(timezone.utc)
    tf_s = max(1, _tf_ms(TF) // 1000)
    lag_s = int(now.timestamp()) % tf_s                 # 距当前K线开始的秒数
    try:
        os.makedirs(os.path.dirname(FILLS_FILE), exist_ok=True)
        new = not os.path.exists(FILLS_FILE)
        with open(FILLS_FILE, 'a', encoding='utf-8') as f:
            if new:
                f.write('time,kind,sym,ref_px,fill_px,slip_bp,lag_s,extra\n')
            f.write(f'{now.isoformat()},{kind},{sym},'
                    f'{ref_px:.8g},{fill_px:.8g},{slip_bp:.2f},{lag_s},{extra}\n')
    except Exception as e:
        log(f'[warn] 写成交记录失败: {e}')


def order_avg_px(ex, order, im):
    """从下单回执取实际成交均价; 回执无则补查一次 fetch_order。失败返回 None。"""
    px = order.get('average') or order.get('price')
    if px:
        return float(px)
    try:
        oid = order.get('id')
        if oid:
            o2 = ex.fetch_order(oid, im)
            return float(o2.get('average') or o2.get('price') or 0) or None
    except Exception:
        pass
    return None


# ==================== 数据 ====================
def _retry(fn, what, tries=None, base_ms=None):
    """网络调用重试, 指数退避。全部失败则抛最后一个异常, 由调用方处理。

    只重试网络/限流类错误。参数错误、余额不足这类重试也没用, 直接抛出去 —— 重试它们
    只会推迟发现问题, 还可能在下单场景里造成重复提交。
    """
    tries = tries or RETRY_TRIES
    base = base_ms or RETRY_BASE_MS
    last = None
    for i in range(tries):
        try:
            return fn()
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable,
                ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
            last = e
            if i == tries - 1:
                break
            wait = base * (2 ** i)              # 500 / 1000 / 2000 ms ...
            log(f'[{what}][retry] {type(e).__name__}: {str(e)[:70]} — {wait}ms 后第{i+2}次尝试')
            time.sleep(wait / 1000.0)
    raise last


def fetch_ohlcv(ex, sym, limit=300):
    """拉最近 limit 根已收盘 4h K线, 返回 ma_pullback 期望的 DataFrame。"""
    o = _retry(lambda: ex.fetch_ohlcv(inst(sym), TF, limit=limit), sym)
    df = pd.DataFrame(o, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    # 丢掉当前未收盘的那根 (最后一根若时间戳属于进行中的周期)
    tf_ms = ex.parse_timeframe(TF) * 1000
    now_ms = ex.milliseconds()
    last_open_ms = int(o[-1][0])
    if last_open_ms + tf_ms > now_ms:      # 最后一根尚未收盘
        df = df.iloc[:-1]
    return df


# ==================== 信号 ====================
def latest_signal(df):
    """对最后一根已收盘K线求信号。返回 (sig, close, atr, ma_slow, ma_fast)。sig: -1 做空 / 0 无。"""
    di = mp.calc_indicators(df, CFG)
    ds = mp.gen_signals(di, CFG)
    warm = CFG['MA_SLOW'] + 2
    if len(ds) <= warm:
        return 0, None, None, None, None
    row = ds.iloc[-1]
    sig = int(row['entry_sig'])
    atr = float(row['atr']) if np.isfinite(row['atr']) else None
    mf = float(row['ma_fast']) if np.isfinite(row['ma_fast']) else None
    return sig, float(row['close']), atr, float(row['ma_slow']), mf


# ==================== 长多填仓: 指标 ====================
def fetch_fill_indicators(df, cfg):
    """对最后一根已收盘K线算填仓指标 (与回测 fill_states 口径一致)。

    返回 dict: close/high/low(最新根), ma_fast/ma_slow(最新根),
    atr_prev(上一根ATR, 出场用, 无前视), atr_cur(最新根ATR, 入场止损用),
    gold(本根金叉=本根 快线>慢线 且 上根 快线<=慢线)。
    src=high 时 MA 按最高价算(ETH 口径), 否则按收盘价(BTC 口径)。
    """
    src = cfg['src']
    ma_src = df['high'] if src == 'high' else df['close']
    ma_f = ma_src.rolling(cfg['ma_f']).mean()
    ma_s = ma_src.rolling(cfg['ma_s']).mean()
    gold = (ma_f > ma_s) & (ma_f.shift(1) <= ma_s.shift(1))
    prev_c = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - prev_c).abs(),
                    (df['low'] - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(cfg['atr_len']).mean()
    need = max(cfg['ma_s'], cfg['atr_len']) + 2
    if len(df) <= need:
        return None
    out = dict(close=float(df['close'].iloc[-1]),
               high=float(df['high'].iloc[-1]),
               low=float(df['low'].iloc[-1]),
               ma_fast=float(ma_f.iloc[-1]) if np.isfinite(ma_f.iloc[-1]) else None,
               ma_slow=float(ma_s.iloc[-1]) if np.isfinite(ma_s.iloc[-1]) else None,
               atr_prev=float(atr.iloc[-2]) if np.isfinite(atr.iloc[-2]) else None,
               atr_cur=float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else None,
               gold=bool(gold.iloc[-1]))
    return out


# ==================== 下单 ====================
def sync_positions(ex, st):
    """与交易所实际持仓核对: 交易所已无仓但本地有记录 => TP/SL已成交, 清本地。以交易所为准。"""
    if DRY_RUN:
        return                       # 干跑无真实持仓, 以本地记录为准即可
    try:
        _pos = [p for p in _retry(ex.fetch_positions, 'fetch_positions')
                if abs(float(p.get('contracts') or 0)) > 0]
        live = {p['symbol']: p for p in _pos}
    except Exception as e:
        log(f'[warn] 拉取持仓失败, 跳过本轮核对: {e}')
        return
    # 双向模式下同一 symbol 会返回 long+short 两条, dict 会互相覆盖 → 另建 (symbol,side) 全集
    live_side = {(p['symbol'], p.get('side')) for p in _pos}
    # fill 多单核对: 交易所已无 long 仓 => 初始 algo 止损已成交或手动平, 清本地
    for sym in list(st.setdefault('fills', {}).keys()):
        if sym.endswith('_done'):
            continue           # done 标记不是仓位
        im = inst(sym)
        if (im, 'long') in live_side:
            continue           # 多单还在
        if (im, 'short') in live_side:
            # 只可能是主策略开了空但对冲平多那一步失败, 或填仓被手动清掉后反手开空。
            # 交易所已无多单 => 按"多单已消失"清理, 防下一次金叉被 done 挡住。
            ff = st['fills'].pop(sym)
            st['fills'].pop(sym + '_done', None)
            log(f'[fill:{sym}] 本币现为交易所空单多头已消失, 清除填仓记录 '
                f'(开于 {ff.get("entry_time")}, 请复核对冲是否失败)')
            continue
        if im not in live:
            ff = st['fills'].pop(sym)
            st['fills'].pop(sym + '_done', None)  # once 窗口标记一并清, 允许下次金叉重入
            log(f'[fill:{sym}] 交易所已无多单 (初始止损已成交或手动平), 清除本地记录 '
                f'(开于 {ff.get("entry_time")})')
    for sym in list(st['positions'].keys()):
        if inst(sym) not in live:
            info = st['positions'].pop(sym)
            # 记录 TP/SL 实际成交滑点: 取最近一笔 buy(平空)成交价, 判定是止盈还是止损
            try:
                im = inst(sym)
                fills = ex.fetch_my_trades(im, limit=10)
                buys = [t for t in fills if t.get('side') == 'buy']
                if buys:
                    fpx = float(buys[-1]['price'])
                    tgt, stp = info.get('target'), info.get('stop')
                    kind = 'tp' if (tgt and abs(fpx - tgt) < abs(fpx - (stp or fpx))) else 'sl'
                    log_fill(kind, sym, tgt if kind == 'tp' else stp, fpx)
            except Exception as e:
                log(f'[{sym}][warn] 取平仓成交价失败(不影响清仓): {e}')
            log(f'[{sym}] 交易所已无持仓 (TP/SL 已平或手动平), 清除本地记录 (开于 {info.get("entry_time")})')

    # ---- 反向核对: 交易所有仓但本地无记录 = 孤儿仓 ----
    # 触发场景: 下单成功但 save_state 前进程被杀 / 状态文件损坏或被删 / 手动开的仓。
    # 这种仓不会被时间止损和 MA 出场管理(两者都遍历 st['positions']), 会一直挂着直到
    # 交易所的 TP/SL 触发; 若连 TP/SL 都没挂上, 就是一个无人看管的裸仓。
    # 只告警不自动接管: 无法可靠重建 entry/stop/target(fetch_positions 给的均价未必是
    # 我们的入场价), 猜错会导致错误的止损位。交由人工决定平掉还是补挂 TP/SL。
    # pending: 下单瞬间被杀留下的标记。交易所若真有仓, 说明单成交了 -> 接管为正式记录;
    # 若没仓, 说明单没成交 -> 丢弃标记。这样"下单成功但未落盘"不再变成孤儿仓。
    for sym in list(st.get('pending', {}).keys()):
        pd_ = st['pending'][sym]
        if sym.startswith('fill_'):
            real = sym[len('fill_'):]
            im = inst(real)
            if im in live and real not in st['fills'] and live[im].get('side') == 'long':
                p = live[im]
                amt = abs(float(p.get('contracts') or 0))
                entry_px = float(p.get('entryPrice') or 0) or None
                st['fills'][real] = dict(
                    sym=real, side='long', contracts=amt, ctv=contract_size(ex, real),
                    entry=entry_px, fill=entry_px, atr_entry=None, ext=entry_px,
                    stop0=None, bars=0, entry_time=pd_.get('ts'), recovered=True)
                log(f'[fill:{real}] 重启接管未落盘的多单: {amt}张 @{entry_px} (下单于 {pd_.get("ts")})。'
                    f'ext/stop0 未知 —— 脚本无法精确追踪, 请确认交易所侧初始止损委托是否已挂上')
            else:
                log(f'[fill:{real}] 丢弃 pending 标记 (交易所无对应多单, 该单未成交)')
            st['pending'].pop(sym, None)
            continue
        im = inst(sym)
        if im in live and sym not in st['positions']:
            p = live[im]
            amt = abs(float(p.get('contracts') or 0))
            entry_px = float(p.get('entryPrice') or 0) or None
            st['positions'][sym] = dict(
                sym=sym, side='short', contracts=amt, ctv=contract_size(ex, sym),
                entry=entry_px, fill=entry_px, stop=None, target=None,
                entry_time=pd_.get('ts'), recovered=True)
            log(f'[{sym}] 重启接管未落盘的持仓: {amt}张 @{entry_px} (下单于 {pd_.get("ts")})。'
                f'stop/target 未知 —— 时停与MA出场仍生效, 但请确认交易所侧 TP/SL 委托是否已挂上')
        else:
            log(f'[{sym}] 丢弃 pending 标记 (交易所无对应持仓, 该单未成交)')
        st['pending'].pop(sym, None)

    _known = ({inst(s) for s in st['positions']} | {inst(s) for s in st['fills']})
    for im, p in live.items():
        if im in _known:
            continue
        try:
            side = p.get('side') or '?'
            amt = abs(float(p.get('contracts') or 0))
            upl = p.get('unrealizedPnl')
            instId, algos = _fetch_algo_orders(ex, im)
            log(f'[warn][孤儿仓] {im} {side} {amt}张 未实现盈亏={upl} '
                f'附带TP/SL委托={len(algos)}个 — 本地无记录, 不受时停/MA出场管理。'
                f'{"请手动处理" if not algos else "有TP/SL兜底, 但仍需手动确认"}')
        except Exception as e:
            log(f'[warn][孤儿仓] {im} 本地无记录 (详情获取失败: {e})')
    save_state(st)


def _bars_held(entry_iso):
    """从开仓时间到现在经过了多少根 TF K线。"""
    try:
        t0 = datetime.fromisoformat(entry_iso)
    except Exception:
        return 0
    tf_hours = TF_MS / 3_600_000.0            # TF 每根的小时数
    elapsed_h = (datetime.now(timezone.utc) - t0).total_seconds() / 3600.0
    return int(elapsed_h // tf_hours) if tf_hours > 0 else 0


def _fetch_algo_orders(ex, im, only_stop=False):
    """查该合约挂起的 algo 委托(附带TP/SL是OCO类型)。用OKX原生端点:
    ccxt 统一接口的 ordType='conditional,oco' 会被拒(51000), 故直接调 private_get。
    only_stop=True 时只返回触发方向为 stop 的单 (平 fill 多单只需撤止损, 避免误撤
    同币主做空的 TP)。"""
    instId = ex.market(im)['id']
    out = []
    for ot in ('oco', 'conditional'):
        try:
            r = ex.private_get_trade_orders_algo_pending({'instId': instId, 'ordType': ot})
            out += r.get('data', [])
        except Exception as e:
            log(f'[warn] 查 {ot} algo 委托失败: {str(e)[:60]}')
    if only_stop and out:
        # algo data 里 slTriggerPx 存在的是止损; oco(TP/SL对)里两条都有 slTriggerPx 会一起撤,
        # 但平 fill 时主策略空单若在同一币, 误撤其 TP 会留裸空 —— 保留判断: 只撤有 slTriggerPx
        # 而 **没有 tpTriggerPx** 的单 (纯止损 = fill 的 algo), oco 对整单撤。
        out = [a for a in out if a.get('slTriggerPx') and not a.get('tpTriggerPx')]
    return instId, out


def _cancel_algo_orders(ex, im, instId, algos):
    """撤销 algo 委托。OKX 原生 cancel-algos, 传 [{algoId, instId}]。返回撤销成功数。"""
    ok = 0
    for a in algos:
        aid = a.get('algoId')
        if not aid:
            continue
        try:
            ex.private_post_trade_cancel_algos([{'algoId': aid, 'instId': instId}])
            ok += 1
        except Exception as e:
            log(f'[warn] 撤销 algo {aid} 失败: {str(e)[:60]}')
    return ok


def close_expired_positions(ex, st):
    """时间止损: 持仓超过 MAX_BARS 根的仓, 市价平仓并撤销残留TP/SL委托。"""
    max_bars = CFG.get('MAX_BARS', 0)
    if not max_bars:
        return
    for sym in list(st['positions'].keys()):
        info = st['positions'][sym]
        held = _bars_held(info.get('entry_time', ''))
        if held < max_bars:
            continue
        # 取最新收盘价作滑点参考价 —— 回测的时停也是按当根收盘平的。
        # 失败不影响平仓, 只是滑点记录退回开仓价(会记成盈亏而非滑点, 判读时剔除)。
        ref = None
        try:
            ref = float(fetch_ohlcv(ex, sym, limit=2)['close'].iloc[-1])
        except Exception:
            pass
        close_position(ex, st, sym, info, 'time_stop',
                       f'持仓{held}根≥{max_bars}', extra=f'held={held}bars', ref_px=ref)
    save_state(st)


def close_position(ex, st, sym, info, kind, why, extra='', ref_px=None):
    """市价平掉一个做空仓并撤销残留 TP/SL 委托。供时间止损 / MA出场 共用。

    ref_px: 滑点统计的参考价, 必须传**本次出场的触发价**(时停=当根收盘, MA出场=触发时收盘)。
      不传则退回开仓价, 那样记出来的不是滑点而是"这笔交易的盈亏" —— 2026-08-08 的
      AVAX ma_exit 记了 +174bp 就是这个原因(开空6.42平仓6.53, 亏1.74%被当成滑点),
      按触发价重算实际只有 +40bp。

    顺序很重要: 先平仓再撤委托。反过来若平仓失败, 仓位就会失去 TP/SL 保护而裸奔。
    """
    im = inst(sym)
    if DRY_RUN:
        log(f'[DRY_RUN][{sym}] {kind}: {why}, 应平仓 (开于 {info.get("entry_time")})')
        st['positions'].pop(sym, None)
        return
    try:
        o = ex.create_order(im, 'market', 'buy', info['contracts'],
                            params={'tdMode': 'isolated',
                                    'reduceOnly': (_POS_MODE != 'long_short_mode'),
                                    **_pos_side_params('buy')})
        avg = order_avg_px(ex, o, im)
        log_fill(kind, sym, ref_px or info.get('fill') or info.get('entry'), avg, extra=extra)
        log(f'[LIVE][{sym}] {kind}平仓 {info["contracts"]}张 @{avg or "?"} ({why})')
        # 撤销残留 TP/SL algo 委托, 避免孤儿单(平仓后SL若触发会反向开仓)
        instId, algos = _fetch_algo_orders(ex, im)
        n = _cancel_algo_orders(ex, im, instId, algos)
        if algos:
            log(f'[{sym}] 撤销残留TP/SL委托 {n}/{len(algos)} 个')
        st['positions'].pop(sym, None)
    except Exception as e:
        log(f'[{sym}][error] {kind}平仓失败: {e}')


def get_equity(ex):
    """取账户权益(USDT)。干跑用名义权益, 真单查余额。每轮取一次即可。"""
    if DRY_RUN:
        return DRY_EQUITY
    bal = _retry(ex.fetch_balance, 'fetch_balance')
    return float(bal['USDT']['total'])


def contract_size(ex, sym):
    """OKX 该合约的真实面值(每张对应的币数量)。干跑无 ex 时退回回测常量。
    注意: 各币面值差异极大(BTC 0.01 / SOL 1 / DOGE 1000), 绝不能一刀切。"""
    if ex is None:
        return mp.CONTRACT_VALUE
    return float(ex.market(inst(sym))['contractSize'])


_BTC_BULL = False        # 当前 BTC 是否牛市 (每轮 run_once 刷新)


def refresh_btc_regime(ex):
    """刷新 BTC 牛熊状态: 日线收盘 > BTC_MA 均线 且 均线上行(相对10日前) => 牛市。
    牛市下做空敞口按 BTC_FILTER 缩小。取上一根已收盘日线, 无未来函数。"""
    global _BTC_BULL
    if BTC_FILTER >= 1.0:
        _BTC_BULL = False          # 过滤关闭
        return
    try:
        o = ex.fetch_ohlcv('BTC/USDT:USDT', '1d', limit=BTC_MA + 20)
        closes = [c[4] for c in o[:-1]]        # 丢掉当前未收盘日线
        if len(closes) < BTC_MA + 11:
            _BTC_BULL = False
            return
        ma_now = sum(closes[-BTC_MA:]) / BTC_MA
        ma_prev = sum(closes[-BTC_MA - 10:-10]) / BTC_MA
        _BTC_BULL = (closes[-1] > ma_now) and (ma_now > ma_prev)
        log(f'BTC趋势: {"牛市→做空敞口×%.2g" % BTC_FILTER if _BTC_BULL else "非牛市→正常敞口"} '
            f'(close={closes[-1]:.0f} MA{BTC_MA}={ma_now:.0f})')
    except Exception as e:
        log(f'[warn] BTC趋势判定失败, 按非牛市处理: {e}')
        _BTC_BULL = False


def size_contracts(ex, sym, equity, close, atr=None):
    """单币张数 = 权益 * MAX_NOTIONAL_PCT * 杠杆 * 牛市系数 * 风险系数 / (价格 * 面值)。

    SIZING='risk' 时乘上风险等分系数 RISK_REF_VOL/(atr/close), 使每笔风险(名义×ATR_SL×
    ATR/价格)在各币间相等。基准用固定常数 RISK_REF_VOL 而非实时全池中位数 —— 回测里
    那是全样本中位数(含未来数据), 实盘必须用事先定好的值, 否则两边口径不一致。
    """
    btc_scale = BTC_FILTER if _BTC_BULL else 1.0
    notional = equity * CFG['MAX_NOTIONAL_PCT'] * LEVERAGE * btc_scale
    if SIZING == 'risk':
        if not (atr and atr > 0 and close > 0):
            return 0.0, contract_size(ex, sym)   # 风险等分必须有 ATR, 否则不开
        notional *= min(RISK_CAP, RISK_REF_VOL / (atr / close))
    ctv = contract_size(ex, sym)
    raw = notional / (close * ctv)
    # 满足交易所最小下单张数与步进
    min_amt, step = 1.0, 1.0
    if ex is not None:
        m = ex.market(inst(sym))
        min_amt = m['limits']['amount']['min'] or min_amt
        step = m['precision']['amount'] or step
    if ex is not None:
        # 用交易所精度规则规范化下单量, 避免浮点尾巴被拒单
        contracts = float(ex.amount_to_precision(inst(sym), raw))
    else:
        contracts = float(int(raw / step)) * step
    if contracts < min_amt:
        return 0.0, ctv                          # 敞口不够开最小一手 -> 放弃, 由调用方跳过
    return contracts, ctv


def pos_notional(info):
    """已持仓位的名义敞口(U) = 张数 * 开仓价 * 该仓真实面值。"""
    return info['contracts'] * info['entry'] * info.get('ctv', mp.CONTRACT_VALUE)


def open_short(ex, sym, close, atr, equity, contracts, ctv):
    """开空 + 挂 ATR 止盈/止损委托。返回开仓信息 dict。equity/contracts/ctv 由调用方预算好。"""
    entry = close * (1 - mp.SLIPPAGE)
    stop = entry + CFG['ATR_SL'] * atr        # 空头止损在上方
    target = entry - CFG['ATR_TP'] * atr      # 空头止盈在下方

    intent = dict(sym=sym, side='short', contracts=contracts, ctv=ctv, entry=round(entry, 6),
                  stop=round(stop, 6), target=round(target, 6),
                  entry_time=datetime.now(timezone.utc).isoformat())

    if DRY_RUN:
        log(f'[DRY_RUN][{sym}] 开空 {contracts}张 @~{entry:.4f}  TP={target:.4f} SL={stop:.4f} '
            f'(名义~{contracts*close*ctv:.0f}U, 保证金~{contracts*close*ctv/LEVERAGE:.0f}U)')
        return intent

    im = inst(sym)
    ex.set_leverage(LEVERAGE, im, params={'mgnMode': 'isolated', **_pos_side_params('sell')})
    # 开空 + 附带止盈止损(attachAlgoOrds): 一次下单让交易所把 TP/SL 绑到本仓,
    # 避免"开仓成功但挂单失败 → 裸空单无止损"。TP/SL 是 algo 委托, 不能当普通 market 单下(上一版 ordType 报错的根因)。
    params = {
        'tdMode': 'isolated',
        **_pos_side_params('sell'),
        'attachAlgoOrds': [{
            'tpTriggerPx': str(intent['target']), 'tpOrdPx': '-1',   # -1 = 市价止盈
            'slTriggerPx': str(intent['stop']),   'slOrdPx': '-1',   # -1 = 市价止损
            'tpTriggerPxType': 'last', 'slTriggerPxType': 'last',
        }],
    }
    o = ex.create_order(im, 'market', 'sell', contracts, params=params)
    avg = order_avg_px(ex, o, im)
    log_fill('entry', sym, close, avg, extra=f'contracts={contracts}')   # 参考=信号收盘价
    intent['fill'] = avg
    log(f'[LIVE][{sym}] 已开空 {contracts}张 @{avg or "~"+format(entry,".4f")}  '
        f'TP={target:.4f} SL={stop:.4f} (TP/SL已附带)')
    return intent


# ==================== 长多填仓: 下单 ====================
def amount_to_contracts(ex, sym, notional, close):
    """名义(U) -> 张数, 按交易所精度/最小单规范化。不足最小单返回 0。"""
    ctv = contract_size(ex, sym)
    raw = notional / (close * ctv)
    min_amt, step = 1.0, 1.0
    if ex is not None:
        m = ex.market(inst(sym))
        min_amt = m['limits']['amount']['min'] or min_amt
        step = m['precision']['amount'] or step
        contracts = float(ex.amount_to_precision(inst(sym), raw))
    else:
        contracts = float(int(raw / step)) * step
    if contracts < min_amt:
        return 0.0
    return contracts


def open_fill_long(ex, sym, close, atr, contracts, ctv, trail_atr):
    """开多一个长多填仓仓位 + 挂初始 ATR 止损委托(algo), 兜底进程崩溃期间裸奔。

    stop0 = 入场价 - trail_atr×ATR: 脚本侧追踪止损的底; 同时挂成交易所 algo 止损(固定值),
    脚本正常时由 process_fills 的追踪止损(upto ext)主动平, algo 只是最后一层保险。
    返回开仓信息 dict (side='long')。
    """
    entry = close * (1 + mp.SLIPPAGE)      # 做多买入, 成交价更高(不利)
    stop0 = entry - trail_atr * atr if (atr and atr > 0) else entry * 0.9
    intent = dict(sym=sym, side='long', contracts=contracts, ctv=ctv,
                  entry=round(entry, 6), fill=entry, atr_entry=atr,
                  ext=entry, stop0=round(stop0, 6), bars=0,
                  entry_time=datetime.now(timezone.utc).isoformat())

    if DRY_RUN:
        log(f'[DRY_RUN][fill:{sym}] 开多 {contracts}张 @~{entry:.4f}  '
            f'初始止损={stop0:.4f} (名义~{contracts*close*ctv:.0f}U, trail={trail_atr}xATR)')
        return intent

    im = inst(sym)
    ex.set_leverage(LEVERAGE, im, params={'mgnMode': 'isolated', **_pos_side_params('buy', 'long')})
    params = {
        'tdMode': 'isolated',
        **_pos_side_params('buy', 'long'),
        'attachAlgoOrds': [{
            'slTriggerPx': str(stop0), 'slOrdPx': '-1', 'slTriggerPxType': 'last',
        }],
    }
    o = ex.create_order(im, 'market', 'buy', contracts, params=params)
    avg = order_avg_px(ex, o, im)
    log_fill('fill_entry', sym, close, avg, extra=f'fill_long contracts={contracts}')
    intent['fill'] = avg
    log(f'[LIVE][fill:{sym}] 已开多 {contracts}张 @{avg or "~"+format(entry,".4f")}  '
        f'初始止损={stop0:.4f} (algo已挂)')
    return intent


def close_fill_long(ex, st, sym, info, why, ref_px=None):
    """市价平掉一个长多填仓并撤销残留 algo 委托。kind 用于 log_fill: fill_exit / fill_hedge。"""
    im = inst(sym)
    if DRY_RUN:
        log(f'[DRY_RUN][fill:{sym}] {why}, 应平多 (开于 {info.get("entry_time")})')
        st['fills'].pop(sym, None)
        return
    try:
        o = ex.create_order(im, 'market', 'sell', info['contracts'],
                            params={'tdMode': 'isolated',
                                    'reduceOnly': (_POS_MODE != 'long_short_mode'),
                                    **_pos_side_params('sell', 'long')})
        avg = order_avg_px(ex, o, im)
        kind = 'fill_hedge' if info.get('hedge') else 'fill_exit'
        log_fill(kind, sym, ref_px or info.get('fill') or info.get('entry'), avg,
                 extra=f'why={why}')
        log(f'[LIVE][fill:{sym}] 平多 {info["contracts"]}张 @{avg or "?"} ({why})')
        instId, algos = _fetch_algo_orders(ex, im, only_stop=True)
        n = _cancel_algo_orders(ex, im, instId, algos)
        if algos:
            log(f'[{sym}] 撤销残留止损委托 {n}/{len(algos)} 个')
        st['fills'].pop(sym, None)
    except Exception as e:
        log(f'[fill:{sym}][error] 平多失败: {e}')


def process_fills(ex, st, equity):
    """每轮在逐币主策略处理完之后, 统一管理长多填仓 (金叉入场 / 追踪止损出场)。

    与主做空腿互不干扰: fill 不占 N 槽位, 名义 = min(单币上限, 当时闲置容量),
    闲置容量 = equity·LEVERAGE·MAX_TOTAL_NOTIONAL_PCT − 已用总名义(主空仓 + 其他 fills)。
    入场条件: 金叉当根 (once=1 只做一次, 快线回落慢线下才解除)。
    出场条件: 最新已收盘 low ≤ stop, stop = max(stop0, ext − trail_atr·ATR_prev)。
    """
    used = sum(pos_notional(i) for i in st['positions'].values())
    # fills 里混着位置 dict 和 _done 布尔标记, 只对位置求名义
    used += sum(pos_notional(i) for i in st.setdefault('fills', {}).values() if isinstance(i, dict))
    cap = equity * LEVERAGE * MAX_TOTAL_NOTIONAL_PCT
    idle = max(0.0, cap - used)
    slot = equity * FILL_SLOT_NOTIONAL
    fills = st.setdefault('fills', {})
    for f in FILLS_CFG:
        sym = f['sym']
        if inst(sym) not in ex.markets:
            continue
        try:
            df = fetch_ohlcv(ex, sym)
        except Exception as e:
            log(f'[fill:{sym}][error] 拉K线失败: {e}')
            continue
        ind = fetch_fill_indicators(df, f)
        if ind is None:
            continue
        fpos = fills.get(sym)
        # ---- 出场: 追踪止损 (无前视: stop 用上一根为止的 ext/ATR, 见回测出场节) ----
        if fpos is not None:
            fpos['bars'] += 1
            gap = f['trail_atr'] * ind['atr_prev'] if (f['trail_atr'] > 0 and ind['atr_prev']) else 0
            # 顺序关键: 先判本根 low 是否触发(用上一根的 ext), 不触发才把本根 high 并入 ext。
            # 反了 = 本根极值决定本根止损位, 前视 (回测注释: "本根高点要等出场判定之后才更新").
            # base 用 0 兜底: 重启接管的 fill stop0=None, 交易所侧仍有 algo 止损兜底, 此处只做宽松上限。
            stop = max(fpos['stop0'] or 0, fpos['ext'] - gap)
            if ind['low'] <= stop:
                close_fill_long(ex, st, sym, fpos, f'trail(停={stop:.4f},低={ind["low"]:.4f})',
                                ref_px=stop)
            else:
                fpos['ext'] = max(fpos['ext'], ind['high'])
            continue                                      # 持仓状态本根不再入场
        # ---- 窗口状态: 快线回落到慢线下方 => 金叉窗口结束, 解除"已做过"标记 ----
        if f['once'] and ind['ma_fast'] and ind['ma_slow'] and ind['ma_fast'] < ind['ma_slow']:
            if fills.pop(sym + '_done', None):
                log(f'[fill:{sym}] 金叉窗口结束 (快线{ind["ma_fast"]:.6g}<慢线{ind["ma_slow"]:.6g}), 允许下次金叉')
        if f['once'] and fills.get(sym + '_done'):
            continue                                      # 本次窗口已做过
        # 主策略已在做空该币 => 不开反向多单(避免自我对冲, 与回测 j not in held 对称)
        if sym in st['positions']:
            continue
        # ---- 入场: 金叉当根 ----
        if not (ind['gold'] and ind['close'] and ind['ma_fast'] and ind['ma_slow']
                and ind['ma_fast'] > ind['ma_slow']):
            continue
        notional = min(slot, idle)
        if notional <= 0:
            log(f'[fill:{sym}] 金叉但无闲置容量 (已用{used:.0f} ≥ 上限{cap:.0f}U), 跳过')
            continue
        contracts = amount_to_contracts(ex, sym, notional, ind['close'])
        if contracts <= 0:
            log(f'[fill:{sym}] 金叉但名义{notional:.0f}U 不足开最小单, 跳过')
            continue
        log(f'[fill:{sym}] 金叉当根 close={ind["close"]:.4f} 快线{ind["ma_fast"]:.6g}>'
            f'慢线{ind["ma_slow"]:.6g}, 开多 {contracts}张 (名义~{contracts*ind["close"]*contract_size(ex,sym):.0f}U)')
        # 下单前落 pending 标记: 重启后 sync_positions 凭它接管, 不形成孤儿仓
        st.setdefault('pending', {})['fill_' + sym] = dict(bar=df.index[-1].isoformat(),
                                                           ts=datetime.now(timezone.utc).isoformat())
        save_state(st)
        try:
            info = open_fill_long(ex, sym, ind['close'], ind['atr_cur'],
                                  contracts, contract_size(ex, sym), f['trail_atr'])
        except Exception:
            st.get('pending', {}).pop('fill_' + sym, None)
            save_state(st)
            raise
        if f['once']:
            fills[sym + '_done'] = True                   # 本窗口已做过, 等快线回落再重置
        fills[sym] = info
        st.get('pending', {}).pop('fill_' + sym, None)
        save_state(st)


# ==================== 主循环 ====================
def process_symbol(ex, st, sym, equity):
    # 预检: 解析出的符号不在交易所 markets 里(如 UNI/APT 在本环境/模拟盘不可交易)则跳过, 不刷错误。
    # 直接问 ex.markets (fetch_ohlcv 的权威依据), 而非派生的 _SYMBOL_MAP。DRY_RUN 也要拉K线, 故不按模式豁免。
    if inst(sym) not in ex.markets:
        if not st.setdefault('skipped', {}).get(sym):
            log(f'[{sym}] 交易所无对应 USDT 永续合约 (解析={inst(sym)}), 本会话跳过该币')
            st['skipped'][sym] = True
        return
    df = fetch_ohlcv(ex, sym)
    if len(df) < CFG['MA_SLOW'] + 5:
        return
    bar_ts = df.index[-1].isoformat()
    first_seen = sym not in st['last_bar']       # 本会话/重启后第一次见到这个币
    if st['last_bar'].get(sym) == bar_ts:
        return                                   # 这根K线已处理过
    st['last_bar'][sym] = bar_ts

    # 新鲜度闸: 首次见到该币时, 若当前K线已开始很久, 不按"上一根收盘价"开新仓。
    # 起因(2026-08-07): 脚本在 15:48 启动, TIA 用 12:00 收盘的价开空, 参考价过期 3h48m,
    # 记出 63.7bp "滑点" —— 实为延迟成本。同期正常轮次(距收盘13s/51s)只有 0.0/10.8bp。
    # 只拦首次: 常规轮次是收盘后几十秒内触发, 不受影响; 已持仓的出场检查也不拦
    # (出场用当根实时收盘价, 不依赖参考价新鲜度, 且延迟平仓比不平仓好)。
    stale = None
    if first_seen and ENTRY_MAX_LAG_S:
        lag = int(time.time()) % max(1, TF_MS // 1000)
        if lag > ENTRY_MAX_LAG_S:
            stale = lag

    sig, close, atr, ma_slow, ma_fast = latest_signal(df)
    # 死叉窗口维护: 快线回到慢线上方 = 空头窗口结束, 允许下次死叉重新做一笔。
    # 打日志是因为这里有个实盘特有的漏判风险: 回测逐根遍历必然抓到上穿, 而实盘每根收盘
    # 才扫一次 —— 若上穿只持续不到一根K线, 标记就不会被清除, 该币会长期不参与。
    # (2026-08-10 XTZ 在回测里有信号而实盘没开, 排查时发现无法确认重置是否正常发生。)
    if COOLDOWN_ONCE and ma_fast and ma_slow and ma_fast >= ma_slow:
        if st.setdefault('done_window', {}).pop(sym, None):
            log(f'[{sym}] 死叉窗口结束 (快线{ma_fast:.6g}≥慢线{ma_slow:.6g}), '
                f'解除同窗口限制, 可再次入场')
    if sym in st['positions']:
        # MA出场: 做空的前提是"价格被慢线压制"; 收盘站上慢线 = 前提已破, 立即平。
        # 比等 ATR 止损(2·ATR)更早离场, 回测显示这是单项改善最大的一条。
        if MA_EXIT and ma_slow and close and close > ma_slow:
            # ref_px 传触发时的收盘价: 回测就是按这个价平仓的, 两边口径才可比
            close_position(ex, st, sym, st['positions'][sym], 'ma_exit',
                           f'收盘{close:.6g}>慢线{ma_slow:.6g}',
                           extra=f'ma_slow={ma_slow:.6g}', ref_px=close)
            save_state(st)
        return                                   # 已有持仓, 其余出场交给交易所的TP/SL
    if not (sig == -1 and atr and atr > 0):
        return
    if stale:
        # 放在信号判定之后: 只有真有信号时才需要提示, 否则启动首轮会刷 49 行无用日志
        log(f'[{sym}] 有做空信号但当前K线已开始 {stale//60} 分钟 '
            f'(>{ENTRY_MAX_LAG_S//60}分), 参考价已过期, 跳过本根; 下根收盘正常参与')
        return
    if COOLDOWN_ONCE and st.setdefault('done_window', {}).get(sym):
        # 本次死叉窗口已做过。回测显示同一窗口内反复入场整体亏损(重复交易吃手续费+滑点,
        # 而首次反弹触碰的信息量最高), 故一个窗口只做一笔。
        # 打日志: 这是回测与实盘产生差异最隐蔽的一处 —— 回测同窗口也只做一次, 但两边对
        # "窗口何时结束"的判定精度不同(见上方注释), 差异只能靠日志对账。
        log(f'[{sym}] 有做空信号但本次死叉窗口已做过一笔, 跳过 '
            f'(快线{ma_fast:.6g} vs 慢线{ma_slow:.6g}, 需快线上穿才解除)')
        return

    # ---- 组合总仓位控制: 三道闸, 任一不满足则跳过本次开仓 ----
    if len(st['positions']) >= MAX_POSITIONS:
        log(f'[{sym}] 有做空信号但已达最大持仓数 {MAX_POSITIONS}, 跳过')
        return
    contracts, ctv = size_contracts(ex, sym, equity, close, atr)
    if contracts <= 0:                           # 敞口不够开最小一手
        log(f'[{sym}] 有做空信号但资金不足以开最小一手 (单币敞口 '
            f'{equity*CFG["MAX_NOTIONAL_PCT"]*LEVERAGE:.0f}U 太小), 跳过')
        return
    used = sum(pos_notional(i) for i in st['positions'].values())
    add = contracts * close * ctv
    cap = equity * LEVERAGE * MAX_TOTAL_NOTIONAL_PCT
    if used + add > cap:
        log(f'[{sym}] 有做空信号但总名义敞口将超上限 '
            f'(已用{used:.0f}+新增{add:.0f} > {cap:.0f}U), 跳过')
        return

    log(f'[{sym}] 4h收盘 {bar_ts}: 做空信号 close={close:.4f} atr={atr:.4f}')
    # 下单前先落一条 pending 标记: 若在"下单已成交"与"save_state"之间进程被杀,
    # 重启后 sync_positions 能凭它认出这个仓是自己的, 而不是报成孤儿仓。
    # 代价是可能留下已下单失败的 pending, 由下面的清理逻辑按交易所实况纠正。
    st.setdefault('pending', {})[sym] = dict(
        bar=bar_ts, contracts=contracts,
        ts=datetime.now(timezone.utc).isoformat())
    save_state(st)
    try:
        info = open_short(ex, sym, close, atr, equity, contracts, ctv)
    except Exception:
        st.get('pending', {}).pop(sym, None)     # 下单失败, 撤掉标记
        save_state(st)
        raise
    st['positions'][sym] = info
    st.get('pending', {}).pop(sym, None)
    if COOLDOWN_ONCE:
        # 开仓即标记本窗口已用掉 (而非平仓时标记): 若等到平仓再置, 期间重启会丢状态。
        st.setdefault('done_window', {})[sym] = True
    # 对冲: 主策略新开空仓后, 平掉同一币的长多填仓 (避免自我对冲, 与回测 hedge 对称)。
    # 放在开空成功之后执行, 且本轮 process_fills 会跳过该 sym (它已进 st['positions'])。
    # 用 get 不先 pop: 平仓成功才删本地记录(close_fill_long 内部 pop)。若平仓失败,
    # 记录保留, 下轮 process_fills 仍能管理该多单, 不会平仓失败时把 fill 变成孤儿仓。
    ffill = st.get('fills', {}).get(sym)
    if ffill is not None:
        log(f'[fill:{sym}] 主策略开空本币, 触发对冲平多')
        close_fill_long(ex, st, sym, dict(ffill, hedge=True), '主策略开空对冲',
                        ref_px=close)
    save_state(st)
    return True                                  # 供 run_once 统计本轮新开仓数


_ROUND = 0               # 已跑轮数
_START_TS = time.time()  # 进程启动时刻, 用于运行时长
_LAST_HB = 0.0           # 上次输出心跳日志的时刻
_LAST_TG = 0.0           # 上次 TG 推送的时刻


def _fmt_dur(sec):
    """秒 -> '3天4小时' / '5小时12分' / '43分'。"""
    d, r = divmod(int(sec), 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    if d:
        return f'{d}天{h}小时'
    if h:
        return f'{h}小时{m}分'
    return f'{m}分'


def run_once(ex, st):
    """跑一轮: 刷新BTC趋势, 核对持仓, 时间止损平超时仓, 最后逐币种处理开仓。"""
    global _ROUND
    _ROUND += 1
    refresh_btc_regime(ex)                       # 刷新牛熊状态, 牛市缩小做空敞口
    sync_positions(ex, st)
    close_expired_positions(ex, st)              # 时间止损: 平掉持仓超 MAX_BARS 根的仓
    equity = get_equity(ex)                      # 每轮取一次权益, 组合仓位控制共用
    n_sig = n_err = 0
    # 持仓币优先: 它们要做 MA出场判断, 晚一轮可能多亏一根K线的幅度; 无仓币只是等入场机会。
    # 其余按固定顺序(SYMBOLS)—— 不随机化, 否则同一根K线内多次运行结果不可复现。
    order = ([s for s in SYMBOLS if s in st['positions']]
             + [s for s in SYMBOLS if s not in st['positions']])
    # 记录处理前的 last_bar 快照: 轮末与之比对, 就能知道哪些币这轮真的推进到了新K线。
    # 用快照而非在 process_symbol 里返回状态 —— 后者有十几个裸 return 分支, 改造易漏。
    _before = dict(st.get('last_bar', {}))
    for sym in order:
        try:
            if process_symbol(ex, st, sym, equity):
                n_sig += 1
        except Exception as e:
            n_err += 1
            log(f'[{sym}][error] {e}')
        # 限速: ccxt 的 enableRateLimit 已按交易所规则排队, 这里只是额外保险。
        # 原值 300ms × 49币 = 14.7s, 占轮次耗时的 15% —— 降到 50ms 仍有余量。
        ex.sleep(PACE_MS)

    # ---- K线推进对账 ----
    # 本轮有币推进到新K线时, 报告推进数与遗漏名单。遗漏意味着该币这根的信号被永久错过
    # (下一轮 last_bar 虽仍会重试, 但那时已是新的一根, 旧信号不再有效)。
    # 2026-08-10 XTZ 在回测有信号而实盘没开, 事后无法判断是"没扫到"还是"被闸拦下" ——
    # 有这行就能直接区分: 名单里有 XTZ = 没扫到; 没有 = 扫到了但被某道闸拦(那会另有日志)。
    adv = [s for s in SYMBOLS if st['last_bar'].get(s) != _before.get(s)]
    if adv:
        miss = [s for s in SYMBOLS if s not in adv and s not in st.get('skipped', {})]
        log(f'[K线] 本轮推进 {len(adv)}/{len(SYMBOLS)} 币到新K线'
            + (f'  ⚠遗漏{len(miss)}个: {",".join(miss[:12])}'
               + (' …' if len(miss) > 12 else '') if miss else ''))

    # ---- 长多填仓: 逐 K 线出场/入场 (须在主策略之后: 主策略先做空对冲 fill) ----
    try:
        process_fills(ex, st, equity)
    except Exception as e:
        n_err += 1
        log(f'[fills][error] {e}')

    # ---- 心跳: 每轮固定输出一行, 证明程序活着 ----
    # 回测统计(analyze_idle.py): 86% 的K线没有新开仓, 牛市里最长可能连续 23 天完全空仓。
    # 没有这行日志时, "很久没动静"和"程序已挂"在日志上完全一样, 容易误判而手动干预。
    pos = st['positions']
    fpos = {k: v for k, v in st.get('fills', {}).items() if not k.endswith('_done')}
    used = sum(pos_notional(i) for i in pos.values())
    used += sum(pos_notional(i) for i in fpos.values())
    fill_txt = ''
    if fpos:
        fill_txt = (' 填仓[' +
                    ' '.join(f'{s}({_bars_held(i.get("entry_time",""))}根)'
                             for s, i in list(fpos.items())[:4]) +
                    (f' …{len(fpos)-4}' if len(fpos) > 4 else '') + ']')
    if pos:
        held_txt = (f'{len(pos)}/{MAX_POSITIONS}仓 [' +
                    ' '.join(f'{s}({_bars_held(i.get("entry_time",""))}根)'
                             for s, i in list(pos.items())[:8]) +
                    (' …' if len(pos) > 8 else '') + ']' +
                    f' 名义{used:.0f}U({used/equity*100:.0f}%权益)' + fill_txt)
    else:
        held_txt = f'0/{MAX_POSITIONS}仓 空仓' + fill_txt
    msg = (f'[心跳] 第{_ROUND}轮 运行{_fmt_dur(time.time()-_START_TS)}  权益{equity:.0f}U  '
           f'{held_txt}  本轮新开{n_sig}  牛市={_BTC_BULL}' + (f'  错误{n_err}' if n_err else ''))
    # 节流: POLL_SECONDS=60 时每轮都打会刷出一天1440行, 淹没真正的交易日志。
    # 有事件(开仓/出错)时立即打, 否则每 HEARTBEAT_MIN 分钟至多一行。
    global _LAST_HB, _LAST_TG
    now = time.time()
    if n_sig or n_err or now - _LAST_HB >= HEARTBEAT_MIN * 60:
        _LAST_HB = now
        log(msg)

    # ---- Telegram 推送: 有事件立即推, 否则按 TG_MIN 节流 ----
    if TG_TOKEN and TG_CHAT and (n_sig or n_err or now - _LAST_TG >= TG_MIN * 60):
        _LAST_TG = now
        icon = '🔴' if n_err else ('🟢' if n_sig else '⚪')
        mode = 'DRY' if DRY_RUN else ('DEMO' if OKX_DEMO else 'LIVE')
        lines = [f'{icon} <b>MA{CFG["MA_SLOW"]}做空</b> [{mode}] 第{_ROUND}轮 '
                 f'· 运行{_fmt_dur(now - _START_TS)}',
                 f'权益 <b>{equity:.2f}</b> U · 牛市={_BTC_BULL}']
        if pos:
            lines.append(f'持仓 {len(pos)}/{MAX_POSITIONS} · 名义 {used:.0f}U '
                         f'({used/equity*100:.0f}%权益)')
            for s, i in list(pos.items())[:12]:
                b = _bars_held(i.get('entry_time', ''))
                lines.append(f'  {s} {i.get("contracts")}张 @{i.get("entry")} '
                             f'· {b}/{CFG["MAX_BARS"]}根')
            if len(pos) > 12:
                lines.append(f'  … 另 {len(pos)-12} 个')
        else:
            lines.append(f'持仓 0/{MAX_POSITIONS} · <i>空仓</i>')
        if fpos:
            lines.append(f'填仓 {len(fpos)} · ' +
                         ' '.join(f'{s} {i.get("contracts")}张@{i.get("entry")}'
                                  for s, i in list(fpos.items())[:4]))
        if n_sig:
            lines.append(f'本轮新开 <b>{n_sig}</b> 仓')
        if n_err:
            lines.append(f'⚠️ 本轮错误 <b>{n_err}</b> 个, 详见日志')
        tg.send('\n'.join(lines), log=log)


def _next_wait():
    """算到下一次该唤醒的秒数, 对齐 K 线边界。

    原来是 run_once 之后固定 sleep(POLL_SECONDS), 于是实际周期 = 轮次耗时 + POLL。
    实测 49 币一轮约 99s(每币 fetch ~1.7s + ex.sleep 300ms), 所以周期是 159s 而非 60s,
    且与 4h 边界无关 —— 收盘后要等 0~159s 才被扫到, 加上该币在池中的排队位置(0~99s),
    总延迟最坏 258s。参考价(上一根收盘价)越晚用越过期, 且币池后段的币系统性更晚入场。

    改为: 收盘后立刻醒一次做当根信号, 其余时间按 POLL_SECONDS 粗轮询(仍需要它来做
    时间止损/MA出场这类不依赖新K线的检查)。lead 提前量留 2s, 避免因时钟误差错过边界。
    """
    tf_s = max(1, TF_MS // 1000)
    now = time.time()
    into = now % tf_s                       # 距当前K线开始已过去多少秒
    to_close = tf_s - into                  # 距下一根K线开始(=当根收盘)还有多少秒
    lead = 2.0
    if to_close <= lead:                    # 已经贴着边界, 立刻再跑一轮
        return max(0.5, to_close)
    return min(POLL_SECONDS, to_close - lead)


def _selfcheck():
    """启动自检: 把"参数看着对但实际跑错"这类问题在第一行日志就暴露出来。

    起因(2026-08-07): docker-compose 的 environment 用 "${VAR:-default}" 从宿主 shell 取值,
    覆盖了 env_file, 导致 .env 已是 MA90/N20/MA出场 而容器实际跑 MA30/N36/无MA出场。
    参数本身都是合法值, 不会报错, 只能靠比对才发现 —— 所以这里做一致性校验而非合法性校验。
    """
    warns = []
    # 1. 两个敞口上限必须自洽: N × 单币名义 应等于 总敞口上限, 否则先撞到哪个闸不明确
    per = CFG['MAX_NOTIONAL_PCT'] * LEVERAGE
    tot = MAX_TOTAL_NOTIONAL_PCT * LEVERAGE
    if abs(MAX_POSITIONS * per - tot) > 0.02:
        warns.append(f'敞口口径不自洽: {MAX_POSITIONS}仓×{per*100:.1f}% = '
                     f'{MAX_POSITIONS*per*100:.0f}% ≠ 总上限{tot*100:.0f}% '
                     f'(满仓前会先撞总敞口闸, 实际持仓数达不到 N)')
    # 2. 定案配置(见 .env 注释)的关键开关, 缺一项收益差一截 —— 静默跑错代价太大
    if not MA_EXIT:
        warns.append('MA_EXIT=0: 回测中这一项单独贡献 牛市段 -19.0%→+0.6%, 确认是有意关闭?')
    if not COOLDOWN_ONCE:
        warns.append('COOLDOWN_ONCE=0: 同一死叉窗口会反复入场, 回测显示这些重复交易整体亏损')
    if CFG['MA_SLOW'] < 50:
        warns.append(f'MA_SLOW={CFG["MA_SLOW"]}: 短慢线在2023-24牛市信号层t值仅0.38(不显著), '
                     f'定案值为90 (85~90为高原)')
    if BTC_FILTER < 1.0:
        warns.append(f'BTC_FILTER={BTC_FILTER}: MA90下牛熊过滤是纯损害'
                     f'(alpha +37%→+24%, 合成年化 +34.5%→+28.8%), 定案为1(关闭)')
    # 3. 预热: MA_SLOW 越长需要越多K线, 拉少了会一直算不出信号
    need = CFG['MA_SLOW'] + 5
    if need > 300:
        warns.append(f'预热需{need}根但 fetch_ohlcv 只拉300根, 信号永远算不出来')
    # 4. 状态文件所在目录必须可写, 否则重启会丢持仓记录
    d = os.path.dirname(os.path.abspath(STATE_FILE)) or '.'
    if not os.access(d, os.W_OK):
        warns.append(f'状态目录不可写: {d} (重启将丢失持仓记录 -> 已有仓会变成孤儿仓)')
    # 5. 长多填仓: 标的必须在币池且单币名义上限不得超过总敞口上限
    for f in FILLS_CFG:
        if f['sym'] not in SYMBOLS:
            warns.append(f'{f["prefix"]}: 标的 {f["sym"]} 不在当前币池 '
                         f'(金叉信号照算, 但主策略不会对它开空, 对冲逻辑退化为纯填仓)')
    if FILLS_CFG and FILL_SLOT_NOTIONAL > tot:
        warns.append(f'FILL_SLOT_NOTIONAL={FILL_SLOT_NOTIONAL*100:.0f}% '
                     f'> 总敞口上限{tot*100:.0f}% (单币填仓名义从未被闲置容量截断, 该值无效)')
    for w in warns:
        log(f'[自检][warn] {w}')
    log(f'[自检] {"发现 %d 项需确认" % len(warns) if warns else "配置一致性 OK"}  '
        f'状态文件={STATE_FILE}')
    return warns


def main():
    if DRY_RUN:
        mode = 'DRY_RUN(不下真单)'
    elif OKX_DEMO:
        mode = '模拟盘下单(demo, 不花真钱)'
    else:
        mode = '★ LIVE 真实下单 ★'
    log(f'启动 MA做空实盘 [{mode}]  周期={TF} 币种={len(SYMBOLS)} 杠杆={LEVERAGE}x')
    # 把生效参数完整打一遍: 排查问题时最先要确认的就是"跑的到底是哪套配置"
    log(f'配置 MA{CFG["MA_FAST"]}/{CFG["MA_SLOW"]} delay={CFG["CROSS_DELAY"]} '
        f'touch={CFG["TOUCH_MODE"]}{CFG["TOUCH_ATR"]} TP{CFG["ATR_TP"]}/SL{CFG["ATR_SL"]} '
        f'时停{CFG["MAX_BARS"]}根 | 仓位={SIZING} N={MAX_POSITIONS} '
        f'单币{CFG["MAX_NOTIONAL_PCT"]*LEVERAGE*100:.1f}% 总敞口{MAX_TOTAL_NOTIONAL_PCT*LEVERAGE:.2f}x '
        f'| MA出场={MA_EXIT} 同窗口一次={COOLDOWN_ONCE} BTC过滤={BTC_FILTER}')
    if FILLS_CFG:
        log('填仓: ' + ' '.join(
            f'{f["sym"]}(MA{f["ma_f"]}/{f["ma_s"]} {f["src"]} '
            f'{"金叉一次" if f["once"] else "水平"} {f["trail_atr"]}xATR{f["atr_len"]})'
            for f in FILLS_CFG))
    log(f'预期节奏(回测): 约86%轮次无新开仓, 均持仓3~4币, 牛市最长可连续空仓23天 —— '
        f'长时间无交易属正常, 以[心跳]行确认存活')
    _selfcheck()
    ex = make_exchange()
    st = load_state()
    _fills = {k for k in st.get('fills', {}) if not k.endswith('_done')}
    log(f'当前本地持仓记录: 空仓{list(st["positions"].keys()) or "无"} '
        f'填仓{list(_fills) or "无"}')
    # 启动/退出各推一条: 能从群里看出进程重启过(否则"一直安静"和"崩了又起"无法区分)
    tg.send(f'🚀 <b>MA{CFG["MA_SLOW"]}做空</b> 启动 [{mode}]\n'
            f'币池{len(SYMBOLS)} · N={MAX_POSITIONS} · 敞口'
            f'{MAX_TOTAL_NOTIONAL_PCT*LEVERAGE:.2f}x · 仓位={SIZING}'
            + (f'\n填仓: {", ".join(sorted(_fills)) or "-"}' if FILLS_CFG else '')
            + f'\n本地持仓: {", ".join(st["positions"]) or "无"}', log=log)
    try:
        while True:
            run_once(ex, st)
            time.sleep(_next_wait())
    except KeyboardInterrupt:
        log('收到中断, 退出')
        tg.send(f'🛑 <b>MA{CFG["MA_SLOW"]}做空</b> 已停止 (手动中断)\n'
                f'持仓 {len(st["positions"])} 个仍在交易所, TP/SL 委托不受影响', log=log)
        raise
    except Exception as e:
        log(f'[fatal] 主循环异常退出: {e}')
        tg.send(f'💥 <b>MA{CFG["MA_SLOW"]}做空</b> 异常退出\n<code>{str(e)[:300]}</code>\n'
                f'持仓 {len(st["positions"])} 个仍在交易所, 请尽快处理', log=log)
        raise


if __name__ == '__main__':
    _utf8_stdout()
    # 支持 --once: 只跑一轮 (便于测试/定时任务), 默认常驻轮询
    if '--once' in sys.argv:
        ex = make_exchange(); st = load_state()
        run_once(ex, st)
        log('单轮完成 (--once)')
    else:
        main()

