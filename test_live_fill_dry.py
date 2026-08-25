"""实盘长多填仓 DRY_RUN 链路测试 (不联网)。

覆盖(对应 plan 的第 5 步):
1. FILLS_CFG 注册在两个 env 前缀下均生效 (BTC 10/120 close 3xATR20, ETH 20/85 high 4xATR18)
2. 无 fill 配置时 process_fills 空转无副作用 (回归)
3. process_fills 入场: 金叉当根 -> 算 contracts/notional, DRY_RUN 只记意图不动手
4. process_fills 出场: 持仓后 low<=stop 触发追踪止损平多 (stop0 兜底)
5. 金叉窗口 once 机制: 窗口内只做一次, 快线回落解除
6. 对冲: 主策略对该币开空 -> fill 先平 (日志顺序)
7. 闲置容量: 主空仓吃满 -> fill 无容量不开

运行: python test_live_fill_dry.py
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
# 必须在 import live_ma_short 之前设, 注册表在 import 时按 env 构建
os.environ['BTC_FILL'] = '1'
os.environ['ETH_FILL'] = '1'
os.environ['DRY_RUN'] = 'true'
import tempfile
import numpy as np
import pandas as pd
import live_ma_short as lm

# 测试会写状态文件 (process_symbol/process_fills 内 save_state), 指向临时文件防污染
lm.STATE_FILE = os.path.join(tempfile.gettempdir(), 'live_state_test.json')
if os.path.exists(lm.STATE_FILE):
    os.remove(lm.STATE_FILE)
lm.ENTRY_MAX_LAG_S = 0                # 关掉"启动首根参考价已过期"闸, 测试与墙钟解耦

PASS = 0
N_FAIL = 0

def ok(cond, name):
    global PASS, N_FAIL
    if cond:
        PASS += 1
        print(f'  ✓ {name}')
    else:
        N_FAIL += 1
        print(f'  ✗ {name}')
    return cond

def syn_df(up_last=150.0):
    """合成 4h K线: 前段缓跌(MA10<MA120), 最后一根跳涨 -> 金叉当根。"""
    L = 240
    closes = 100.0 - 0.05 * np.arange(L, dtype=float)
    closes[-1] = up_last
    highs = closes * 1.01
    lows = closes * 0.99
    opens = np.concatenate([closes[:1], closes[:-1]])
    opens[-1] = closes[-2]
    idx = pd.date_range('2025-01-01 00:00', periods=L, freq='4h', tz='UTC')
    return pd.DataFrame({'open': opens, 'high': highs, 'low': lows,
                         'close': closes, 'vol': np.ones(L)}, index=idx)

class FakeEx:
    """process_fills 只用到: markets 集合 / market() / amount_to_precision / fetch_ohlcv。
    其余(下单等)走 DRY_RUN 分支不联网。"""
    C = 0.01   # contractSize
    def __init__(self, syms=('BTC', 'ETH')):
        self.syms = syms
        self.markets = {lm.inst(s) for s in syms}
    def market(self, im):
        return {'id': im.replace('/', '-').replace(':USDT', '-USDT-SWAP'),
                'contractSize': self.C,
                'limits': {'amount': {'min': 1.0}},
                'precision': {'amount': 1e-2}}
    def amount_to_precision(self, im, raw):
        return str(round(raw / 1e-2) * 1e-2)

def new_state():
    return {'positions': {}, 'fills': {}, 'pending': {}, 'done_window': {},
            'last_bar': {}, 'skipped': {}}

def syn_death_cross():
    """先涨后跌 -> MA5 下穿 MA90 (死叉早于末根≥6根), 最末一根反弹贴近 MA90 触碰窗口,
    直至 latest_signal 返回 -1 (自校验, 不手算信号条件)。"""
    L = 240
    c = np.empty(L)
    c[:110] = 50.0 + np.arange(110) * (50.0 / 109)        # 50 -> 100 (涨)
    c[110:] = 100.0 - np.arange(L - 110) * (30.0 / (L - 111))  # 100 -> ~70 (跌)
    o = np.empty(L); o[0] = c[0]; o[1:] = c[:-1]
    h = np.maximum(c, o) * 1.001
    l = np.minimum(c, o) * 0.999
    idx = pd.date_range('2025-01-01', periods=L, freq='4h', tz='UTC')
    df = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}, index=idx)
    # 在最后一根搜索: 把收盘价推到"贴近 MA90 但不上破"的触碰窗口
    di = lm.mp.calc_indicators(df, lm.CFG)
    ma90 = di['ma_slow'].iloc[-2]     # 未含末根(上一根)的慢线, 末根拉高影响其一成
    atr = di['atr'].iloc[-2]
    for f_pos in (0.10, 0.15, 0.20, 0.25, 0.30, 0.05, 0.40):
        hh = ma90 - f_pos * atr
        df2 = df.copy()
        df2.iloc[-1, df2.columns.get_loc('close')] = hh * 0.997
        df2.iloc[-1, df2.columns.get_loc('open')] = hh * 1.001
        df2.iloc[-1, df2.columns.get_loc('high')] = hh
        df2.iloc[-1, df2.columns.get_loc('low')] = hh * 0.99
        sig, close, atr_cur, ma_s, mf = lm.latest_signal(df2)
        if sig == -1:
            return df2
    raise RuntimeError('syn_death_cross 未能构造出做空信号 (自校验失败)')

def run_fills(st, df_map, equity=10000.0):
    """喂 df_map: sym->DataFrame, 跑一轮 process_fills, 返回 (ex, equity)。"""
    ex = FakeEx()
    _orig = lm.fetch_ohlcv
    def fake_fetch(e, sym, limit=300):
        return df_map[sym]
    lm.fetch_ohlcv = fake_fetch
    try:
        lm.process_fills(ex, st, equity)
    finally:
        lm.fetch_ohlcv = _orig
    return ex

def main():
    print('== 1. 注册表 ==')
    cfgs = {f['sym']: f for f in lm.FILLS_CFG}
    ok('BTC' in cfgs and 'ETH' in cfgs, f'双 fill 已注册: {sorted(cfgs)}')
    if 'BTC' in cfgs:
        c = cfgs['BTC']
        ok((c['ma_f'], c['ma_s'], c['src']) == (10, 120, 'close'), 'BTC: MA10/120 close')
        ok((c['trail_atr'], c['atr_len'], c['once']) == (3.0, 20, True), 'BTC: 3xATR20 金叉一次')
    if 'ETH' in cfgs:
        c = cfgs['ETH']
        ok((c['ma_f'], c['ma_s'], c['src']) == (20, 85, 'high'), 'ETH: MA20/85 high')
        ok((c['trail_atr'], c['atr_len'], c['once']) == (4.0, 18, True), 'ETH: 4xATR18 金叉一次')
    ok(lm.FILL_SLOT_NOTIONAL == lm._env_float('MAX_NOTIONAL_PCT', 0.025) * lm.LEVERAGE,
       f'FILL_SLOT_NOTIONAL={lm.FILL_SLOT_NOTIONAL} (=单币上限 {lm._env_float("MAX_NOTIONAL_PCT", 0.025)*lm.LEVERAGE})')

    print('== 2. 无 fill 配置空转 (回归) ==')
    saved = lm.FILLS_CFG
    lm.FILLS_CFG = []
    st = new_state()
    run_fills(st, {'BTC': syn_df(), 'ETH': syn_df()})
    lm.FILLS_CFG = saved
    ok(not st['fills'] and not st['pending'], '无 fill 时 process_fills 不动状态')

    print('== 3. 金叉入场 ==')
    st = new_state()
    run_fills(st, {'BTC': syn_df(), 'ETH': syn_df()})
    ok('BTC' in st['fills'] and 'ETH' in st['fills'], f'两币金叉都入多: {sorted(k for k in st["fills"])}')
    fb = st['fills'].get('BTC')
    ok(fb and fb['side'] == 'long' and fb['contracts'] > 0, f'BTC 持仓信息: {fb and fb["contracts"]}张 @{fb and fb.get("entry")}')
    ok(fb and fb['stop0'] < fb['entry'], f'BTC 初始止损低于入场: stop0={fb and fb.get("stop0")}')
    ok(st['fills'].get('BTC_done') and st['fills'].get('ETH_done'), 'once 标记已置')
    # 名义合理性: slot=equity*0.075=750U, contracts = 750/(close*ctv)
    ctv = FakeEx.C
    expect = int(750 / (fb['entry'] / (1 + lm.mp.SLIPPAGE) * ctv))
    ok(abs(fb['contracts'] - expect) <= 1, f'BTC 张数与名义一致: {fb["contracts"]} vs ~{expect}')

    print('== 4. 同一窗口不重复入场 (once) ==')
    st = new_state()
    st['fills']['BTC_done'] = True      # 模拟前一轮已做过
    run_fills(st, {'BTC': syn_df(), 'ETH': syn_df()})
    ok('BTC' not in st['fills'] and 'ETH' in st['fills'],
       'BTC_done 挡住重复入场, ETH 不受影响')

    print('== 5. 快线回落后窗口解除 + 可再入场 ==')
    st = new_state()
    st['fills']['BTC_done'] = True
    flat = syn_df()
    flat['close'].iloc[-1] = flat['close'].iloc[-2]      # 去掉末根跳涨, 整段缓跌
    flat['high'] = flat['close'] * 1.01
    flat['low'] = flat['close'] * 0.99
    run_fills(st, {'BTC': flat, 'ETH': syn_df()})
    ok('BTC_done' not in st['fills'], f'快线回落解除 done: {st["fills"].get("BTC_done")}')
    ok('BTC' not in st['fills'], '解除后非金叉根不加仓')

    print('== 6. 追踪止损出场 ==')
    st = new_state()
    fb = dict(sym='BTC', side='long', contracts=10, ctv=FakeEx.C,
              entry=100.0, fill=100.0, atr_entry=2.0, ext=100.0,
              stop0=100 - 3.0 * 2.0, bars=0,
              entry_time='2025-02-01T00:00:00+00:00')
    st['fills']['BTC'] = fb
    drop = syn_df()
    drop['low'].iloc[-1] = fb['stop0'] - 0.1    # 最新已收盘 low 破止损
    run_fills(st, {'BTC': drop, 'ETH': syn_df()})
    ok('BTC' not in st['fills'], f'trail 触发平多: 剩余={sorted(st["fills"])}')

    print('== 6b. 出场无前视: 本根冲高不参与本根 stop ==')
    # 末根盘中冲高到 110 又回落到 96。stop 若先并入本根 high(=110) 会推到 110-gap(≈104),
    # low=96 误触发; 正确用上一根 ext=100, stop≈94, 96>94 不触发, 且 ext 并入 110 供下根。
    st = new_state()
    st['fills']['BTC'] = dict(sym='BTC', side='long', contracts=10, ctv=FakeEx.C,
                              entry=100.0, fill=100.0, atr_entry=2.0, ext=100.0,
                              stop0=80.0, bars=0, entry_time='2025-02-01T00:00:00+00:00')
    sp = syn_df()
    sp['high'].iloc[-1] = 110.0
    sp['low'].iloc[-1] = 96.0
    sp['close'].iloc[-1] = 97.0
    run_fills(st, {'BTC': sp, 'ETH': syn_df()})
    ok('BTC' in st['fills'], '本根冲高回落不误触发 (ext 前视修复)')
    ok(st['fills']['BTC']['ext'] == 110.0, f'ext 已并入本根 high 供下根用: {st["fills"]["BTC"]["ext"]}')

    print('== 6c. 重启接管 fill (stop0=None) 不崩溃 ==')
    # 模拟 sync_positions 恢复的 fill: stop0=None (ext/ATR 未知)。process_fills 必须
    # 正常走完出场判定 (stop 用 ext-gap 兜底), 不能因 max(None,...) 抛 TypeError。
    st = new_state()
    st['fills']['BTC'] = dict(sym='BTC', side='long', contracts=10, ctv=FakeEx.C,
                              entry=100.0, fill=100.0, atr_entry=None, ext=100.0,
                              stop0=None, bars=0, entry_time='2025-02-01T00:00:00+00:00',
                              recovered=True)
    run_fills(st, {'BTC': syn_df(), 'ETH': syn_df()})
    ok('BTC' in st['fills'], f'恢复的 fill 出场判定不崩溃: {st["fills"].get("BTC") is not None}')

    print('== 7. 对冲: 主策略开空 -> fill 先平 ==')
    # 走真实 process_symbol: BTC 出死叉做空信号 -> 开空成功 -> 对冲平掉 BTC 的 fill
    # (DRY_RUN 下 open_short/close_fill_long 都只记日志)。BTC 的 K 线要构成:
    # 死叉 6 根前已发生 + MA5 持续在 MA90 下 + 末根反弹触碰 0.5ATR 阴线确认。
    crops = syn_death_cross()
    st = new_state()
    st['fills']['BTC'] = dict(sym='BTC', side='long', contracts=5, ctv=FakeEx.C,
                              entry=88.0, fill=88.0, atr_entry=1.0, ext=88.0,
                              stop0=85.0, bars=0, entry_time='2025-02-01T00:00:00+00:00')
    st['fills']['BTC_done'] = True        # 模拟该金叉窗口已做过
    ex = FakeEx()
    _orig = lm.fetch_ohlcv
    lm.fetch_ohlcv = lambda e, sym, limit=300: crops if sym == 'BTC' else syn_df()
    try:
        r = lm.process_symbol(ex, st, 'BTC', 10000.0)
    finally:
        lm.fetch_ohlcv = _orig
    ok(bool(r), 'BTC 死叉信号被接受 (开空成功)')
    ok('BTC' in st['positions'], f'BTC 主空已开: {st["positions"].get("BTC", {}).get("contracts")}张')
    ok('BTC' not in st['fills'], '对冲: BTC 的 fill 已平 (本地记录弹出)')
    ok(st['fills'].get('BTC_done') is True,
       '对冲后 done 标记保留 (金叉窗口未结束, 与回测口径一致 — 主策略平空后、快线回落前不会重入)')

    print('== 8. 闲置容量闸 ==')
    st = new_state()
    st['positions']['X'] = dict(sym='X', side='short', contracts=1000, entry=100.0,
                                ctv=FakeEx.C)   # 名义=1000*100*0.01=1000U
    st['positions']['Y'] = dict(sym='Y', side='short', contracts=1000, entry=100.0,
                                ctv=FakeEx.C)
    # 已用 2000U... cap=10000*3*0.5=15000 => 仍有 idle, 改为用小权益压掉
    run_fills(st, {'BTC': syn_df(), 'ETH': syn_df()}, equity=100.0)
    ok(not st['fills'], f'权益100U cap=150U, 金叉但无容量: fills={sorted(st["fills"])}')

    if os.path.exists(lm.STATE_FILE):
        os.remove(lm.STATE_FILE)
    print(f'\n--- {PASS} 项通过, 失败 {N_FAIL} 项 ---')
    return 0 if N_FAIL == 0 else 1

if __name__ == '__main__':
    sys.exit(main())