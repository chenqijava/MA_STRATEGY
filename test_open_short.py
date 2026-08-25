"""手动调用 open_short() 的测试脚本。

用途: 单独验证「开空 + 挂TP/SL」这段逻辑, 不必等策略在 4h 收盘时恰好出信号。

安全:
  - 默认继承 live_ma_short 的 DRY_RUN (=.env, 缺省 True): 只打印开仓意图, 绝不下真单。
  - 只有显式 DRY_RUN=false 且填了凭证时才会真实下单。真单不可逆, 谨慎。

两种取参方式:
  1) --live SYMBOL : 从 OKX 拉该币最新已收盘K线, 用真实 close/atr 和真实合约面值/账户权益,
                     完整走一遍 size_contracts + open_short (最贴近实盘)。
  2) 纯手填(默认)  : 用命令行/内置的样例数字直接调 open_short, 不连交易所(ex=None), 纯干跑演算。

示例:
  python test_open_short.py                          # 干跑, 内置样例(DOGE)
  python test_open_short.py --sym BTC --close 60000 --atr 800 --equity 5000
  python test_open_short.py --live DOGE              # 连所拉真实数据算, 仍受 DRY_RUN 保护
  DRY_RUN=false python test_open_short.py --live DOGE --sym DOGE   # ★真单★ 谨慎
"""
import argparse
import live_ma_short as L


def run_manual(sym, close, atr, equity, contracts, ctv):
    """不连交易所, 直接用给定数字调 open_short (ex=None)。"""
    L.log(f'[manual] 调 open_short: sym={sym} close={close} atr={atr} '
          f'equity={equity} contracts={contracts} ctv={ctv}')
    info = L.open_short(ex=None, sym=sym, close=close, atr=atr,
                        equity=equity, contracts=contracts, ctv=ctv)
    L.log(f'[manual] 返回 intent: {info}')
    return info


def run_live(sym, equity_override=None):
    """连 OKX 拉真实 close/atr/面值/权益, 走 size_contracts 后调 open_short。"""
    ex = L.make_exchange()
    df = L.fetch_ohlcv(ex, sym)
    if len(df) < L.CFG['MA_SLOW'] + 5:
        L.log(f'[{sym}] K线不足, 无法算 ATR')
        return None
    sig, close, atr = L.latest_signal(df)[:3]
    L.log(f'[{sym}] 最新收盘 close={close:.6f} atr={atr} (信号={sig}, 仅参考; 本脚本强制走开空)')
    if not (atr and atr > 0):
        L.log(f'[{sym}] ATR 无效, 放弃')
        return None
    equity = equity_override if equity_override is not None else L.get_equity(ex)
    contracts, ctv = L.size_contracts(ex, sym, equity, close, atr)
    if contracts <= 0:
        L.log(f'[{sym}] 该敞口开不出最小一手 (单币名义 '
              f'{equity*L.CFG["MAX_NOTIONAL_PCT"]*L.LEVERAGE:.0f}U), 放弃')
        return None
    info = L.open_short(ex, sym, close, atr, equity, contracts, ctv)
    L.log(f'[live] 返回 intent: {info}')
    return info


def main():
    ap = argparse.ArgumentParser(description='手动测试 open_short()')
    ap.add_argument('--live', metavar='SYMBOL',
                    help='连所拉真实数据算(如 DOGE); 不加则纯手填干跑')
    ap.add_argument('--sym', default='DOGE', help='币种 (手填模式用)')
    ap.add_argument('--close', type=float, default=0.12, help='收盘价 (手填模式)')
    ap.add_argument('--atr', type=float, default=0.004, help='ATR (手填模式)')
    ap.add_argument('--equity', type=float, default=5000, help='账户权益 (手填/覆盖 live)')
    ap.add_argument('--contracts', type=float, default=100, help='下单张数 (手填模式)')
    ap.add_argument('--ctv', type=float, default=1.0, help='合约面值/张 (手填模式, DOGE=1000, BTC=0.01)')
    args = ap.parse_args()

    L._utf8_stdout()
    mode = 'DRY_RUN(不下真单)' if L.DRY_RUN else '★ LIVE 真实下单 ★'
    L.log(f'open_short 测试 [{mode}]  杠杆={L.LEVERAGE}x  '
          f'ATR_TP={L.CFG["ATR_TP"]} ATR_SL={L.CFG["ATR_SL"]}')

    if args.live:
        run_live(args.live, equity_override=args.equity)
    else:
        run_manual(args.sym, args.close, args.atr, args.equity, args.contracts, args.ctv)


if __name__ == '__main__':
    main()
