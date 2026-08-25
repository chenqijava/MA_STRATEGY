"""模拟盘诊断: 验证 ccxt 连接 / .env 凭证登录 / ETH 挂止盈止损委托并撤销。

★ 强制 OKX_DEMO=true, 只打到模拟盘, 不碰真实资金。★

检查项:
  1) ccxt 版本 + 模拟盘登录 (set_sandbox_mode)
  2) 读模拟盘余额 (验证凭证是"模拟盘专用key"; 真盘key在模拟盘会报错)
  3) 持仓模式探测 posMode
  4) ETH 挂一张附带 TP/SL 的极小委托 -> 查到该 algo 委托 -> 撤销 -> 确认已撤
     (与 live_ma_short.close_expired_positions 的撤单方式一致)

用法:
  python diag_demo.py                      # 默认只做 1~3 (只读, 零下单)
  python diag_demo.py --order              # 额外做 4 (会在模拟盘挂单并撤销)
  PROXY=socks5h://127.0.0.1:10808 python diag_demo.py --order
"""
import os, sys
# ---- 强制模拟盘, 必须在 import live_ma_short 之前设好, 使其 OKX_DEMO=True ----
os.environ['OKX_DEMO'] = 'true'
os.environ['DRY_RUN'] = 'false'          # 需真实调用API(打到模拟盘), 但绝不碰真盘
os.environ['PROXY'] = 'socks5h://127.0.0.1:10808'
import ccxt
import live_ma_short as L

if sys.stdout.encoding and not sys.stdout.encoding.lower().startswith('utf'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    do_order = '--order' in sys.argv
    print('=' * 60)
    print(f'模拟盘诊断  ccxt={ccxt.__version__}  OKX_DEMO={L.OKX_DEMO}  DRY_RUN={L.DRY_RUN}')
    print('=' * 60)
    if not L.OKX_DEMO or L.DRY_RUN:
        print('✗ 环境未强制到模拟盘, 中止'); return

    # 1) 连接 + 登录
    try:
        ex = L.make_exchange()
        print('✓ [1] ccxt 连接 + 模拟盘登录成功 (set_sandbox_mode)')
    except Exception as e:
        print(f'✗ [1] 连接/登录失败: {e}')
        print('   排查: 代理(PROXY)? 凭证是否为"模拟盘专用key"(实盘key在模拟盘无效)?')
        return

    # 2) 读模拟盘余额
    try:
        bal = ex.fetch_balance()
        usdt = bal.get('USDT', {}).get('total', 0)
        print(f'✓ [2] 模拟盘余额可读: USDT total = {usdt}')
    except Exception as e:
        print(f'✗ [2] 读余额失败(凭证可能不是模拟盘key): {e}'); return

    # 3) 持仓模式
    print(f'✓ [3] 持仓模式 posMode = {L._POS_MODE}')

    if not do_order:
        print('\n只读检查完成。加 --order 可测 ETH 挂单+撤单。')
        return

    # 4) ETH 挂附带 TP/SL 的极小委托, 再撤销
    sym = 'ETH'
    im = L.inst(sym)
    print(f'\n[4] ETH 挂单+撤单测试 ({im})')
    try:
        m = ex.market(im)
        min_amt = m['limits']['amount']['min'] or 1
        ctv = m['contractSize']
        px = float(ex.fetch_ticker(im)['last'])
        print(f'    ETH 现价≈{px}  最小下单量={min_amt}张  面值={ctv}')
        # 开一张最小空单, 附带 TP/SL (远离现价, 不会立刻触发)
        params = {'tdMode': 'isolated', **L._pos_side_params('sell'),
                  'attachAlgoOrds': [{
                      'tpTriggerPx': str(round(px * 0.80, 2)), 'tpOrdPx': '-1',
                      'slTriggerPx': str(round(px * 1.20, 2)), 'slOrdPx': '-1',
                      'tpTriggerPxType': 'last', 'slTriggerPxType': 'last'}]}
        ex.set_leverage(L.LEVERAGE, im, params={'mgnMode': 'isolated', **L._pos_side_params('sell')})
        o = ex.create_order(im, 'market', 'sell', min_amt, params=params)
        oid = o.get('id')
        print(f'    ✓ 已开最小空单 (id={oid}), 附带 TP/SL')

        instId = m['id']            # OKX 原生 instId, 如 ETH-USDT-SWAP
        import time; time.sleep(1)
        # 查该合约的挂起 algo 委托: 附带TP/SL是 OCO 类型, 用 OKX 原生端点 (ccxt统一接口不支持这个ordType)
        def fetch_algos():
            out = []
            for ot in ('oco', 'conditional'):
                try:
                    r = ex.private_get_trade_orders_algo_pending({'instId': instId, 'ordType': ot})
                    out += r.get('data', [])
                except Exception as e:
                    print(f'    ⚠ 查 {ot} 委托失败: {str(e)[:60]}')
            return out
        algos = fetch_algos()
        ids = [a.get('algoId') for a in algos]
        print(f'    查到 algo 委托 {len(algos)} 个: {ids}')

        # 撤销 algo 委托: OKX 原生 cancel-algos, 传 [{algoId, instId}]
        for a in algos:
            aid = a.get('algoId')
            try:
                ex.private_post_trade_cancel_algos([{'algoId': aid, 'instId': instId}])
                print(f'    ✓ 撤销 algo 委托 {aid}')
            except Exception as e:
                print(f'    ✗ 撤销 {aid} 失败: {str(e)[:60]}')

        # 平掉刚开的空单 (市价买回), 恢复到无持仓
        ex.create_order(im, 'market', 'buy', min_amt,
                        params={'tdMode': 'isolated',
                                'reduceOnly': (L._POS_MODE != 'long_short_mode'),
                                **L._pos_side_params('buy')})
        print('    ✓ 已平掉测试仓 (市价买回)')

        # 确认无残留 algo 委托
        time.sleep(1)
        left = fetch_algos()
        print(f'    收尾确认: 残留 algo 委托 = {len(left)} 个 ' + ('✓ 干净' if not left else '✗ 有残留!'))
    except Exception as e:
        print(f'    ✗ 挂单/撤单测试失败: {e}')
        print('      如为参数错, 把完整报错发我按 ccxt 版本调 attachAlgoOrds/cancel 用法。')


if __name__ == '__main__':
    main()
