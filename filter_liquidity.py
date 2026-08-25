"""流动性过滤: 从 universe_pass.txt 剔除 24h 成交额 < 阈值 的币。

成交额(USDT) = OKX ticker 的 volCcy24h(成交的基础币数量) × last价。
注: swap 的 volCcy24h 是"币数"不是"U", 故要乘价格。

用法:
  python filter_liquidity.py                 # 默认阈值100万U, 读 universe_pass.txt, 写 universe_liquid.txt
  MIN_VOL_USD=2e6 python filter_liquidity.py  # 自定义阈值
"""
import os, sys, json
import ccxt

PROXY = os.environ.get('PROXY') or 'socks5h://127.0.0.1:10808'
MIN_VOL_USD = float(os.environ.get('MIN_VOL_USD', '1e6'))
SRC = os.environ.get('SRC', 'universe_pass.txt')
OUT = os.environ.get('OUT', 'universe_liquid.txt')

if sys.stdout.encoding and not sys.stdout.encoding.lower().startswith('utf'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    d = os.path.dirname(os.path.abspath(__file__))
    syms = open(os.path.join(d, SRC), encoding='utf-8').read().strip().split(',')
    ex = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'},
                   'timeout': 20000, 'socksProxy': PROXY})
    ex.load_markets()
    vols = {}
    for s in syms:
        try:
            t = ex.fetch_ticker(f'{s}/USDT:USDT')['info']
            vols[s] = float(t['volCcy24h']) * float(t['last'])   # 成交额(U)
        except Exception as e:
            print(f'[{s}] ticker 失败: {str(e)[:50]}')
            vols[s] = -1.0

    keep = [s for s in syms if vols[s] >= MIN_VOL_USD]
    drop = [s for s in syms if 0 <= vols[s] < MIN_VOL_USD]
    err = [s for s in syms if vols[s] < 0]

    print(f'\n阈值: 24h成交额 >= {MIN_VOL_USD/1e6:.1f}M U')
    print(f'保留 {len(keep)} / 剔除 {len(drop)}' + (f' / 取价失败 {len(err)}' if err else ''))
    print('剔除(低流动性):', ','.join(sorted(drop, key=lambda x: vols[x])) or '无')
    for s in sorted(drop, key=lambda x: vols[x]):
        print(f'  {s:8s} {vols[s]/1e6:8.3f} M')
    if err:
        print('取价失败(保守剔除):', ','.join(err))

    open(os.path.join(d, OUT), 'w', encoding='utf-8').write(','.join(keep))
    json.dump(vols, open(os.path.join(d, 'vol24h.json'), 'w'))
    print(f'\n已写 {OUT} ({len(keep)}币) 与 vol24h.json')


if __name__ == '__main__':
    main()
