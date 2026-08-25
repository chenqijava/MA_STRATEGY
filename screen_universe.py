"""筛选 OKX 上「市值 > 阈值」的 USDT 永续候选池。

市值来自 CoinGecko (OKX API 不提供市值)。步骤:
  1. CoinGecko 按市值降序拉 top N 币, 筛 market_cap > MIN_MCAP
  2. 与 OKX 线性USDT永续 base 求交集 (只保留 OKX 能交易的)
  3. 排除美股代币等非加密标的 (CoinGecko 交集天然排除, 因其只收录加密资产)
输出候选 base 列表, 供 fetch_new + validate_new 走验证流程。
"""
import os, sys, io, json, urllib.request
import ccxt
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

PROXY = os.environ.get('PROXY') or 'socks5h://127.0.0.1:10808'
MIN_MCAP = float(os.environ.get('MIN_MCAP', 1e8))   # 1亿美元
PAGES = int(os.environ.get('CG_PAGES', 4))          # 每页250, 4页=top1000

# CoinGecko symbol -> 可能与 OKX base 不一致的少数映射 (按需补)
SYMBOL_FIX = {}

# 严格过滤: 稳定币(无波动, 均线策略无意义) + 交易所/包装类 + 明显非币标的
EXCLUDE = {
    'USDC','USDT','DAI','TUSD','USDE','FDUSD','USDD','PYUSD','USDS','SKY','STABLE',  # 稳定币/稳定币生态
    'WBTC','WETH','WBETH','STETH','WEETH','WSTETH','CBBTC','RETH',                   # 包装/质押衍生
    'OKB','BGB','BNB','CRO','GT','KCS','HT','LEO',                                    # 交易所平台币(BNB是主流公链, 保留除外见下)
}
# BNB 是主流公链, 不当平台币剔除
EXCLUDE.discard('BNB')
# 上市需早于此日期才有足够回测历史 (回测窗 2025-01 起)
LISTED_BEFORE = '2025-01-01'


def cg_markets():
    """走代理拉 CoinGecko 市值榜。返回 [(symbol_upper, mcap, name), ...]。"""
    import socks, socket
    if PROXY.startswith('socks'):
        host = PROXY.split('://')[1].split(':')[0]
        port = int(PROXY.rsplit(':', 1)[1])
        socks.set_default_proxy(socks.SOCKS5, host, port, rdns=True)
        socket.socket = socks.socksocket
    out = []
    for pg in range(1, PAGES + 1):
        url = ('https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd'
               f'&order=market_cap_desc&per_page=250&page={pg}')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        if not data:
            break
        for c in data:
            mc = c.get('market_cap') or 0
            sym = (c.get('symbol') or '').upper()
            out.append((SYMBOL_FIX.get(sym, sym), mc, c.get('name')))
        print(f'  CoinGecko page{pg}: 累计 {len(out)}')
    return out


def okx_perp_bases():
    ex = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'},
                   'timeout': 20000, 'socksProxy': PROXY})
    ex.load_markets()
    return {m['base'] for m in ex.markets.values()
            if m.get('swap') and m.get('linear') and m.get('quote') == 'USDT'
            and m.get('settle') == 'USDT' and m.get('active', True)}


def main():
    print(f'筛选: 市值 > {MIN_MCAP/1e8:.1f}亿, OKX USDT永续')
    cg = cg_markets()
    okx = okx_perp_bases()
    print(f'OKX 永续 base 数: {len(okx)}')

    over = [(s, mc, nm) for (s, mc, nm) in cg if mc > MIN_MCAP]
    inter = [(s, mc, nm) for (s, mc, nm) in over if s in okx and s not in EXCLUDE]
    # 去重保市值最高
    seen, rows = set(), []
    for s, mc, nm in sorted(inter, key=lambda x: -x[1]):
        if s in seen:
            continue
        seen.add(s)
        rows.append((s, mc, nm))
    excluded = sorted({s for (s, mc, _) in over if s in okx and s in EXCLUDE})
    if excluded:
        print(f'\n已排除(稳定币/平台币/包装类): {",".join(excluded)}')
    print('注: 上市晚于回测窗(2025-01)的次新币, 会在拉数据阶段因历史不足被自动识别')

    print(f'\n市值>{MIN_MCAP/1e8:.0f}亿 且 OKX有永续: {len(rows)} 个')
    for s, mc, nm in rows:
        nm = ''.join(ch for ch in (nm or '') if ch.isprintable() and ord(ch) < 0x3000)
        line = f'  {s:8s} {mc/1e9:8.2f}B  {nm}'
        print(line.encode('utf-8', 'ignore').decode('utf-8'))
    bases = [s for s, _, _ in rows]
    print('\n=== 候选 base 列表 ===')
    print(','.join(bases))
    # 落盘供后续脚本读
    with open(os.path.join(os.path.dirname(__file__), 'universe.txt'), 'w', encoding='utf-8') as f:
        f.write(','.join(bases))
    print(f'\n已写 universe.txt ({len(bases)} 币)')


if __name__ == '__main__':
    main()
