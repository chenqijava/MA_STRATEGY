"""拉取更早的历史 K 线 (OKX history-candles 只保留到 2025-01, 更早需换源)。

Binance USDT 永续 / Bybit 都能给到 2023-01-01。同一标的的 4h OHLC 在
交易所间差异极小 (< 几个 bp), 对 MA/ATR 信号无实质影响, 可用于样本外检验。

用法:
  python fetch_hist.py --start 2023-01-01 --end 2025-01-01 --universe universe_live.txt
  python fetch_hist.py BTC ETH --start 2023-01-01 --end 2025-01-01
"""
import os, sys, argparse
import ccxt
import pandas as pd

PROXY = os.environ.get('PROXY') or 'socks5h://127.0.0.1:10808'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
EXCHANGES = ['binanceusdm', 'bybit', 'binance']   # 依次尝试 (swap → spot)


def make_ex(name):
    cfg = {'enableRateLimit': True, 'timeout': 20000, 'options': {'defaultType': 'swap'}}
    if PROXY.startswith('socks'):
        cfg['socksProxy'] = PROXY
    elif PROXY:
        cfg['httpProxy'] = PROXY
    ex = getattr(ccxt, name)(cfg)
    ex.load_markets()
    return ex


def resolve(ex, base):
    # 迷因币在 Binance/Bybit 上是 1000x/1000000x 计价合约。价格整体缩放不改变
    # MA 交叉、ATR/价格 比例等信号, 可直接当原币用。
    for cand in (base, '1000' + base, '1000000' + base, base + '1000'):
        for m in ex.markets.values():
            if (m.get('swap') and m.get('linear') and m.get('base') == cand
                    and m.get('quote') == 'USDT' and m.get('settle') == 'USDT'):
                return m['symbol']
            # 现货 Binance (BTC/USDT 等)，数据早于永续合约
            if (m.get('spot') and m.get('base') == cand
                    and m.get('quote') == 'USDT'):
                return m['symbol']
    return None


def fetch_one(ex, sym, tf, start_ms, end_ms):
    """since-forward 分页 (Binance/Bybit 支持), 返回 ohlcv 列表。"""
    out, cursor, retries = [], start_ms, 0
    while cursor < end_ms:
        try:
            o = ex.fetch_ohlcv(sym, tf, since=cursor, limit=1000)
            retries = 0
        except Exception as e:
            retries += 1
            if retries >= 4:
                raise
            ex.sleep(1500)
            continue
        if not o:
            break
        out += o
        nxt = o[-1][0] + 1
        if nxt <= cursor:
            break
        cursor = nxt
        print(f'\r  已拉 {len(out)} 根, 到 {pd.to_datetime(o[-1][0], unit="ms")}', end='')
    print()
    return out


def fetch(exs, base, tf, start, end):
    start_ms = ccxt.Exchange.parse8601(start + ' 00:00:00')
    end_ms = ccxt.Exchange.parse8601(end + ' 00:00:00')
    best = None
    for name, ex in exs.items():
        sym = resolve(ex, base)
        if not sym:
            continue
        print(f'[{base} {tf}] {name} {sym}')
        try:
            data = fetch_one(ex, sym, tf, start_ms, end_ms)
        except Exception as e:
            print(f'  {name} 失败: {str(e)[:70]}')
            continue
        if not data:
            continue
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates('timestamp').sort_values('timestamp')
        df = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
        if len(df) < 200:
            print(f'  {name} 仅 {len(df)} 根, 太短, 换源')
            continue
        print(f'  {name} 得到 {len(df)} 根 ({df.timestamp.iloc[0]} ~ {df.timestamp.iloc[-1]})')
        if best is None or len(df) > len(best):
            best = df
    if best is None:
        print(f'[{base} {tf}] 所有源均无数据')
        return False
    tag = f'{start.replace("-", "")}_{end.replace("-", "")}'
    out = os.path.join(OUT_DIR, f'{base}-USDT-SWAP_{tf}_{tag}.csv')
    best.to_csv(out, index=False)
    print(f'  保存 {len(best)} 根 ({best.timestamp.iloc[0]} ~ {best.timestamp.iloc[-1]})')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bases', nargs='*')
    ap.add_argument('--start', default='2023-01-01')
    ap.add_argument('--end', default='2025-01-01')
    ap.add_argument('--tf', default='4h')
    ap.add_argument('--universe', default=None)
    args = ap.parse_args()

    bases = [b.upper() for b in args.bases]
    if args.universe:
        # 文件可能是一行一个币, 也可能是逗号分隔 (universe_live.txt)
        with open(os.path.join(os.path.dirname(__file__), args.universe)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                bases += [t.strip().upper() for t in line.split(',') if t.strip()]
    bases = list(dict.fromkeys(bases))

    exs = {}
    for name in EXCHANGES:
        try:
            exs[name] = make_ex(name)
            print(f'{name} 就绪')
        except Exception as e:
            print(f'{name} 初始化失败: {str(e)[:60]}')

    ok, fail = [], []
    for b in bases:
        (ok if fetch(exs, b, args.tf, args.start, args.end) else fail).append(b)
    print(f'\n成功 {len(ok)} / 失败 {len(fail)}')
    if fail:
        print('失败:', ' '.join(fail))


if __name__ == '__main__':
    main()
