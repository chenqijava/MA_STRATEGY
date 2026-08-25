"""拉取 OKX ETH-USDT-SWAP 的 1h / 4h K线, 缓存到 data_cache, 供回测。

代理 socks5h://127.0.0.1:10808 (已验证可连)。列名对齐现有缓存: timestamp,open,high,low,close,vol
"""
import os
import ccxt
import pandas as pd

PROXY = 'socks5h://127.0.0.1:10808'
SYMBOL = 'ETH/USDT:USDT'
START_DATE = '2025-01-01 00:00:00'
END_DATE = '2026-08-01 00:00:00'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def fetch(tf):
    cfg = {'enableRateLimit': True, 'options': {'defaultType': 'swap'}, 'timeout': 15000}
    cfg['socksProxy'] = PROXY
    ex = ccxt.okx(cfg)
    tf_ms = ex.parse_timeframe(tf) * 1000
    since = ex.parse8601(START_DATE)
    end = ex.parse8601(END_DATE)
    all_data, retries = [], 0
    while since < end:
        try:
            o = ex.fetch_ohlcv(SYMBOL, tf, since=since, limit=100,
                               params={'method': 'publicGetMarketHistoryCandles'})
            retries = 0
        except Exception as e:
            retries += 1
            print(f'[{tf}] 出错: {str(e)[:60]} 第{retries}次重试')
            if retries >= 5:
                break
            ex.sleep(2000)
            continue
        if not o:
            break
        all_data += o
        nxt = o[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
        print(f'\r[{tf}] 已拉 {len(all_data)} 根, 至 {pd.to_datetime(o[-1][0], unit="ms")}', end='')
    print()
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    df = df[(df['timestamp'] >= START_DATE) & (df['timestamp'] < END_DATE)]
    out = os.path.join(OUT_DIR, f'ETH-USDT-SWAP_{tf}_20250101_20260801.csv')
    df.to_csv(out, index=False)
    print(f'[{tf}] 保存 {len(df)} 根 -> {out}  区间 {df.timestamp.iloc[0]} ~ {df.timestamp.iloc[-1]}')


if __name__ == '__main__':
    for tf in ['1h', '4h']:
        fetch(tf)
