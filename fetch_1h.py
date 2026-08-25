"""拉取全部币种的 1h K线(补齐, 已存在则跳过), 用于 4h→1h 稳健性对比。"""
import os
import ccxt
import pandas as pd

PROXY = 'socks5h://127.0.0.1:10808'
SYMBOLS = {s: f'{s}/USDT:USDT' for s in
           ['BTC', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'LTC',
            'DOT', 'TRX', 'ATOM', 'UNI', 'FIL', 'APT']}  # ETH 已有
TF = '1h'
START_DATE = '2025-01-01 00:00:00'
END_DATE = '2026-08-01 00:00:00'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def fetch(name, symbol):
    out = os.path.join(OUT_DIR, f'{name}-USDT-SWAP_{TF}_20250101_20260801.csv')
    if os.path.exists(out):
        print(f'[{name}] 已存在, 跳过'); return
    ex = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'},
                   'timeout': 15000, 'socksProxy': PROXY})
    tf_ms = ex.parse_timeframe(TF) * 1000
    since, end = ex.parse8601(START_DATE), ex.parse8601(END_DATE)
    data, retries = [], 0
    while since < end:
        try:
            o = ex.fetch_ohlcv(symbol, TF, since=since, limit=100,
                               params={'method': 'publicGetMarketHistoryCandles'})
            retries = 0
        except Exception as e:
            retries += 1
            if retries >= 5:
                break
            ex.sleep(2000); continue
        if not o:
            break
        data += o
        nxt = o[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    df = df[(df['timestamp'] >= START_DATE) & (df['timestamp'] < END_DATE)]
    df.to_csv(out, index=False)
    print(f'[{name}] 保存 {len(df)} 根')


if __name__ == '__main__':
    for n, s in SYMBOLS.items():
        fetch(n, s)
    print('全部完成')
