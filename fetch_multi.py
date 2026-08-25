"""拉取多币种 4h K线到 data_cache, 用于跨标的显著性验证。同一时段 2025.1~2026.8。"""
import os
import ccxt
import pandas as pd

PROXY = 'socks5h://127.0.0.1:10808'
SYMBOLS = {
    # 第一批(已拉): BTC ETH SOL BNB XRP DOGE
    # 第二批: 主流+中小市值扩样本
    'ADA': 'ADA/USDT:USDT',
    'AVAX': 'AVAX/USDT:USDT',
    'LINK': 'LINK/USDT:USDT',
    'LTC': 'LTC/USDT:USDT',
    'DOT': 'DOT/USDT:USDT',
    'TRX': 'TRX/USDT:USDT',
    'ATOM': 'ATOM/USDT:USDT',
    'UNI': 'UNI/USDT:USDT',
    'FIL': 'FIL/USDT:USDT',
    'APT': 'APT/USDT:USDT',
}
TF = '4h'
START_DATE = '2025-01-01 00:00:00'
END_DATE = '2026-08-01 00:00:00'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def fetch(name, symbol):
    out = os.path.join(OUT_DIR, f'{name}-USDT-SWAP_{TF}_20250101_20260801.csv')
    if os.path.exists(out):
        print(f'[{name}] 已存在, 跳过')
        return
    cfg = {'enableRateLimit': True, 'options': {'defaultType': 'swap'}, 'timeout': 15000,
           'socksProxy': PROXY}
    ex = ccxt.okx(cfg)
    tf_ms = ex.parse_timeframe(TF) * 1000
    since = ex.parse8601(START_DATE)
    end = ex.parse8601(END_DATE)
    all_data, retries = [], 0
    while since < end:
        try:
            o = ex.fetch_ohlcv(symbol, TF, since=since, limit=100,
                               params={'method': 'publicGetMarketHistoryCandles'})
            retries = 0
        except Exception as e:
            retries += 1
            print(f'[{name}] 出错 {str(e)[:50]} 重试{retries}')
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
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    df = df[(df['timestamp'] >= START_DATE) & (df['timestamp'] < END_DATE)]
    df.to_csv(out, index=False)
    print(f'[{name}] 保存 {len(df)} 根 -> {os.path.basename(out)}  '
          f'{df.timestamp.iloc[0]} ~ {df.timestamp.iloc[-1]}')


if __name__ == '__main__':
    for name, sym in SYMBOLS.items():
        fetch(name, sym)
    print('全部完成')
