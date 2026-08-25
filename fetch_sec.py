"""拉秒级K线 (高频做市回测用)。OKX 支持 1s, 但单次 limit 有上限, 需分页。

注意: 1s K线只有 OHLCV, **没有盘口深度**。做市回测用它只能近似:
  - 无法知道真实买卖价差(spread), 只能用 high-low 或固定假设
  - 无法知道排队位置与成交概率, 只能假设"价格触及即成交"(乐观)
所以这类回测的结论上限是"筛掉明显不可行的参数", 不能当作收益预期。

用法: python fetch_sec.py BTC --hours 24
"""
import os, sys, argparse, time
import ccxt
import pandas as pd

PROXY = os.environ.get('PROXY') or 'socks5h://127.0.0.1:10808'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base', nargs='?', default='BTC')
    ap.add_argument('--hours', type=float, default=24)
    ap.add_argument('--tf', default='1s')
    a = ap.parse_args()
    cfg = {'enableRateLimit': True, 'timeout': 20000, 'options': {'defaultType': 'swap'}}
    if PROXY.startswith('socks'):
        cfg['socksProxy'] = PROXY
    ex = ccxt.okx(cfg)
    ex.load_markets()
    sym = f'{a.base}/USDT:USDT'
    end = ex.milliseconds()
    start = end - int(a.hours * 3600 * 1000)
    rows, cursor = [], end
    while cursor > start:
        try:
            o = ex.fetch_ohlcv(sym, a.tf, limit=100,
                               params={'method': 'publicGetMarketHistoryCandles', 'after': cursor})
        except Exception as e:
            print(f'\n出错 {str(e)[:60]}, 重试')
            ex.sleep(1500)
            continue
        if not o:
            break
        rows = o + rows
        earliest = o[0][0]
        if earliest >= cursor:
            break
        cursor = earliest
        print(f'\r已拉 {len(rows)} 根, 最早 {pd.to_datetime(earliest, unit="ms")}', end='')
    print()
    if not rows:
        print('无数据')
        return
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    out = os.path.join(OUT_DIR, f'{a.base}-USDT-SWAP_{a.tf}_recent.csv')
    df.to_csv(out, index=False)
    span = (df.timestamp.iloc[-1] - df.timestamp.iloc[0]).total_seconds() / 3600
    print(f'保存 {len(df)} 根 ({span:.1f}小时) -> {out}')
    print(f'区间 {df.timestamp.iloc[0]} ~ {df.timestamp.iloc[-1]}')


if __name__ == '__main__':
    main()
