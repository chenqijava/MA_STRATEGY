"""拉取真实资金费率历史 -> data_cache/funding_{base}_{tag}.csv

资金费是这个策略绕不开的成本:
  做空腿  资金费率>0 时空头**收**钱, <0 时付钱 (永续多头付空头)。
  对冲腿  做多 BTC, 资金费率>0 时多头**付**钱 —— 币圈长期为正, 所以对冲腿常年付费。
每 8h 结算一次, 按名义计。之前 FUNDING_BP 默认 0 (未建模), 这里换成真实历史。

用法:
  python fetch_funding.py BTC --start 2023-01-01 --end 2025-01-01
  python fetch_funding.py --universe universe_live.txt --start 2025-01-01 --end 2026-08-01
"""
import os, sys, argparse
import ccxt
import pandas as pd

PROXY = os.environ.get('PROXY') or 'socks5h://127.0.0.1:10808'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def make_ex():
    cfg = {'enableRateLimit': True, 'timeout': 20000, 'options': {'defaultType': 'swap'}}
    if PROXY.startswith('socks'):
        cfg['socksProxy'] = PROXY
    else:
        cfg['httpProxy'] = PROXY
    ex = ccxt.binanceusdm(cfg)
    ex.load_markets()
    return ex


def resolve(ex, base):
    for cand in (base, '1000' + base, '1000000' + base):
        for m in ex.markets.values():
            if (m.get('swap') and m.get('linear') and m.get('base') == cand
                    and m.get('quote') == 'USDT' and m.get('settle') == 'USDT'):
                return m['symbol']
    return None


def fetch(ex, base, start, end):
    sym = resolve(ex, base)
    if not sym:
        print(f'[{base}] 无对应合约, 跳过')
        return False
    start_ms = ccxt.Exchange.parse8601(start + ' 00:00:00')
    end_ms = ccxt.Exchange.parse8601(end + ' 00:00:00')
    rows, cursor, retries = [], start_ms, 0
    while cursor < end_ms:
        try:
            r = ex.fetch_funding_rate_history(sym, since=cursor, limit=1000)
            retries = 0
        except Exception as e:
            retries += 1
            if retries >= 4:
                print(f'  [{base}] 失败: {str(e)[:60]}')
                return False
            ex.sleep(1500)
            continue
        if not r:
            break
        rows += [(x['timestamp'], x['fundingRate']) for x in r]
        nxt = r[-1]['timestamp'] + 1
        if nxt <= cursor:
            break
        cursor = nxt
        print(f'\r  [{base}] {len(rows)} 条, 到 {pd.to_datetime(r[-1]["timestamp"], unit="ms")}', end='')
    print()
    if not rows:
        print(f'[{base}] 无数据')
        return False
    df = pd.DataFrame(rows, columns=['timestamp', 'rate'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    df = df[(df['timestamp'] >= start) & (df['timestamp'] < end)]
    tag = f'{start.replace("-", "")}_{end.replace("-", "")}'
    out = os.path.join(OUT_DIR, f'funding_{base}_{tag}.csv')
    df.to_csv(out, index=False)
    ann = df['rate'].mean() * 3 * 365 * 100      # 每8h一次 -> 每天3次
    print(f'[{base}] 保存 {len(df)} 条 -> 均费率 {df["rate"].mean()*100:.4f}%/8h '
          f'(折年化 {ann:+.1f}%)  {df.timestamp.iloc[0]} ~ {df.timestamp.iloc[-1]}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bases', nargs='*')
    ap.add_argument('--start', default='2023-01-01')
    ap.add_argument('--end', default='2025-01-01')
    ap.add_argument('--universe', default=None)
    args = ap.parse_args()
    bases = [b.upper() for b in args.bases]
    if args.universe:
        with open(os.path.join(os.path.dirname(__file__), args.universe), encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    bases += [t.strip().upper() for t in line.split(',') if t.strip()]
    bases = list(dict.fromkeys(bases))
    ex = make_ex()
    ok = sum(fetch(ex, b, args.start, args.end) for b in bases)
    print(f'\n成功 {ok}/{len(bases)}')


if __name__ == '__main__':
    main()
