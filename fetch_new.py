"""拉取指定币种的 4h (可选1h) K线到 data_cache, 供回测新增币验证。

复用 fetch_htf 的分页/重试逻辑, 支持多币。代理默认 127.0.0.1:10808 (与 fetch_htf 一致),
可用环境变量 PROXY 覆盖。列名对齐现有缓存: timestamp,open,high,low,close,vol。

用法:
  python fetch_new.py SUI NEAR ARB OP INJ TIA          # 默认拉 4h
  python fetch_new.py SUI --tf 1h 4h                   # 指定周期
"""
import os, sys, argparse
import ccxt
import pandas as pd

PROXY = os.environ.get('PROXY') or 'socks5h://127.0.0.1:10808'
# 可用环境变量覆盖区间, 输出文件名自动跟随区间, 便于并存多段样本外数据
START_DATE = os.environ.get('START_DATE') or '2025-01-01 00:00:00'
END_DATE = os.environ.get('END_DATE') or '2026-08-01 00:00:00'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def tag(d):
    return d[:10].replace('-', '')


def make_ex():
    cfg = {'enableRateLimit': True, 'options': {'defaultType': 'swap'}, 'timeout': 20000}
    if PROXY.startswith('socks'):
        cfg['socksProxy'] = PROXY
    elif PROXY:
        cfg['httpProxy'] = PROXY
    ex = ccxt.okx(cfg)
    # ccxt 4.x 实测: httpProxy 之外还需 session.proxies, 否则同步 requests 通道直连
    # 解析 DNS 失败 (getaddrinfo)。见 live_ma_short.make_exchange 同款修复。
    if PROXY and not PROXY.startswith('socks'):
        ex.session.proxies = {'http': PROXY, 'https': PROXY}
    ex.load_markets()
    return ex


def resolve(ex, base):
    """按 base 币找线性 USDT 永续 ccxt 符号; 找不到返回 None。"""
    for m in ex.markets.values():
        if (m.get('swap') and m.get('linear') and m.get('base') == base
                and m.get('quote') == 'USDT' and m.get('settle') == 'USDT' and m.get('active', True)):
            return m['symbol']
    return None


def fetch(ex, base, tf):
    sym = resolve(ex, base)
    if not sym:
        print(f'[{base}] OKX 无对应 USDT 永续, 跳过')
        return False
    start_ms = ex.parse8601(START_DATE)
    end_ms = ex.parse8601(END_DATE)
    # 从最新往旧翻页: OKX 的 since-forward 分页对部分币(合约历史早于窗口)会直接返回空,
    # 而 history-candles 端点 + after=游标 从新回溯到目标起点对所有币都可靠。
    all_data, retries, cursor = [], 0, end_ms
    while True:
        try:
            o = ex.fetch_ohlcv(sym, tf, limit=100,
                               params={'method': 'publicGetMarketHistoryCandles', 'after': cursor})
            retries = 0
        except Exception as e:
            retries += 1
            print(f'\n[{base} {tf}] 出错: {str(e)[:60]} 第{retries}次重试')
            if retries >= 5:
                break
            ex.sleep(2000)
            continue
        if not o:
            break
        all_data = o + all_data
        earliest = o[0][0]
        if earliest >= cursor:         # 游标没前进 -> 到头了, 防死循环
            break
        cursor = earliest              # 下一页取更早的
        print(f'\r[{base} {tf}] 已拉 {len(all_data)} 根, 最早 {pd.to_datetime(earliest, unit="ms")}', end='')
        if earliest <= start_ms:
            break                      # 已回溯到目标起点
    print()
    if not all_data:
        print(f'[{base} {tf}] 无数据')
        return False
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    # 过滤起点往前多留 7 天 (28 根 4h) 作为重叠缓冲, 避免新旧文件拼接时因边界丢棒:
    # 旧文件到 08-31 04:00, 新文件若严格从 09-01 00:00 开始, 中间 08-31 08:00~20:00
    # 四根就被 >=START_DATE 过滤掉, 形成 16h 空洞 (会让 MA90 计算错位)。
    # 多留的缓冲在拼合时由 drop_duplicates 去重, 不影响结果。
    filter_start = pd.Timestamp(START_DATE) - pd.Timedelta(days=7)
    df = df[(df['timestamp'] >= filter_start) & (df['timestamp'] < END_DATE)]
    out = os.path.join(OUT_DIR, f'{base}-USDT-SWAP_{tf}_{tag(START_DATE)}_{tag(END_DATE)}.csv')
    df.to_csv(out, index=False)
    print(f'[{base} {tf}] 保存 {len(df)} 根 -> {out}  区间 {df.timestamp.iloc[0]} ~ {df.timestamp.iloc[-1]}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bases', nargs='+', help='币种, 如 SUI NEAR ARB')
    ap.add_argument('--tf', nargs='+', default=['4h'], help='周期, 默认 4h')
    args = ap.parse_args()
    ex = make_ex()
    for base in args.bases:
        for tf in args.tf:
            fetch(ex, base.upper(), tf)


if __name__ == '__main__':
    main()
