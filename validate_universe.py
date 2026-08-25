"""对 universe.txt 全候选池逐币做空腿验证, 输出通过名单。

口径复用 analyze_long 的 R倍数做空。纳入标准(用户定): 样本外 R>0 且 样本外交易 >= MIN_OOS_TR。
额外标注每币数据起始日(次新币历史不足会体现为样本外笔数少)。
"""
import os, sys
import numpy as np
import pandas as pd
import ma_pullback as mp
from analyze_long import BASE, path, norm_trades, full_span
if sys.stdout.encoding and not sys.stdout.encoding.lower().startswith('utf'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

OLD = ['BTC','ETH','SOL','BNB','XRP','DOGE','ADA','AVAX','LINK','LTC','DOT','TRX','ATOM','UNI','FIL','APT']
MIN_OOS_TR = int(os.environ.get('MIN_OOS_TR', 15))


def load_universe():
    p = os.path.join(os.path.dirname(__file__), 'universe.txt')
    return open(p, encoding='utf-8').read().strip().split(',')


def short_pnls(sym, start=None, end=None):
    pt = path(sym)
    if not pt:
        return None
    df = mp.cc.load_data(pt)
    if start is not None:
        df = df[(df.index >= start) & (df.index < end)]
    return norm_trades(df, dict(BASE, ENABLE_LONG=False, ENABLE_SHORT=True))


def agg(pnls):
    arr = np.array(pnls or [], float)
    n = len(arr)
    if n < 2 or arr.std(ddof=1) == 0:
        return dict(n=n, sum=float(arr.sum()) if n else 0.0, t=0.0,
                    win=float((arr > 0).mean()*100) if n else 0.0)
    return dict(n=n, sum=float(arr.sum()), t=float(arr.mean()/(arr.std(ddof=1)/np.sqrt(n))),
                win=float((arr > 0).mean()*100))


def start_of(sym):
    pt = path(sym)
    if not pt:
        return None
    return mp.cc.load_data(pt).index[0]


def main():
    pd.set_option('display.width', 240)
    pd.set_option('display.unicode.east_asian_width', True)
    uni = load_universe()
    lo, mid, hi = full_span()
    hi_end = hi + pd.Timedelta(hours=1)
    print('='*100)
    print(f'全候选池做空腿验证 ({len(uni)}币, 4h, 切分点={mid.date()})  纳入: 样本外R>0 且 样本外≥{MIN_OOS_TR}笔')
    print('='*100)

    rows = []
    for s in uni:
        if not path(s):
            rows.append(dict(币=s, 起始='无数据', 全R=None, 外笔=0, 外R=None, 外t=None, 判定='无数据'))
            continue
        st = start_of(s)
        full = agg(short_pnls(s))
        oos = agg(short_pnls(s, mid, hi_end))
        # 短历史币(数据起始晚于切分点): 无样本内外可分, 全样本即"样本外", 验证偏弱
        short_hist = st is not None and st > pd.Timestamp('2025-02-01', tz=st.tz)
        ok = oos['sum'] > 0 and oos['n'] >= MIN_OOS_TR
        if ok:
            verdict = '纳入(短史)' if short_hist else '纳入'
        elif short_hist and oos['n'] < MIN_OOS_TR:
            verdict = '短史样本不足'
        else:
            verdict = '不纳入'
        rows.append(dict(币=s, 起始=str(st.date()) if st is not None else '?',
                         全R=round(full['sum'], 2), 外笔=oos['n'], 外R=round(oos['sum'], 2),
                         外t=round(oos['t'], 2), 判定=verdict))
    dfp = pd.DataFrame(rows)
    keep_strong = [r['币'] for r in rows if r['判定'] == '纳入']          # 完整历史, 强验证
    keep_short = [r['币'] for r in rows if r['判定'] == '纳入(短史)']     # 短历史, 弱验证
    keep = keep_strong + keep_short

    print('\n### 全部候选(按样本外R降序)')
    show = dfp.copy()
    show['_k'] = show['外R'].fillna(-1e9)
    print(show.sort_values('_k', ascending=False).drop(columns='_k').to_string(index=False))

    print(f'\n强验证纳入(完整历史, {len(keep_strong)}): {",".join(keep_strong)}')
    print(f'弱验证纳入(短历史仅~8月, {len(keep_short)}): {",".join(keep_short)}')
    news = [k for k in keep_strong if k not in OLD]
    print(f'强验证中新增(不含原16币, {len(news)}): {",".join(news)}')

    # 整体 t 对比
    def pool(syms, a, b):
        out = []
        for s in syms:
            p = short_pnls(s, a, b)
            if p:
                out += p
        return out
    for tag, a, b in [('全样本', None, None), ('样本外', mid, hi_end)]:
        b16 = agg(pool(OLD, a, b))
        bs = agg(pool(keep_strong, a, b))
        ball = agg(pool(keep, a, b))
        print(f'[{tag}] 原16币 t={b16["t"]:.2f}(R={b16["sum"]:.0f}) | '
              f'强验证池{len(keep_strong)}币 t={bs["t"]:.2f}(R={bs["sum"]:.0f}) | '
              f'全通过池{len(keep)}币 t={ball["t"]:.2f}(R={ball["sum"]:.0f})')

    d = os.path.dirname(__file__)
    with open(os.path.join(d, 'universe_pass.txt'), 'w', encoding='utf-8') as f:
        f.write(','.join(keep_strong))          # 默认名单只含强验证币
    with open(os.path.join(d, 'universe_pass_all.txt'), 'w', encoding='utf-8') as f:
        f.write(','.join(keep))                 # 含短历史币的完整通过名单
    print(f'\n已写 universe_pass.txt (强验证 {len(keep_strong)}币) 与 universe_pass_all.txt (全 {len(keep)}币)')


if __name__ == '__main__':
    main()
