"""把两段 4h 数据拼成一段连续数据 (20230101_20260801), 供跨段连续回测用。

为什么要拼: 两段分别回测再把年化相乘是错的 —— 第二段起始权益应是第一段的期末,
而且跨段边界上的持仓会被强制截断/重置。拼成一条连续序列再跑一次才是真年化。

边界已核对无重叠: 段一末根 2024-12-31 20:00, 段二首根 2025-01-01 00:00。
"""
import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
A, B, OUT = '20230101_20250101', '20250101_20260801', '20230101_20260801'
TF = '4h'


def main():
    made = 0
    for pa in sorted(glob.glob(os.path.join(DATA, f'*-USDT-SWAP_{TF}_{A}.csv'))):
        sym = os.path.basename(pa).split(f'_{TF}_')[0]
        pb = os.path.join(DATA, f'{sym}_{TF}_{B}.csv')
        if not os.path.exists(pb):
            print(f'  跳过 {sym}: 缺第二段')
            continue
        da = pd.read_csv(pa, parse_dates=['timestamp'])
        db = pd.read_csv(pb, parse_dates=['timestamp'])
        m = (pd.concat([da, db], ignore_index=True)
               .drop_duplicates('timestamp')
               .sort_values('timestamp'))
        gap = (m['timestamp'].diff().dt.total_seconds() / 3600).max()
        m.to_csv(os.path.join(DATA, f'{sym}_{TF}_{OUT}.csv'), index=False)
        made += 1
        if gap > 4:
            print(f'  {sym}: {len(m)}根 最大间隔 {gap:.0f}h (有缺K, 该币在此期间可能未上线)')
    print(f'\n完成: {made} 个币 -> *_{TF}_{OUT}.csv')


if __name__ == '__main__':
    main()
