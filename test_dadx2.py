"""ADX 过滤的关键追问: 单笔质量提升是真的吗? 提杠杆能不能把收益补回来?

第一轮发现 ADX>25 让 PF 从 1.29/1.33/1.55 升到 1.49/1.31/1.96, 胜率 35.2%->39.0%,
夏普基本不变但回撤显著小(24.6%->12.1%)。这与日线MA过滤(PF 反而下降)本质不同。

但敞口只有基线 0.09~0.17x (基线 0.21~0.39x), 只剩 4 成。所以要问:
  ① PF/胜率提升是不是纯统计噪声? -> 看笔数与单笔 R 的 t 值
  ② 按同等回撤(而非同等敞口)放大杠杆, 谁的收益更高? -> 回撤对齐才是风控口径下的公平比较
  ③ 提杠杆到多少会爆? -> 单段最差回撤决定杠杆上限
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import importlib
import numpy as np
import pandas as pd
import analyze_long as al
import portfolio_sim as ps

SPANS = [('20210101_20230101', '2021-01~2023-01'),
         ('20230101_20250101', '2023-01~2025-01'),
         ('20250101_20260801', '2025-01~2026-08')]
N = int(os.environ.get('N', '20'))
BASE_ENV = dict(MA_SLOW='90', MA_EXIT='1', COOLDOWN_ONCE='1',
                ATR_TP='10', ATR_SL='10', SIZING='notional',
                CROSS_DELAY='6', TOUCH_ATR='0.5', MAX_BARS='40',
                UNIVERSE='universe_live.txt', TOTAL='1.5')
_RESET = ('D_FILTER', 'D_MODE', 'D_MA', 'D_SLOPE', 'D_ADX_LEN', 'D_ADX_TH',
          'D_INVERT', 'TOTAL')
ADX = dict(D_FILTER='1', D_MODE='adx', D_ADX_LEN='14', D_ADX_TH='25')


def run(span, env):
    os.environ['SPAN'] = span
    for k in _RESET:
        os.environ.pop(k, None)
    for k, v in {**BASE_ENV, **env}.items():
        os.environ[k] = str(v)
    importlib.reload(al)
    importlib.reload(ps)
    idx, C, H, L, A, S, syms = ps.build_panel()
    curve, trades, avg_held = ps.simulate(N, idx, C, H, L, A, S)
    m = ps.metrics(curve, trades, idx)
    tr = np.array(trades, float)
    w, l = tr[tr > 0].sum(), -tr[tr < 0].sum()
    m['pf'] = w / l if l > 0 else float('inf')
    m['held'] = avg_held
    m['expo'] = avg_held * ps.TOTAL_NOTIONAL / N
    # 单笔盈亏的 t 值: 均值显著大于 0 吗 (笔数少时 PF 高可能只是噪声)
    m['t'] = (tr.mean() / (tr.std(ddof=1) / np.sqrt(len(tr)))
              if len(tr) > 1 and tr.std(ddof=1) > 0 else 0)
    m['avg'] = tr.mean() if len(tr) else 0
    return m


def part1():
    print('=' * 100)
    print('① 单笔质量: PF/胜率的提升是否显著 (笔数少 -> PF 高也可能是噪声)')
    print('=' * 100)
    for span, lab in SPANS:
        print(f'\n### {lab}')
        print(f'{"配置":22s}{"笔数":>7s}{"胜率":>8s}{"PF":>7s}{"单笔均":>9s}{"单笔t值":>9s}')
        print('-' * 70)
        for name, env in (('基线', dict(D_FILTER='0')), ('ADX>25', dict(ADX))):
            m = run(span, env)
            print(f'{name:22s}{m["trades"]:7d}{m["win"]:7.1f}%{m["pf"]:7.2f}'
                  f'{m["avg"]:+9.1f}{m["t"]:9.2f}')


def part2():
    """回撤对齐: 把两者都放大到"三段最差回撤 ~= 24%"再比收益。"""
    print('\n' + '=' * 100)
    print('② 回撤对齐 (风控口径的公平比较): 都放大到三段最差回撤接近 24~25%')
    print('=' * 100)
    cases = [('基线 1.5x', dict(D_FILTER='0'))]
    for tot in ('1.5', '2.5', '3.0', '3.5', '4.0'):
        cases.append((f'ADX>25 {tot}x', dict(ADX, TOTAL=tot)))
    store = {}
    for span, lab in SPANS:
        for name, env in cases:
            store.setdefault(name, []).append(run(span, env))
    print(f'{"配置":18s}{"三段年化":>28s}{"最差年化":>10s}{"最大回撤":>10s}'
          f'{"最差夏普":>10s}{"均敞口":>9s}')
    print('-' * 100)
    for name, _ in cases:
        rs = store[name]
        anns = [m['ann'] for m in rs]
        s = '  '.join(f'{v:+7.1f}%' for v in anns)
        print(f'{name:18s}{s:>28s}{min(anns):+9.1f}%{max(m["mdd"] for m in rs):9.1f}%'
              f'{min(m["sharpe"] for m in rs):10.2f}{np.mean([m["expo"] for m in rs]):8.2f}x')


def main():
    print(f'ADX 过滤深入检验  N={N}  MA90  币池=universe_live\n')
    part1()
    part2()


if __name__ == '__main__':
    main()
