# MA_STRATEGY — MA5/MA90 死叉反弹做空 + BTC/ETH 长多填仓 (OKX 永续, 4h)

在 OKX USDT 永续上对一篮子中大市值币做**纯做空**:MA5 死叉 MA90 后,价格反弹到慢线阻力附近出现阴线时开空,ATR 极端保险 + MA 出场 + 时间止损离场,交易所侧挂 algo 委托自动平仓。

**另含 BTC/ETH 长多填仓叠加层**(当前实盘已开启 `BTC_FILL=1` / `ETH_FILL=1`):主做空仓位不足时,用闲置容量对 BTC(MA10/120 金叉一次 + 3×ATR20 trail)与 ETH(MA20/85 金叉一次 + 4×ATR18 trail)做多,不占空仓槽位、不推高总敞口,主策略对该币开空时先对冲平多。

> **策略细节、参数定案过程、三段样本外检验结果见 [策略文档.md](策略文档.md)**(权威文档)。
> 进行中的设计([设计方案.md](设计方案.md)):趋势 + 均值回归 Regime Switching 结合策略。

## 目录结构

| 路径 | 作用 |
|---|---|
| `live_ma_short.py` | 实盘执行脚本(OKX 4h,常驻轮询/`--once` 单轮;v2 集成 BTC/ETH 长多填仓) |
| `ma_pullback.py` | 策略信号核心(多空对称,可开关单腿),被实盘与回测复用 |
| `test_btc_fill.py` / `test_eth_fill.py` | 长多填仓回归验证(叠加层,默认关,`BTC_FILL`/`ETH_FILL` 开启) |
| `combo_core.py` | 单币回测核心(账户/成本/风控框架) |
| `portfolio_sim.py` / `mno_sim.py` / `monte_carlo.py` | 组合层模拟 / 多空腿模拟 / 蒙特卡洛稳健性 |
| `fetch_*.py` / `screen_universe.py` / `filter_liquidity.py` | 数据抓取、币池筛选 |
| `run_*.py` / `wf_*.py` / `test_*.py` / `validate*.py` | 回测跑批 / 参数寻优(walk-forward) / 部件测试与验证 |
| `tg_notify.py` | Telegram 推送 |
| `策略文档.md` / `设计方案.md` | 策略与设计文档 |
| `universe*.txt` / `vol24h.json` | 币池名单与成交额快照 |

## 实盘运行

```bash
# 1. 配置凭证与参数(模板 -> 复制为 .env 填真实值)
cp .env.example .env

# 2. 安全开关确认(DRY_RUN=true 只算信号不下单)
# 3. 直接跑 或 docker 跑
python live_ma_short.py --once     # 单轮(配 cron)
python live_ma_short.py            # 常驻轮询
docker compose up -d --build
```

- **凭证只从环境变量读**(`OKX_API_KEY / OKX_SECRET / OKX_PASSWORD`),绝不硬编码;`.env` 已在 `.gitignore`/`.dockerignore`,不会提交或打进镜像。
- 所有策略参数只从 `.env` 读取(见文件内注释),docker-compose 用 `env_file` 单一来源注入,避免 `environment` 与 `.env` 静默不一致。
- 本地持仓状态存 `live_state.json`(gitignored),每轮与交易所实际持仓核对,不一致以交易所为准。

## 回测 / 研究

数据脚本把 OKX K线缓存到 `data/`(gitignored),回测脚本按需读入。成本模型统一 taker 0.05% + 滑点,绩效指标含年化/夏普/回撤/胜率/盈亏比/Alpha-Beta。

## 安全说明

`DRY_RUN` 默认 `true`——只算信号不下单。确认无误后手动改为 `false` 才会真实下单;`OKX_DEMO=true` 可先打到模拟盘验证下单链路。

---

最后更新:2026-08-25(初始提交)。文档状态:策略参数已定案,模拟盘完整滑点校准尚未完成(仅 2 条真实成交样本)。
