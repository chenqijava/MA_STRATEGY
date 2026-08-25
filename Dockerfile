# MA5/MA24 死叉反弹做空 实盘执行 (OKX 永续)
FROM python:3.9-slim

# 时区: OKX K线用 UTC, 日志时间用东八区便于对照; 按需改
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# 先装依赖 (利用层缓存)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 策略代码不打进镜像: 由 docker-compose 把项目目录挂到 /app (改代码免重新 build)。
# live 依赖 ma_pullback, 后者依赖 combo_core, 均从挂载目录读取。

# 状态文件默认写 /data (docker-compose 挂卷持久化)
ENV STATE_FILE=/data/live_state.json
VOLUME ["/data"]

# 默认常驻轮询; 传 --once 可只跑一轮 (配合外部 cron/定时器)
ENTRYPOINT ["python", "live_ma_short.py"]
