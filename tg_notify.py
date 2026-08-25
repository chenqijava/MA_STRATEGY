"""Telegram 推送 (零依赖, 只用标准库)。

给实盘用: 每轮把状态推到群, 离开电脑时也能确认策略还活着。
本策略约 86% 的轮次没有新开仓、牛市里最长可连续空仓 23 天(analyze_idle.py),
所以"很久没消息"是正常的 —— 心跳推送就是用来区分"安静"和"挂了"。

设计取舍:
  失败绝不抛异常。推送只是观测手段, 网络抖动/TG被墙不能影响交易主流程。
  代理沿用 PROXY 环境变量 (与 ccxt 同一套), 国内直连 api.telegram.org 通常不通。

用法:
  python tg_notify.py --resolve          # 列出 bot 能看到的 chat, 用来取 chat_id
  python tg_notify.py --test             # 按 .env 的 TG_TOKEN/TG_CHAT 发一条测试消息
"""
import os
import json
import socket
import urllib.parse
import urllib.request

API = 'https://api.telegram.org'
_WARNED = False          # 只提示一次配置缺失, 避免每轮刷屏


def _opener():
    """按 PROXY 环境变量构造 opener。socks 需要 PySocks, 没有则退回直连。"""
    proxy = os.environ.get('PROXY', '')
    if not proxy:
        return urllib.request.build_opener()
    if proxy.startswith('socks'):
        try:
            import socks               # PySocks, ccxt 装了它
            from sockshandler import SocksiPyHandler
            u = urllib.parse.urlparse(proxy)
            return urllib.request.build_opener(
                SocksiPyHandler(socks.SOCKS5, u.hostname, u.port or 1080))
        except Exception:
            # 没有 sockshandler 时用底层 monkeypatch 兜底
            try:
                import socks
                u = urllib.parse.urlparse(proxy)
                socks.set_default_proxy(socks.SOCKS5, u.hostname, u.port or 1080)
                socket.socket = socks.socksocket
                return urllib.request.build_opener()
            except Exception:
                return urllib.request.build_opener()
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({'http': proxy, 'https': proxy}))


def _call(method, params, timeout=15):
    """调 TG API, 返回解析后的 dict。优先用 requests(对 socks5h 支持可靠),
    没有则退回标准库。urllib+SocksiPyHandler 在部分环境会 SSL 握手失败。"""
    proxy = os.environ.get('PROXY', '')
    try:
        import requests
        px = {'http': proxy, 'https': proxy} if proxy else None
        r = requests.post(f'{API}/bot{method}', data=params, proxies=px, timeout=timeout)
        return r.json()
    except ImportError:
        pass
    data = urllib.parse.urlencode(params).encode() if params else None
    req = urllib.request.Request(f'{API}/bot{method}', data=data)
    with _opener().open(req, timeout=timeout) as r:
        return json.load(r)


def send(text, token=None, chat=None, timeout=15, log=None):
    """推一条消息。返回 True/False, 永不抛异常。"""
    global _WARNED
    token = token or os.environ.get('TG_TOKEN', '')
    chat = chat or os.environ.get('TG_CHAT', '')
    if not token or not chat:
        if not _WARNED:
            _WARNED = True
            if log:
                log('[tg] 未配置 TG_TOKEN/TG_CHAT, 跳过推送 (仅提示一次)')
        return False
    try:
        r = _call(f'{token}/sendMessage', {
            'chat_id': chat, 'text': text,
            'parse_mode': 'HTML', 'disable_web_page_preview': 'true',
        }, timeout=timeout)
        if not r.get('ok') and log:
            log(f'[tg][warn] TG 拒绝: {r.get("description")}')
        return r.get('ok', False)
    except Exception as e:
        if log:
            log(f'[tg][warn] 推送失败(不影响交易): {str(e)[:90]}')
        return False


def resolve(token=None):
    """列出 bot 能看到的 chat。需要先把 bot 拉进群并在群里发过一条消息。"""
    token = token or os.environ.get('TG_TOKEN', '')
    if not token:
        print('缺 TG_TOKEN')
        return
    try:
        me = _call(f'{token}/getMe', None, timeout=20).get('result', {})
        print(f'bot: @{me.get("username")}  id={me.get("id")}')
        ups = _call(f'{token}/getUpdates', {'limit': 100}, timeout=20).get('result', [])
    except Exception as e:
        print(f'请求失败: {str(e)[:120]}')
        return
    seen = {}
    for x in ups:
        for k in ('message', 'channel_post', 'edited_message', 'my_chat_member'):
            m = x.get(k)
            if m and m.get('chat'):
                c = m['chat']
                seen[c['id']] = (c.get('type'),
                                 c.get('title') or c.get('username') or c.get('first_name'))
    if not seen:
        print(f'\n收到 {len(ups)} 条 update, 未发现任何 chat。请按顺序做:')
        print(f'  1. 把 @{me.get("username")} 拉进目标群 (并给发消息权限)')
        print('  2. 在群里随便发一条消息 (群里 bot 默认只能看到 @它 或命令, '
              '要么 @它一次, 要么在 BotFather 里关掉 Privacy Mode)')
        print('  3. 重新运行 python tg_notify.py --resolve')
        print('\n提示: 群 chat_id 是负数(超级群形如 -100xxxxxxxxxx)。')
        return
    print('\n发现的 chat (把要用的 id 填到 .env 的 TG_CHAT):')
    for cid, (t, n) in seen.items():
        print(f'  TG_CHAT={cid}    type={t}  name={n}')


if __name__ == '__main__':
    import sys
    # 独立运行时自己加载 .env (实盘里由 live_ma_short 负责加载)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                v = v.strip()
                if not (v.startswith('"') or v.startswith("'")):
                    for i, ch in enumerate(v):
                        if ch == '#' and i > 0 and v[i - 1] in ' \t':
                            v = v[:i]
                            break
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if '--resolve' in sys.argv:
        resolve()
    elif '--test' in sys.argv:
        ok = send('✅ <b>MA90做空策略</b> 推送测试\n配置正常, 可以收到消息。',
                  log=lambda m: print(m))
        print('发送成功' if ok else '发送失败 (检查 TG_TOKEN/TG_CHAT/PROXY)')
    else:
        print(__doc__)
