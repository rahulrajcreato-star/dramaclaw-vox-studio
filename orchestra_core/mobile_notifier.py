import os
import urllib.request
import urllib.parse
import json
from orchestra_core.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_alert(message: str, bot_token: str = None, chat_id: str = None) -> bool:
    """
    Sends an instant notification to the user's mobile device via Telegram Bot API.
    """
    token = bot_token or TELEGRAM_BOT_TOKEN
    target_chat = chat_id or TELEGRAM_CHAT_ID
    if not token or not target_chat:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[NOTIFIER] Telegram alert failed: {e}")
        return False
