import requests

from api.config import config

BASE_URL = "https://discord.com/api/v10"

CHANNEL_ID = "1317046649494962246"


def send_message(message: str):
    r = requests.post(
        f"{BASE_URL}/channels/{CHANNEL_ID}/messages",
        headers={
            "Contnet-Type": "application/json",
            "Authorization": f"Bot {config.discord_token}",
        },
        json={"content": message},
    )
    data = r.json()
    return data
