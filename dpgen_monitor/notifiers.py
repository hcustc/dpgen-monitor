from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import requests
from requests_toolbelt import MultipartEncoder

from .config import NotificationConfig
from .events import MonitorEvent


class Notifier(Protocol):
    name: str

    def send(self, event: MonitorEvent) -> None:
        ...


def _option_or_env(options: dict, key: str, env_key: str) -> str:
    env_name = options.get(env_key)
    if env_name:
        value = os.environ.get(str(env_name))
        if not value:
            raise ValueError(f"环境变量 {env_name} 未设置")
        return value
    value = options.get(key)
    if not value:
        raise ValueError(f"通知器缺少 {key} 或 {env_key}")
    return str(value)


class ConsoleNotifier:
    def __init__(self, name: str):
        self.name = name

    def send(self, event: MonitorEvent) -> None:
        print(f"[通知:{self.name}] {event.title}")
        print(event.message)
        for image_path in event.image_paths:
            print(f"  image: {image_path}")


class GenericWebhookNotifier:
    """JSON webhook adapter suitable for custom services and automation tools."""

    def __init__(self, config: NotificationConfig):
        self.name = config.name
        self.url = _option_or_env(config.options, "url", "url_env")
        self.timeout = int(config.options.get("timeout", 15))
        self.headers = {
            "Content-Type": "application/json",
            **dict(config.options.get("headers", {})),
        }

    def send(self, event: MonitorEvent) -> None:
        data = {
            "event_key": event.key,
            "event_type": event.event_type,
            "title": event.title,
            "message": event.message,
            "iteration": event.iteration,
            "images": [str(path) for path in event.image_paths],
            "payload": event.payload,
        }
        response = requests.post(
            self.url,
            headers=self.headers,
            json=data,
            timeout=self.timeout,
        )
        response.raise_for_status()


class FeishuNotifier:
    """Feishu app image upload plus custom bot delivery adapter."""

    TOKEN_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    IMAGE_URL = "https://open.feishu.cn/open-apis/im/v1/images"

    def __init__(self, config: NotificationConfig):
        self.name = config.name
        self.bot_url = _option_or_env(config.options, "bot_url", "bot_url_env")
        self.app_id = _option_or_env(config.options, "app_id", "app_id_env")
        self.app_secret = _option_or_env(
            config.options, "app_secret", "app_secret_env"
        )
        self.timeout = int(config.options.get("timeout", 20))

    def _token(self) -> str:
        response = requests.post(
            self.TOKEN_URL,
            headers={"Content-Type": "application/json; charset=utf-8"},
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"获取飞书 token 失败: {payload}")
        return str(payload["tenant_access_token"])

    def _upload_image(self, token: str, path: Path) -> str:
        with path.open("rb") as image_file:
            form = {
                "image_type": "message",
                "image": (path.name, image_file, "image/png"),
            }
            multipart = MultipartEncoder(form)
            response = requests.post(
                self.IMAGE_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": multipart.content_type,
                },
                data=multipart,
                timeout=self.timeout,
            )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"上传飞书图片失败: {payload}")
        return str(payload["data"]["image_key"])

    def send(self, event: MonitorEvent) -> None:
        content = [[{"tag": "text", "text": event.message}]]
        if event.image_paths:
            token = self._token()
            for path in event.image_paths:
                if path.is_file():
                    image_key = self._upload_image(token, path)
                    content.append([{"tag": "img", "image_key": image_key}])
        data = {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": event.title,
                        "content": content,
                    }
                }
            },
        }
        response = requests.post(
            self.bot_url,
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        result_code = payload.get("code", payload.get("StatusCode"))
        if result_code is not None and result_code != 0:
            raise RuntimeError(f"飞书机器人返回失败: {payload}")


def build_notifier(config: NotificationConfig) -> Notifier:
    notifier_type = config.type.lower()
    if notifier_type == "console":
        return ConsoleNotifier(config.name)
    if notifier_type == "feishu":
        return FeishuNotifier(config)
    if notifier_type in {"webhook", "generic_webhook"}:
        return GenericWebhookNotifier(config)
    raise ValueError(f"未知通知器类型: {config.type}")
