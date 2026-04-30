import re
import time
import aiohttp
import base64
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

from astrbot.api import logger
from astrbot.api.star import Star, Context
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Plain, Image

PLUGIN_NAME = "astrbot_plugin_sd_prompt"


class SDPromptPlugin(Star):

    def __init__(self, context: Context, config: dict):
        super().__init__(context, config)

        self._session = None

        # =========================
        # LLM配置（通用）
        # =========================
        self.api_key = config.get("API_KEY", "")

        base = config.get(
            "API_BASE",
            "https://ark.cn-beijing.volces.com/api/v3"
        )

        self.api_url = base.rstrip("/") + "/chat/completions"

        self.model = config.get(
            "MODEL_NAME",
            "doubao-seed-2-0-mini-260215"
        )

        # =========================
        # SD配置（🔥已改为可配置）
        # =========================
        self.sd_api_url = config.get(
            "SD_API_URL",
            "http://127.0.0.1:7860/sdapi/v1/txt2img"
        )

        # =========================
        # 图片目录
        # =========================
        try:
            self.img_dir = self.context.get_data_dir() / "images"
            self.img_dir.mkdir(parents=True, exist_ok=True)
        except:
            self.img_dir = Path("./images")
            self.img_dir.mkdir(exist_ok=True)

        asyncio.create_task(self.auto_cleanup())

        logger.info(f"✅ 插件启动 | 模型: {self.model}")
        logger.info(f"🖼 SD地址: {self.sd_api_url}")

    @property
    def session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # =========================
    # 文本处理
    # =========================
    def normalize(self, text):
        text = text.replace("，", ",").replace("。", ".")
        text = text.replace("（", "(").replace("）", ")")
        text = text.replace("：", ":").replace("、", ",")
        text = re.sub(r",+", ",", text)
        return text.strip()

    def split_pos_neg(self, text):
        if "--neg" in text:
            a, b = text.split("--neg", 1)
            return a.strip(), b.strip()
        return text.strip(), ""

    # =========================
    # AI翻译
    # =========================
    async def translate(self, text):

        if not self.api_key:
            return ""

        cn = re.sub(r"[^\u4e00-\u9fa5，, ]", "", text)
        if not cn:
            return ""

        try:
            async with self.session.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"将中文绘图描述转为stable diffusion英文tag，逗号分隔，只返回英文：{cn}"
                        }
                    ],
                    "temperature": 0.2
                },
                timeout=15
            ) as resp:

                if resp.status != 200:
                    logger.warning(f"AI失败: {resp.status}")
                    return ""

                data = await resp.json()

            res = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = re.sub(r"[^a-zA-Z0-9, ]", "", res).lower().strip()

            return res

        except Exception as e:
            logger.warning(f"AI异常: {e}")
            return ""

    # =========================
    # 本地兜底词库
    # =========================
    def fallback(self, text):

        mp = {
            "少女": "1girl",
            "男孩": "1boy",
            "双马尾": "twintails",
            "洛丽塔": "lolita dress",
            "教室": "classroom",
            "街道": "street",
            "可爱": "cute",
            "开心": "happy"
        }

        tags = []

        for k, v in mp.items():
            if k in text:
                tags.append(v)

        return ", ".join(tags)

    # =========================
    # Prompt构建
    # =========================
    async def build_prompt(self, raw):

        raw = self.normalize(raw)
        pos, neg = self.split_pos_neg(raw)

        # 保留英文
        user_en = re.sub(r"[\u4e00-\u9fa5]+", "", pos)

        # AI翻译
        trans = await self.translate(pos)

        if not trans:
            trans = self.fallback(pos)

        base = "masterpiece, best quality, ultra detailed, highres"

        parts = [trans, user_en, base]

        final = []
        seen = set()

        for p in parts:
            if p:
                for tag in p.split(","):
                    tag = tag.strip()
                    if tag and tag not in seen:
                        seen.add(tag)
                        final.append(tag)

        final_prompt = ", ".join(final)

        neg_base = "lowres, worst quality, blurry, bad anatomy, bad hands"
        final_neg = f"{neg}, {neg_base}" if neg else neg_base

        return final_prompt, final_neg

    # =========================
    # SD生成
    # =========================
    async def txt2img(self, prompt, negative):

        payload = {
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": 28,
            "cfg_scale": 7,
            "sampler_name": "DPM++ 2M Karras",
            "width": 768,
            "height": 1216
        }

        try:
            async with self.session.post(
                self.sd_api_url,
                json=payload,
                timeout=120
            ) as resp:

                if resp.status != 200:
                    logger.error(f"SD错误: {resp.status}")
                    return None

                data = await resp.json()

            return data.get("images", [None])[0]

        except Exception as e:
            logger.error(f"SD连接失败: {e}")
            return None

    # =========================
    # 主命令
    # =========================
    @filter.command("绘图")
    async def run(self, event: AstrMessageEvent):

        content = event.message_str.replace("绘图", "").strip()

        if not content:
            yield event.plain_result("请输入描述")
            return

        yield event.plain_result("🧠 构建提示词...")

        prompt, neg = await self.build_prompt(content)

        yield event.plain_result(prompt)

        yield event.plain_result("🎨 生成中...")

        img = await self.txt2img(prompt, neg)

        if not img:
            yield event.plain_result("❌ SD生成失败（检查SD地址/API）")
            return

        path = self.img_dir / f"{int(time.time())}.png"

        with open(path, "wb") as f:
            f.write(base64.b64decode(img))

        yield event.chain_result([
            Image.fromFileSystem(str(path)),
            Plain("完成")
        ])

    # =========================
    # 自动清理
    # =========================
    async def auto_cleanup(self):

        while True:
            await asyncio.sleep(3600)

            try:
                now = datetime.now()

                for f in self.img_dir.iterdir():
                    if f.is_file():
                        if now - datetime.fromtimestamp(f.stat().st_mtime) > timedelta(days=3):
                            f.unlink()

            except:
                pass