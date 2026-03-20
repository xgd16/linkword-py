"""
配图生成服务 - 调用 dmxapi /v1/images/generations（OpenAI 兼容）

不同模型有不同尺寸要求：
- dall-e-3 / dall-e-2: 1024x1024
- doubao-seedream 系列: 总像素至少 3686400，默认 1920x1920

生成后下载并转 webp 保存到配置目录，返回 /upload/covers/xxx.webp
"""
import time
import uuid
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from src.core.config import load_config, ROOT_DIR
from src.core.logger import logger

# 各模型合规的 size，豆包 seedream 要求总像素至少 3686400
IMAGE_SIZE_BY_MODEL = {
    "dall-e-3": "1024x1024",
    "dall-e-2": "1024x1024",
    "doubao-seedream-4-5-251128": "1920x1920",
    "doubao-seedream-5-0-lite": "1920x1920",
}


def _get_image_size(model: str) -> str:
    """根据模型名称返回合规的 size 字符串"""
    model_lower = (model or "").lower().strip()
    for key, size in IMAGE_SIZE_BY_MODEL.items():
        if key in model_lower:
            return size
    if "seedream" in model_lower or "doubao" in model_lower:
        return "1920x1920"
    return "1024x1024"


def _get_image_gen_config() -> dict:
    cfg = load_config()
    return cfg.get("ai", {}).get("imageGen", {})


def _get_api_key() -> str:
    import os
    key = os.environ.get("AI_API_KEY") or os.environ.get("DMXAPI_API_KEY")
    if key:
        return str(key).strip()
    cfg = load_config()
    return str(cfg.get("ai", {}).get("dmxapi", {}).get("apiKey", "")).strip()


def _get_base_url() -> str:
    cfg = load_config()
    base = cfg.get("ai", {}).get("dmxapi", {}).get("baseUrl", "https://www.dmxapi.cn/v1")
    return str(base).rstrip("/")


def _resolve_output_dir(output_dir_raw: str) -> Path:
    out_path = Path(output_dir_raw)
    if not out_path.is_absolute():
        out_path = (ROOT_DIR / out_path).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def _generate_and_save_webp(
    prompt: str,
    *,
    out_path: Path,
    public_url_prefix: str,
    size: str | None = None,
) -> str | None:
    """
    调用 images/generations，下载并保存为 webp。
    public_url_prefix 如 /upload/covers、/upload/banners（对应 Go static /upload 下子目录）
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning("图像生成：未配置 AI_API_KEY / dmxapi.apiKey，跳过配图生成")
        return None

    img_cfg = _get_image_gen_config()
    model = img_cfg.get("model", "dall-e-3")
    size = size or _get_image_size(model)

    url = f"{_get_base_url()}/images/generations"
    payload = {
        "model": model,
        "prompt": prompt[:4000],
        "n": 1,
        "size": size,
        "response_format": "url",
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.RequestError as e:
        logger.warning("配图生成请求失败，跳过配图: %s", e)
        return None

    if resp.status_code != 200:
        logger.warning("配图生成失败 HTTP %s: %s", resp.status_code, resp.text[:500])
        return None

    data = resp.json()
    items = data.get("data") or []
    if not items:
        logger.warning("配图生成返回无数据: %s", data)
        return None

    img_url = items[0].get("url")
    if not img_url:
        return None

    logger.info("配图生成成功，正在下载保存: %s", img_url[:80])

    try:
        with httpx.Client(timeout=60.0) as client:
            dl_resp = client.get(img_url)
        if dl_resp.status_code != 200:
            logger.warning("下载配图失败 HTTP %s: %s", dl_resp.status_code, img_url[:80])
            return None
        img_bytes = dl_resp.content
    except Exception as e:
        logger.warning("下载配图失败: %s", e)
        return None

    suffix = uuid.uuid4().hex[:8]
    filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix}.webp"
    dest = out_path / filename
    try:
        img = Image.open(BytesIO(img_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(dest, "WEBP", quality=85)
    except Exception as e:
        logger.warning("图片转 webp 保存失败: %s", e)
        return None

    prefix = public_url_prefix.rstrip("/")
    return f"{prefix}/{filename}"


def generate_and_save_cover(prompt: str, *, size: str | None = None) -> str | None:
    """
    生成配图，下载后转 webp 保存到配置目录。

    :param prompt: 图像描述提示词（建议 80-400 字）
    :param size: 可选，不填则按模型自动选择合规尺寸
    :return: 路径如 "/upload/covers/xxx.webp"，失败返回 None
    """
    img_cfg = _get_image_gen_config()
    output_dir_raw = img_cfg.get("outputDir", "uploads/covers")
    out_path = _resolve_output_dir(output_dir_raw)
    return _generate_and_save_webp(
        prompt, out_path=out_path, public_url_prefix="/upload/covers", size=size
    )


def generate_and_save_banner(prompt: str, *, size: str | None = None) -> str | None:
    """
    生成轮播图用横版配图，保存到 bannerOutputDir（默认同级目录 banners）。

    :return: 路径如 "/upload/banners/xxx.webp"，失败返回 None
    """
    img_cfg = _get_image_gen_config()
    banner_raw = (img_cfg.get("bannerOutputDir") or "").strip()
    if not banner_raw:
        covers_raw = img_cfg.get("outputDir", "uploads/covers")
        covers_path = _resolve_output_dir(covers_raw)
        out_path = covers_path.parent / "banners"
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = _resolve_output_dir(banner_raw)
    if size is None:
        model = str(img_cfg.get("model", ""))
        size = _get_image_size(model)
        # DALL·E 3 支持宽幅，更适合轮播首屏
        if "dall-e-3" in model.lower() and "dall-e-2" not in model.lower():
            size = "1792x1024"
    return _generate_and_save_webp(
        prompt, out_path=out_path, public_url_prefix="/upload/banners", size=size
    )
