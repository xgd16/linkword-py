"""链接 AI 生成相关 API"""
import os
import re
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.core.config import load_config
from src.core.logger import logger

router = APIRouter(prefix="/api/link", tags=["link"])


class AiFillRequest(BaseModel):
    url: str


class AiFillResponse(BaseModel):
    title: str
    icon: str
    slogan: str
    description: str


class _AiFillOutput(BaseModel):
    """AI 输出结构，供 PydanticOutputParser 解析用"""

    title: str = Field(
        description="显示在卡片上的名称，8-32字，简洁明了，适合卡片展示"
    )
    icon: str = Field(
        description="Remix Icon 类名，必须以 ri- 开头，如 ri-gitlab-fill、ri-chrome-fill、ri-code-s-slash-fill"
    )
    slogan: str = Field(
        description="核心功能标语，10-20字，精炼概括该网站核心功能，适合放在卡片箭头位置"
    )
    description: str = Field(
        description="详细描述，150-400字，支持 Markdown：**粗体**、- 列表、`代码`、## 小标题，需包含核心功能、适用场景、主要优势"
    )


def _get_llm() -> ChatOpenAI:
    """从配置创建 OpenAI 兼容的 LLM"""
    api_key = os.environ.get("AI_API_KEY") or os.environ.get("DMXAPI_API_KEY")
    cfg = load_config()
    dmx = cfg.get("ai", {}).get("dmxapi", {})
    if not api_key:
        api_key = dmx.get("apiKey", "")
    if not api_key or not str(api_key).strip():
        raise HTTPException(
            status_code=500,
            detail="请配置 AI_API_KEY 或 DMXAPI_API_KEY 环境变量，或在 config.yaml 中设置 ai.dmxapi.apiKey",
        )
    base_url = dmx.get("baseUrl", "https://www.dmxapi.cn/v1")
    model = dmx.get("model", "gpt-4o-mini")
    return ChatOpenAI(
        api_key=str(api_key).strip(),
        base_url=base_url,
        model=model,
        temperature=0.7,
    )


def _fetch_page_info(url: str) -> dict[str, Any]:
    """获取页面基本信息（标题、meta 等）"""
    info: dict[str, Any] = {"title": "", "snippet": ""}
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            text = resp.text
            # 提取 <title>
            m = re.search(r"<title[^>]*>([^<]*)</title>", text, re.I | re.S)
            if m:
                info["title"] = re.sub(r"\s+", " ", m.group(1).strip())[:200]
            # 提取 meta description
            m = re.search(
                r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']',
                text,
                re.I,
            )
            if not m:
                m = re.search(
                    r'<meta[^>]+content=["\']([^"\']*)["\'][^>]+name=["\']description["\']',
                    text,
                    re.I,
                )
            if m:
                info["snippet"] = re.sub(r"\s+", " ", m.group(1).strip())[:500]
    except Exception as e:
        logger.warning("获取页面信息失败 %s: %s", url, e)
    return info


@router.post("/ai-fill", response_model=AiFillResponse)
async def ai_fill(req: AiFillRequest) -> AiFillResponse:
    """
    根据链接地址，使用 AI 生成标题、图标、核心标语、详细描述。
    供人工审核后发布。
    """
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="请输入链接地址")
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="链接格式不正确")

    page_info = _fetch_page_info(url)
    page_title = page_info.get("title") or ""
    page_snippet = page_info.get("snippet") or ""

    parser = PydanticOutputParser(pydantic_object=_AiFillOutput)
    format_instructions = parser.get_format_instructions()

    llm = _get_llm()
    prompt = f"""你是一个导航网站的文案编辑。根据以下网站信息，生成导航卡片所需的四段内容。

链接地址：{url}
网页标题：{page_title or '(未获取到)'}
网页描述：{page_snippet or '(未获取到)'}

{format_instructions}

要求：
- title 简短有力，适合卡片展示
- icon 必须是 Remix Icon 的类名格式，以 ri- 开头，根据网站类型选择最合适的，如 ri-github-fill、ri-chrome-fill
- slogan 适合放在卡片箭头位置
- description 使用 Markdown：**粗体**、- 列表、`代码`、## 小标题"""

    try:
        msg = llm.invoke(prompt)
        text = (msg.content or "").strip()
    except Exception as e:
        logger.exception("AI 调用失败")
        raise HTTPException(status_code=500, detail=f"AI 服务异常: {str(e)}") from e

    try:
        parsed = parser.parse(text)
        title = (parsed.title or "").strip()[:128] or "未知站点"
        icon = (parsed.icon or "").strip()
        if not icon.startswith("ri-"):
            icon = "ri-link"
        slogan = (parsed.slogan or "").strip()[:128] or "导航链接"
        description = (parsed.description or "").strip()[:1024] or "暂无描述"
        return AiFillResponse(title=title, icon=icon, slogan=slogan, description=description)
    except Exception as e:
        logger.warning("AI 返回解析失败: %s", text[:200], exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI 返回解析失败: {str(e)}") from e
