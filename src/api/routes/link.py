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
from src.services.internet_search import (
    get_web_search_settings,
    normalize_model_json_output,
    run_llm_with_internet_search,
)

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


def _build_llm(
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    timeout: int = 600,
) -> ChatOpenAI:
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
    kwargs: dict = {
        "api_key": str(api_key).strip(),
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": 1,
    }
    if max_tokens is not None:
        kwargs["model_kwargs"] = {"max_tokens": max_tokens}
    return ChatOpenAI(**kwargs)


def _get_llm() -> ChatOpenAI:
    return _build_llm()


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
    ws = get_web_search_settings()
    system_prompt = """你是导航网站的文案编辑。
当网页标题或描述缺失、含糊、可能过时，或你需要核实该网站的真实用途、产品线、所属公司、是否仍维护时，请使用 internet_search；仅凭已有信息已能准确写卡片时不要搜索。
最终回复必须且只能是一段符合用户消息中格式说明的 JSON，不要添加 Markdown 代码围栏或其它说明文字。"""

    user_prompt = f"""根据以下网站信息，生成导航卡片所需的四段内容。

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
        text = run_llm_with_internet_search(
            llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enabled=ws["enabled"],
            max_results=ws["maxResults"],
            max_tool_rounds=ws["maxToolRounds"],
        )
        text = normalize_model_json_output(text)
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


class LinkAiTranslateRequest(BaseModel):
    title: str = ""
    slogan: str = ""
    description: str = ""


class LinkAiTranslateResponse(BaseModel):
    titleEn: str = ""
    sloganEn: str = ""
    descriptionEn: str = ""


class _LinkTranslateShortOutput(BaseModel):
    titleEn: str = Field(description="中文标题的英文翻译；若源标题为空则返回空字符串")
    sloganEn: str = Field(description="中文核心标语的英文翻译；若源标语为空则返回空字符串")


@router.post("/ai-translate", response_model=LinkAiTranslateResponse)
async def link_ai_translate(req: LinkAiTranslateRequest) -> LinkAiTranslateResponse:
    """
    将导航链接的中文标题、核心标语、Markdown 详细描述译为英文，供后台英文字段使用。
    """
    title = (req.title or "").strip()
    slogan = (req.slogan or "").strip()
    description = (req.description or "").strip()
    if not title and not slogan and not description:
        raise HTTPException(status_code=400, detail="请至少填写中文标题、核心标语或详细描述中的一项")

    title_en = ""
    slogan_en = ""
    description_en = ""

    if title or slogan:
        parser = PydanticOutputParser(pydantic_object=_LinkTranslateShortOutput)
        format_instructions = parser.get_format_instructions()
        llm = _build_llm(temperature=0.25, max_tokens=2048, timeout=600)
        prompt = f"""你是专业中英翻译。将下列导航链接卡片字段译为自然、地道的英文，适合产品与工具导航站。
若某项源文本为空或标记为 (none)，对应英文字段必须为空字符串，不要编造。

源标题：{title if title else "(none)"}
源核心标语：{slogan if slogan else "(none)"}

{format_instructions}
"""
        try:
            msg = llm.invoke(prompt)
            text = (msg.content or "").strip()
            parsed = parser.parse(text)
            if title:
                title_en = (parsed.titleEn or "").strip()[:128]
            if slogan:
                slogan_en = (parsed.sloganEn or "").strip()[:128]
        except Exception as e:
            logger.exception("标题/标语翻译失败")
            raise HTTPException(status_code=500, detail=f"标题/标语翻译失败: {str(e)}") from e

    if description:
        llm = _build_llm(temperature=0.25, max_tokens=16384, timeout=600)
        prompt = f"""You are a professional translator. Translate the following Chinese Markdown into natural English.
Preserve Markdown structure (headings, lists, fences, links); translate only the Chinese prose.
Output ONLY the translated Markdown, no preamble.

---SOURCE---
{description}
---END---"""
        try:
            msg = llm.invoke(prompt)
            description_en = (msg.content or "").strip()[:20000]
        except Exception as e:
            logger.exception("详细描述翻译失败")
            raise HTTPException(status_code=500, detail=f"详细描述翻译失败: {str(e)}") from e

    return LinkAiTranslateResponse(
        titleEn=title_en,
        sloganEn=slogan_en,
        descriptionEn=description_en,
    )
