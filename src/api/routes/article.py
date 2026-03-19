"""文章 AI 生成相关 API"""
import os

from fastapi import APIRouter, HTTPException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.core.config import load_config
from src.core.logger import logger
from src.services.image_gen import generate_and_save_cover

router = APIRouter(prefix="/api/article", tags=["article"])


class ArticleAiFillRequest(BaseModel):
    keywords: str


class ArticleAiFillResponse(BaseModel):
    title: str
    summary: str
    cover: str
    categoryName: str
    tagNames: list[str]
    slug: str
    content: str


class _ArticleAiFillOutput(BaseModel):
    """AI 输出结构，供 PydanticOutputParser 解析用"""

    title: str = Field(
        description="文章标题，20-60字，吸引眼球、概括主题"
    )
    summary: str = Field(
        description="文章摘要，80-200字，简要概述文章核心内容，用于列表展示"
    )
    cover: str = Field(
        description="封面图 URL，可选，留空则返回空字符串。若主题适合配图可生成描述性占位如 https://picsum.photos/800/400"
    )
    categoryName: str = Field(
        description="分类名称，如：技术、教程、随笔、产品等，1-4个字"
    )
    tagNames: list[str] = Field(
        description="标签名称列表，3-6个，如 ['Vue','前端','教程']"
    )
    slug: str = Field(
        description="SEO 友好 URL 片段，英文小写+连字符，如 vue3-composition-api-guide"
    )
    content: str = Field(
        description="正文内容，支持 Markdown：## 标题、**粗体**、- 列表、`代码`、```代码块```。800-3000字，结构清晰、有引言和总结"
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
        timeout=600,
        max_retries=1,
    )


@router.post("/ai-fill", response_model=ArticleAiFillResponse)
async def article_ai_fill(req: ArticleAiFillRequest) -> ArticleAiFillResponse:
    """
    根据关键词（一句话或几个词），使用 AI 生成文章标题、摘要、封面、分类、标签、Slug、正文。
    供人工审核后发布。
    """
    keywords = (req.keywords or "").strip()
    if not keywords:
        raise HTTPException(status_code=400, detail="请输入关键词")

    parser = PydanticOutputParser(pydantic_object=_ArticleAiFillOutput)
    format_instructions = parser.get_format_instructions()

    llm = _get_llm()
    prompt = f"""你是一位经验丰富的博客作者。根据以下关键词/主题，生成一篇完整的博客文章内容。

关键词/主题：{keywords}

{format_instructions}

要求：
- title 吸引眼球，适合 SEO 和分享
- summary 简明扼要，适合列表预览
- cover 可为空，若主题适合配图可返回占位 URL
- categoryName 为常见博客分类，如技术、教程、随笔
- tagNames 为相关技术或主题标签，小写或首字母大写
- slug 英文、小写、连字符，用于 URL
- content 使用 Markdown 格式，结构清晰：引言、主体、总结"""

    try:
        msg = llm.invoke(prompt)
        text = (msg.content or "").strip()
    except Exception as e:
        logger.exception("AI 调用失败")
        raise HTTPException(status_code=500, detail=f"AI 服务异常: {str(e)}") from e

    try:
        parsed = parser.parse(text)
        title = (parsed.title or "").strip()[:128] or "未命名文章"
        summary = (parsed.summary or "").strip()[:500] or ""
        cover = (parsed.cover or "").strip()[:512] or ""
        categoryName = (parsed.categoryName or "").strip()[:32] or "未分类"
        tagNames = [str(t).strip() for t in (parsed.tagNames or []) if str(t).strip()][:10]
        slug = (parsed.slug or "").strip()[:128] or ""
        content = (parsed.content or "").strip()[:50000] or ""

        # 使用 doubao-seedream-5.0-lite 生成配图：cover 为空或占位时自动生成
        _is_placeholder = any(
            p in (cover or "").lower()
            for p in ("picsum.photos", "placeholder.com", "via.placeholder", "placehold")
        )
        if not cover or _is_placeholder:
            cover_prompt = f"{title}。{summary[:200]}" if summary else title
            cover_path = generate_and_save_cover(cover_prompt)
            if cover_path:
                cover = cover_path

        return ArticleAiFillResponse(
            title=title,
            summary=summary,
            cover=cover,
            categoryName=categoryName,
            tagNames=tagNames,
            slug=slug,
            content=content,
        )
    except Exception as e:
        logger.warning("AI 返回解析失败: %s", text[:200] if text else "", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI 返回解析失败: {str(e)}") from e
