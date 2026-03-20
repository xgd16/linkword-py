"""轮播图 AI 生图 API"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.logger import logger
from src.services.image_gen import generate_and_save_banner

router = APIRouter(prefix="/api/banner", tags=["banner"])


class BannerAiImageRequest(BaseModel):
    prompt: str = Field(default="", description="画面描述，建议说明风格、主色、文案区域留白等")


class BannerAiImageResponse(BaseModel):
    imageUrl: str


@router.post("/ai-image", response_model=BannerAiImageResponse)
async def banner_ai_image(req: BannerAiImageRequest) -> BannerAiImageResponse:
    """
    根据提示词生成轮播图，保存到 admin 可访问的 public/upload/banners，返回 /upload/banners/xxx.webp。
    """
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="请输入提示词")
    try:
        path = generate_and_save_banner(prompt)
    except Exception as e:
        logger.exception("轮播图生成异常")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}") from e
    if not path:
        raise HTTPException(
            status_code=500,
            detail="图片生成失败，请检查 AI 与 imageGen 配置、模型与尺寸是否匹配",
        )
    return BannerAiImageResponse(imageUrl=path)
