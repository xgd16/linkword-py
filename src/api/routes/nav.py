"""导航向量检索与索引维护 API。"""
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.logger import logger
from src.services.nav_vector import nav_vector_service

router = APIRouter(prefix="/api/nav", tags=["nav"])


class NavVectorItem(BaseModel):
    id: int
    categoryId: int = 0
    title: str = ""
    titleEn: str = ""
    url: str = ""
    icon: str = ""
    cover: str = ""
    slogan: str = ""
    sloganEn: str = ""
    description: str = ""
    descriptionEn: str = ""
    sort: int = 0


class NavVectorUpsertRequest(BaseModel):
    items: list[NavVectorItem] = Field(default_factory=list)


class NavVectorDeleteRequest(BaseModel):
    ids: list[int] = Field(default_factory=list)


class NavVectorSearchRequest(BaseModel):
    keyword: str
    categoryId: int | None = None
    limit: int = 24
    offset: int = 0


class NavVectorSearchItem(BaseModel):
    id: int
    score: float


class NavVectorSearchResponse(BaseModel):
    items: list[NavVectorSearchItem]
    total: int
    truncated: bool = False


def _items_to_dict(items: list[NavVectorItem]) -> list[dict[str, Any]]:
    return [item.model_dump() for item in items]


@router.post("/vector/upsert")
async def upsert_nav_vector(req: NavVectorUpsertRequest) -> dict[str, int]:
    try:
        count = nav_vector_service.upsert(_items_to_dict(req.items))
        return {"count": count}
    except Exception as exc:
        logger.exception("导航向量 upsert 失败")
        raise HTTPException(status_code=500, detail=f"导航向量 upsert 失败: {exc}") from exc


@router.post("/vector/delete")
async def delete_nav_vector(req: NavVectorDeleteRequest) -> dict[str, int]:
    try:
        count = nav_vector_service.delete(req.ids)
        return {"count": count}
    except Exception as exc:
        logger.exception("导航向量 delete 失败")
        raise HTTPException(status_code=500, detail=f"导航向量 delete 失败: {exc}") from exc


@router.post("/vector/rebuild")
async def rebuild_nav_vector(req: NavVectorUpsertRequest) -> dict[str, int]:
    try:
        count = nav_vector_service.rebuild(_items_to_dict(req.items))
        return {"count": count}
    except Exception as exc:
        logger.exception("导航向量 rebuild 失败")
        raise HTTPException(status_code=500, detail=f"导航向量 rebuild 失败: {exc}") from exc


@router.post("/search", response_model=NavVectorSearchResponse)
async def search_nav_vector(req: NavVectorSearchRequest) -> NavVectorSearchResponse:
    try:
        data = nav_vector_service.search(
            keyword=req.keyword,
            category_id=req.categoryId,
            limit=max(1, min(200, req.limit)),
            offset=max(0, req.offset),
        )
        return NavVectorSearchResponse(**data)
    except Exception as exc:
        logger.exception("导航向量 search 失败")
        raise HTTPException(status_code=500, detail=f"导航向量 search 失败: {exc}") from exc
