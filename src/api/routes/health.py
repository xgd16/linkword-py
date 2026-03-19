"""健康检查及根路由"""
from fastapi import APIRouter

router = APIRouter(tags=["common"])


@router.get("/")
async def root():
    return {"message": "Hello, py-server!"}


@router.get("/health")
async def health():
    return {"status": "ok"}
