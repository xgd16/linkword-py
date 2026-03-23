"""FastAPI 应用定义"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import health_router
from src.api.routes.article import router as article_router
from src.api.routes.banner import router as banner_router
from src.api.routes.link import router as link_router
from src.api.routes.nav import router as nav_router
from src.core.config import ROOT_DIR

app = FastAPI(title="py-server")

# 配图静态文件服务：uploads/covers 下的 webp 可通过 /uploads/ 访问
uploads_dir = ROOT_DIR / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# 注册路由
app.include_router(health_router)
app.include_router(link_router)
app.include_router(nav_router)
app.include_router(article_router)
app.include_router(banner_router)
