"""API 路由"""
from src.api.routes.health import router as health_router
from src.api.routes.link import router as link_router
from src.api.routes.nav import router as nav_router

__all__ = ["health_router", "link_router", "nav_router"]
