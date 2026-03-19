"""FastAPI HTTP 服务入口"""
import uvicorn

from src.core.config import get_server_config
from src.core.logger import logger


def main():
    """从 config.yaml 读取服务配置并启动"""
    cfg = get_server_config()
    logger.info("workers: %s", cfg["workers"])
    uvicorn.run(
        "src.app:app",
        host=cfg["host"],
        port=cfg["port"],
        workers=cfg["workers"],
        reload=cfg["reload"],
    )


if __name__ == "__main__":
    main()
