"""配置加载模块，从 YAML 文件读取配置"""
from pathlib import Path

import yaml

# 项目根目录（py_server/）
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 默认配置文件路径
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """加载 YAML 配置文件，默认查找项目根目录下的 config.yaml"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_server_config(config: dict | None = None) -> dict:
    """获取 server 配置，用于 uvicorn 启动参数"""
    if config is None:
        config = load_config()

    server = config.get("server", {})
    return {
        "host": server.get("host", "0.0.0.0"),
        "port": server.get("port", 8000),
        "workers": server.get("workers", 1),
        "reload": server.get("reload", False),
    }
