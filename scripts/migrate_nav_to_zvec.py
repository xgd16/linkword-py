#!/usr/bin/env python3
"""将数据库中的导航数据迁移到 zvec 向量库。

用法:
  # 使用 admin 配置文件中的数据库连接（默认）
  uv run python scripts/migrate_nav_to_zvec.py

  # 指定数据库连接
  uv run python scripts/migrate_nav_to_zvec.py --db-link "root:pass@tcp(127.0.0.1:3306)/linkword?charset=utf8mb4"

  # 指定 py_server 地址
  uv run python scripts/migrate_nav_to_zvec.py --py-server-url http://127.0.0.1:9902

  # 仅读取数据库并打印，不实际迁移
  uv run python scripts/migrate_nav_to_zvec.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import httpx
import pymysql
import yaml


# GoFrame 数据库 link 格式: user:password@tcp(host:port)/dbname?params
_LINK_RE = re.compile(
    r"^(?P<user>[^:]+):(?P<password>[^@]*)@tcp\((?P<host>[^:]+):(?P<port>\d+)\)/(?P<database>[^?]+)"
)


def parse_db_link(link: str) -> dict[str, str | int]:
    """解析 GoFrame 风格的 MySQL 连接字符串。"""
    m = _LINK_RE.match(link.strip())
    if not m:
        raise ValueError(f"无法解析数据库连接字符串: {link}")
    return {
        "user": m.group("user"),
        "password": m.group("password") or "",
        "host": m.group("host"),
        "port": int(m.group("port")),
        "database": m.group("database"),
    }


def load_admin_config(admin_config_path: Path) -> dict:
    """加载 admin 的 config.yaml。"""
    with open(admin_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_nav_links_from_db(db_config: dict) -> list[dict]:
    """从 MySQL 读取 nav_link 全量数据，返回符合 API 格式的列表。"""
    conn = pymysql.connect(
        host=db_config["host"],
        port=db_config["port"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, categoryId, title, title_en, url, icon, cover,
                       slogan, slogan_en, description, description_en, sort
                FROM nav_link
                ORDER BY sort DESC, id ASC
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "categoryId": r.get("categoryId") or r.get("category_id") or 0,
            "title": r["title"] or "",
            "titleEn": r["title_en"] or "",
            "url": r["url"] or "",
            "icon": r["icon"] or "",
            "cover": r["cover"] or "",
            "slogan": r["slogan"] or "",
            "sloganEn": r["slogan_en"] or "",
            "description": r["description"] or "",
            "descriptionEn": r["description_en"] or "",
            "sort": r["sort"] or 0,
        })
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将数据库导航数据迁移到 py_server 的 zvec 向量库"
    )
    parser.add_argument(
        "--db-link",
        default=None,
        help="MySQL 连接字符串，格式: user:pass@tcp(host:port)/dbname",
    )
    parser.add_argument(
        "--admin-config",
        default=None,
        type=Path,
        help="admin config.yaml 路径，不传则自动探测",
    )
    parser.add_argument(
        "--py-server-url",
        default="http://127.0.0.1:9902",
        help="py_server 服务地址",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅读取数据并打印，不发起迁移请求",
    )
    args = parser.parse_args()

    # 1. 确定数据库配置
    if args.db_link:
        db_config = parse_db_link(args.db_link)
    else:
        admin_config_path = args.admin_config
        if not admin_config_path:
            # 默认从 py_server 同级 admin 目录读取
            repo_root = Path(__file__).resolve().parent.parent.parent
            admin_config_path = repo_root / "admin" / "manifest" / "config" / "config.yaml"
        if not admin_config_path.exists():
            print(f"错误: 未指定 --db-link，且找不到 admin 配置: {admin_config_path}")
            return 1
        config = load_admin_config(admin_config_path)
        link = config.get("database", {}).get("default", {}).get("link")
        if not link:
            print("错误: admin 配置中未找到 database.default.link")
            return 1
        db_config = parse_db_link(link)

    # 2. 从数据库拉取 nav_link
    print("正在从数据库读取 nav_link...")
    try:
        items = fetch_nav_links_from_db(db_config)
    except Exception as e:
        print(f"读取数据库失败: {e}")
        return 1

    print(f"共读取 {len(items)} 条导航数据")

    if args.dry_run:
        for i, item in enumerate(items[:5]):
            print(f"  [{i+1}] id={item['id']} title={item['title'][:40]}...")
        if len(items) > 5:
            print(f"  ... 还有 {len(items) - 5} 条")
        print("(dry-run 模式，未发起迁移)")
        return 0

    if not items:
        print("无数据需要迁移")
        return 0

    # 3. 调用 py_server 的 rebuild 接口
    url = f"{args.py_server_url.rstrip('/')}/api/nav/vector/rebuild"
    print(f"正在调用 {url} 进行迁移...")
    try:
        with httpx.Client(timeout=300) as client:
            resp = client.post(url, json={"items": items})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP 错误: {e.response.status_code} - {e.response.text}")
        return 1
    except Exception as e:
        print(f"请求失败: {e}")
        return 1

    count = data.get("count", 0)
    print(f"迁移成功，已导入 {count} 条到 zvec 向量库")
    return 0


if __name__ == "__main__":
    sys.exit(main())
