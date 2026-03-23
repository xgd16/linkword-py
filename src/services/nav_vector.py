"""导航向量检索与 zvec 索引维护。"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

import httpx
import zvec

from src.core.config import ROOT_DIR, load_config
from src.core.logger import logger

T = TypeVar("T")

def _chunked(items: list[T], size: int) -> list[list[T]]:
    if size <= 0:
        size = 32
    return [items[i : i + size] for i in range(0, len(items), size)]


class NavVectorService:
    """封装 zvec 集合、embedding 调用与搜索能力。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._collection: zvec.Collection | None = None

    def _settings(self) -> dict[str, Any]:
        cfg = load_config()
        ai_cfg = cfg.get("ai", {})
        dmx_cfg = ai_cfg.get("dmxapi", {})
        emb_cfg = ai_cfg.get("embedding", {})
        nav_cfg = cfg.get("navVector", {})

        base_url = str(emb_cfg.get("baseUrl") or dmx_cfg.get("baseUrl") or "").rstrip("/")
        if not base_url:
            raise ValueError("未配置 ai.embedding.baseUrl")

        api_key = (
            os.environ.get("AI_EMBEDDING_API_KEY")
            or os.environ.get("AI_API_KEY")
            or os.environ.get("DMXAPI_API_KEY")
            or emb_cfg.get("apiKey")
            or dmx_cfg.get("apiKey")
            or ""
        )
        if not str(api_key).strip():
            raise ValueError("未配置 ai.embedding.apiKey 或环境变量 AI_EMBEDDING_API_KEY/AI_API_KEY/DMXAPI_API_KEY")

        collection_path = Path(str(nav_cfg.get("collectionPath") or "./data/zvec/nav_links"))
        if not collection_path.is_absolute():
            collection_path = (ROOT_DIR / collection_path).resolve()

        metric_name = str(nav_cfg.get("metricType") or "COSINE").upper()
        metric_map = {
            "COSINE": zvec.MetricType.COSINE,
            "IP": zvec.MetricType.IP,
            "L2": zvec.MetricType.L2,
        }
        metric_type = metric_map.get(metric_name)
        if metric_type is None:
            raise ValueError(f"不支持的 navVector.metricType: {metric_name}")

        return {
            "base_url": base_url,
            "api_key": str(api_key).strip(),
            "model": str(emb_cfg.get("model") or "text-embedding-3-small").strip(),
            "dimensions": max(0, int(emb_cfg.get("dimensions") or 0)),
            "embedding_batch_size": max(1, int(emb_cfg.get("batchSize") or 32)),
            "embedding_timeout": max(10, int(emb_cfg.get("timeout") or 120)),
            "collection_path": collection_path,
            "collection_name": str(nav_cfg.get("collectionName") or "nav_links").strip() or "nav_links",
            "metric_type": metric_type,
            "index_batch_size": max(1, int(nav_cfg.get("batchSize") or emb_cfg.get("batchSize") or 32)),
            "default_topk": max(1, int(nav_cfg.get("defaultTopK") or 120)),
            "search_max_results": max(1, int(nav_cfg.get("searchMaxResults") or 800)),
        }

    @staticmethod
    def _build_embedding_text(item: dict[str, Any]) -> str:
        url = str(item.get("url") or "").strip()
        host = ""
        if url:
            try:
                host = (urlparse(url).hostname or "").replace("www.", "").strip()
            except Exception:
                host = ""

        parts = [
            f"标题: {str(item.get('title') or '').strip()}",
            f"英文标题: {str(item.get('titleEn') or '').strip()}",
            f"标语: {str(item.get('slogan') or '').strip()}",
            f"英文标语: {str(item.get('sloganEn') or '').strip()}",
            f"描述: {str(item.get('description') or '').strip()}",
            f"英文描述: {str(item.get('descriptionEn') or '').strip()}",
            f"域名: {host}",
            f"链接: {url}",
        ]
        return "\n".join(part for part in parts if part and not part.endswith(": "))

    def _embed_texts(self, texts: list[str], settings: dict[str, Any]) -> list[list[float]]:
        if not texts:
            return []

        url = f"{settings['base_url']}/embeddings"
        all_vectors: list[list[float]] = []
        for batch in _chunked(texts, settings["embedding_batch_size"]):
            with httpx.Client(timeout=settings["embedding_timeout"]) as client:
                resp = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {settings['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings["model"],
                        "input": batch,
                    },
                )
                resp.raise_for_status()
                body = resp.json()

            data = body.get("data")
            if not isinstance(data, list):
                raise ValueError("embedding 接口返回缺少 data")

            vectors: list[list[float]] = []
            for row in data:
                emb = row.get("embedding")
                if not isinstance(emb, list) or not emb:
                    raise ValueError("embedding 接口返回无效向量")
                vectors.append([float(v) for v in emb])
            all_vectors.extend(vectors)

        return all_vectors

    def _get_or_create_collection(self, vector_dim: int | None = None) -> zvec.Collection:
        with self._lock:
            settings = self._settings()
            collection_path: Path = settings["collection_path"]
            if self._collection is not None:
                schema_vec = self._collection.schema.vector("embedding")
                if schema_vec and vector_dim and schema_vec.dimension != vector_dim:
                    raise ValueError(
                        f"当前 embedding 维度 {vector_dim} 与现有向量库维度 {schema_vec.dimension} 不一致，请先执行全量重建"
                    )
                return self._collection

            if collection_path.exists():
                self._collection = zvec.open(
                    path=str(collection_path),
                    option=zvec.CollectionOption(read_only=False, enable_mmap=True),
                )
                schema_vec = self._collection.schema.vector("embedding")
                if schema_vec and vector_dim and schema_vec.dimension != vector_dim:
                    raise ValueError(
                        f"当前 embedding 维度 {vector_dim} 与现有向量库维度 {schema_vec.dimension} 不一致，请先执行全量重建"
                    )
                return self._collection

            actual_dim = vector_dim or settings["dimensions"]
            if actual_dim <= 0:
                probe_vecs = self._embed_texts(["导航站向量库维度探测"], settings)
                actual_dim = len(probe_vecs[0])

            collection_path.parent.mkdir(parents=True, exist_ok=True)
            schema = zvec.CollectionSchema(
                name=settings["collection_name"],
                fields=[
                    zvec.FieldSchema(
                        name="category_id",
                        data_type=zvec.DataType.INT64,
                        index_param=zvec.InvertIndexParam(enable_range_optimization=True),
                    ),
                    zvec.FieldSchema(name="title", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(name="url", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(name="slogan", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(name="description", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(
                        name="sort",
                        data_type=zvec.DataType.INT64,
                        index_param=zvec.InvertIndexParam(enable_range_optimization=True),
                    ),
                ],
                vectors=[
                    zvec.VectorSchema(
                        name="embedding",
                        data_type=zvec.DataType.VECTOR_FP32,
                        dimension=actual_dim,
                        index_param=zvec.HnswIndexParam(
                            metric_type=settings["metric_type"],
                            m=16,
                            ef_construction=200,
                        ),
                    )
                ],
            )
            self._collection = zvec.create_and_open(
                path=str(collection_path),
                schema=schema,
                option=zvec.CollectionOption(read_only=False, enable_mmap=True),
            )
            logger.info("已创建导航 zvec 集合: %s", collection_path)
            return self._collection

    @staticmethod
    def _check_status(status: Any, action: str) -> None:
        if hasattr(status, "ok") and callable(status.ok):
            if status.ok():
                return
            code = status.code() if hasattr(status, "code") else "UNKNOWN"
            message = status.message() if hasattr(status, "message") else ""
            raise RuntimeError(f"{action} 失败: {code} {message}".strip())
        if isinstance(status, dict) and status.get("code") == 0:
            return
        raise RuntimeError(f"{action} 返回异常: {status}")

    def upsert(self, items: list[dict[str, Any]]) -> int:
        if not items:
            return 0

        settings = self._settings()
        count = 0
        for batch in _chunked(items, settings["index_batch_size"]):
            texts = [self._build_embedding_text(item) for item in batch]
            vectors = self._embed_texts(texts, settings)
            collection = self._get_or_create_collection(vector_dim=len(vectors[0]))
            docs = [
                zvec.Doc(
                    id=str(item["id"]),
                    vectors={"embedding": vectors[idx]},
                    fields={
                        "category_id": int(item.get("categoryId") or 0),
                        "title": str(item.get("title") or "").strip(),
                        "url": str(item.get("url") or "").strip(),
                        "slogan": str(item.get("slogan") or "").strip(),
                        "description": str(item.get("description") or "").strip(),
                        "sort": int(item.get("sort") or 0),
                    },
                )
                for idx, item in enumerate(batch)
            ]
            result = collection.upsert(docs)
            if isinstance(result, list):
                for status in result:
                    self._check_status(status, "zvec upsert")
            else:
                self._check_status(result, "zvec upsert")
            collection.flush()
            count += len(batch)

        logger.info("导航向量索引已 upsert %d 条", count)
        return count

    def delete(self, ids: list[int]) -> int:
        normalized = [str(i) for i in ids if int(i) > 0]
        if not normalized:
            return 0
        collection = self._get_or_create_collection()
        result = collection.delete(normalized if len(normalized) > 1 else normalized[0])
        if isinstance(result, list):
            for status in result:
                self._check_status(status, "zvec delete")
        else:
            self._check_status(result, "zvec delete")
        collection.flush()
        logger.info("导航向量索引已删除 %d 条", len(normalized))
        return len(normalized)

    def rebuild(self, items: list[dict[str, Any]]) -> int:
        settings = self._settings()
        collection_path: Path = settings["collection_path"]
        with self._lock:
            if self._collection is not None:
                try:
                    self._collection.destroy()
                finally:
                    self._collection = None
            elif collection_path.exists():
                zvec.open(
                    path=str(collection_path),
                    option=zvec.CollectionOption(read_only=False, enable_mmap=True),
                ).destroy()

        self._get_or_create_collection()
        count = self.upsert(items)
        collection = self._get_or_create_collection()
        if count > 0:
            collection.optimize()
        collection.flush()
        logger.info("导航向量索引已全量重建，文档数=%d", count)
        return count

    def search(
        self,
        *,
        keyword: str,
        limit: int,
        offset: int,
        category_id: int | None = None,
    ) -> dict[str, Any]:
        kw = (keyword or "").strip()
        if not kw:
            return {"items": [], "total": 0, "truncated": False}

        settings = self._settings()
        collection = self._get_or_create_collection()
        topk = min(
            settings["search_max_results"],
            max(settings["default_topk"], offset + max(1, limit)),
        )
        query_vector = self._embed_texts([kw], settings)[0]
        filter_expr = None
        if category_id is not None and int(category_id) > 0:
            filter_expr = f"category_id = {int(category_id)}"

        docs = collection.query(
            vectors=zvec.VectorQuery(field_name="embedding", vector=query_vector),
            topk=topk,
            filter=filter_expr,
            output_fields=["sort"],
        )
        items = [
            {
                "id": int(doc.id),
                "score": float(doc.score or 0.0),
                "sort": int((doc.fields or {}).get("sort") or 0),
            }
            for doc in docs
        ]
        items.sort(key=lambda item: (-item["score"], -item["sort"], item["id"]))
        total = len(items)
        sliced = items[offset : offset + limit]
        return {
            "items": [{"id": item["id"], "score": item["score"]} for item in sliced],
            "total": total,
            "truncated": total >= settings["search_max_results"],
        }


nav_vector_service = NavVectorService()
