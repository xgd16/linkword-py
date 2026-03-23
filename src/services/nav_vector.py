"""导航向量检索与 zvec 索引维护。

向量语义以 description / descriptionEn 为主。若扩展了标量字段或改过 schema，
需删除 navVector.collectionPath 对应目录后执行全量 rebuild。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, TypeVar

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
        raw_max_score = nav_cfg.get("maxScore")
        if raw_max_score is None and nav_cfg.get("minScore") is not None:
            raw_max_score = nav_cfg.get("minScore")
        max_score = 0.6 if raw_max_score is None else float(raw_max_score)
        max_score = max(0.0, min(1.0, max_score))

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
            "search_max_score": max_score,
        }

    @staticmethod
    def _build_embedding_text(item: dict[str, Any]) -> str:
        """以 description / descriptionEn 为主；无描述时回退到标题、标语、链接，避免空文本。"""
        desc = str(item.get("description") or "").strip()
        desc_en = str(item.get("descriptionEn") or "").strip()
        parts: list[str] = []
        if desc:
            parts.append(desc)
        if desc_en and desc_en != desc:
            parts.append(desc_en)
        if not parts:
            title = str(item.get("title") or "").strip()
            slogan = str(item.get("slogan") or "").strip()
            url = str(item.get("url") or "").strip()
            if title:
                parts.append(title)
            if slogan:
                parts.append(slogan)
            if url:
                parts.append(url)
        text = "\n".join(parts) if parts else ""
        return text if text.strip() else " "

    _OUTPUT_FIELDS: list[str] = [
        "category_id",
        "title",
        "title_en",
        "url",
        "icon",
        "cover",
        "slogan",
        "slogan_en",
        "description",
        "description_en",
        "sort",
    ]

    @staticmethod
    def _fields_to_search_meta(fields: dict[str, Any] | None) -> dict[str, Any]:
        f = fields or {}
        return {
            "categoryId": int(f.get("category_id") or 0),
            "title": str(f.get("title") or ""),
            "titleEn": str(f.get("title_en") or ""),
            "url": str(f.get("url") or ""),
            "icon": str(f.get("icon") or ""),
            "cover": str(f.get("cover") or ""),
            "slogan": str(f.get("slogan") or ""),
            "sloganEn": str(f.get("slogan_en") or ""),
            "description": str(f.get("description") or ""),
            "descriptionEn": str(f.get("description_en") or ""),
            "sort": int(f.get("sort") or 0),
        }

    @staticmethod
    def _summarize_items(items: list[dict[str, Any]], max_items: int = 8) -> str:
        if not items:
            return "[]"
        parts: list[str] = []
        for item in items[:max_items]:
            title = str(item.get("title") or item.get("titleEn") or "").strip() or f"id={item.get('id')}"
            score = float(item.get("score") or 0.0)
            parts.append(f"{item.get('id')}:{score:.4f}:{title}")
        if len(items) > max_items:
            parts.append(f"...(+{len(items) - max_items})")
        return "[" + " | ".join(parts) + "]"

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
                    zvec.FieldSchema(name="title_en", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(name="url", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(name="icon", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(name="cover", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(name="slogan", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(name="slogan_en", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(name="description", data_type=zvec.DataType.STRING, nullable=True),
                    zvec.FieldSchema(name="description_en", data_type=zvec.DataType.STRING, nullable=True),
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
                        "title_en": str(item.get("titleEn") or "").strip(),
                        "url": str(item.get("url") or "").strip(),
                        "icon": str(item.get("icon") or "").strip(),
                        "cover": str(item.get("cover") or "").strip(),
                        "slogan": str(item.get("slogan") or "").strip(),
                        "slogan_en": str(item.get("sloganEn") or "").strip(),
                        "description": str(item.get("description") or "").strip(),
                        "description_en": str(item.get("descriptionEn") or "").strip(),
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
        logger.info(
            "导航向量检索开始 keyword=%r category_id=%s limit=%d offset=%d topk=%d max_score=%.4f filter=%s",
            kw,
            category_id,
            limit,
            offset,
            topk,
            settings["search_max_score"],
            filter_expr or "<none>",
        )

        docs = collection.query(
            vectors=zvec.VectorQuery(field_name="embedding", vector=query_vector),
            topk=topk,
            filter=filter_expr,
            output_fields=self._OUTPUT_FIELDS,
        )
        raw_items: list[dict[str, Any]] = []
        for doc in docs:
            fields = doc.fields or {}
            meta = self._fields_to_search_meta(fields if isinstance(fields, dict) else {})
            raw_items.append(
                {
                    "id": int(doc.id),
                    "score": float(doc.score or 0.0),
                    **meta,
                }
            )
        logger.info(
            "导航向量原始候选 raw_count=%d top=%s",
            len(raw_items),
            self._summarize_items(raw_items),
        )
        max_score = settings["search_max_score"]
        items: list[dict[str, Any]] = []
        for item in raw_items:
            score = float(item["score"] or 0.0)
            if score > max_score:
                continue
            items.append(item)
        items.sort(key=lambda item: (item["score"], -item["sort"], item["id"]))
        total = len(items)
        sliced = items[offset : offset + limit]
        logger.info(
            "导航向量过滤完成 kept=%d returned=%d top=%s",
            total,
            len(sliced),
            self._summarize_items(sliced),
        )
        return {
            "items": sliced,
            "total": total,
            "truncated": total >= settings["search_max_results"],
        }


nav_vector_service = NavVectorService()
