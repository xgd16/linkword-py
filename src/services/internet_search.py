"""DuckDuckGo 联网搜索：供 LLM 以工具形式调用，在需要核实事实或补充时效信息时使用。"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from src.core.config import load_config
from src.core.logger import logger


def normalize_model_json_output(raw: str) -> str:
    """去掉模型偶发包裹的 Markdown JSON 代码块，便于 PydanticOutputParser 解析。"""
    t = (raw or "").strip()
    m = re.match(r"^```(?:json)?\s*\n?([\s\S]*?)\n?```\s*$", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _tool_call_args(tc: dict) -> dict:
    args = tc.get("args")
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def get_web_search_settings() -> dict:
    cfg = load_config()
    ws = cfg.get("ai", {}).get("webSearch", {})
    return {
        "enabled": bool(ws.get("enabled", True)),
        "maxResults": max(1, min(15, int(ws.get("maxResults", 6)))),
        "maxToolRounds": max(1, min(10, int(ws.get("maxToolRounds", 4)))),
    }


def search_duckduckgo(query: str, *, max_results: int = 6) -> str:
    """
    执行 DuckDuckGo 文本搜索，返回适合拼进上下文的纯文本摘要。
    """
    q = (query or "").strip()
    if not q:
        return "（未提供搜索关键词，请给出简短明确的查询词。）"

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("未安装 duckduckgo-search，跳过联网搜索")
        return "搜索模块未安装，无法检索。"

    lines: list[str] = []
    try:
        with DDGS(timeout=20) as ddgs:
            for i, r in enumerate(ddgs.text(q, max_results=max_results), 1):
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                href = (r.get("href") or "").strip()
                if not title and not body:
                    continue
                part = f"{i}. {title}\n   {body}"
                if href:
                    part += f"\n   来源: {href}"
                lines.append(part)
    except Exception as e:
        logger.warning("DuckDuckGo 搜索失败: %s", e)
        return f"搜索暂时不可用：{e}"

    if not lines:
        return "未检索到有效结果，可尝试更换关键词（如产品名、公司名 + 功能）。"
    return "\n\n".join(lines)


def build_internet_search_tool(*, max_results: int) -> StructuredTool:
    """构建名为 internet_search 的 LangChain 工具。"""

    def _run(query: str) -> str:
        return search_duckduckgo(query, max_results=max_results)

    return StructuredTool.from_function(
        name="internet_search",
        description=(
            "在互联网上检索公开网页摘要。用于核实事实、产品/公司名称、官网说明、版本或近期动态；"
            "当用户提供的标题、摘要或关键词不足以准确、时效地作答时使用。"
            "参数 query 为简短中文或英文搜索词，可包含站点名+产品名等。"
        ),
        func=_run,
    )


def run_llm_with_internet_search(
    llm: ChatOpenAI,
    *,
    system_prompt: str,
    user_prompt: str,
    enabled: bool,
    max_results: int,
    max_tool_rounds: int,
) -> str:
    """
    在启用时让模型通过 internet_search 自主多轮检索后再输出最终文本；关闭时等价于普通单轮对话。
    """
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    if not enabled:
        msg = llm.invoke(messages)
        return (msg.content or "").strip()

    tool = build_internet_search_tool(max_results=max_results)
    llm_tools = llm.bind_tools([tool], tool_choice="auto")

    rounds = 0
    while rounds < max_tool_rounds:
        rounds += 1
        ai: AIMessage = llm_tools.invoke(messages)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            return (ai.content or "").strip()

        messages.append(ai)
        for tc in tool_calls:
            if isinstance(tc, dict):
                tid = tc.get("id") or ""
                name = tc.get("name") or ""
                args = _tool_call_args(tc)
            else:
                tid = str(getattr(tc, "id", "") or "")
                name = str(getattr(tc, "name", "") or "")
                raw_args = getattr(tc, "args", None)
                args = _tool_call_args({"args": raw_args} if raw_args is not None else {})
            if name == "internet_search":
                q = args.get("query", "")
                out = search_duckduckgo(str(q), max_results=max_results)
            else:
                out = f"未知工具：{name}"
            messages.append(ToolMessage(content=out, tool_call_id=tid))

    # 达到轮次上限：再请求一次纯文本终答（不再绑定工具）
    messages.append(
        HumanMessage(
            content="请根据上文检索结果与已知信息直接给出最终回答，不要再调用任何工具。"
        )
    )
    final = llm.invoke(messages)
    return (final.content or "").strip()
