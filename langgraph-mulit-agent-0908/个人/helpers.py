"""公共模型工厂模块。

集中封装本项目常用的 LLM 客户端创建逻辑，避免在每个 Notebook /
脚本里重复样板代码。所有工厂函数在缺少所需环境变量时返回 ``None``，
由调用方决定降级策略（如退回备用模型或跳过断言）。

设计要点：

- 默认模型与笔记里一直使用的 ``MiniMax-M2.7-highspeed`` 保持一致，
  温度默认 ``0.3``，``base_url`` 读取 ``MINIMAX_BASE_URL``，未设置时
  退回 ``https://api.minimaxi.com/v1``。
- 顶部仅调用一次 :func:`load_dotenv`，供后续 ``os.getenv`` 使用。
- 类型注解一律中文 ``#`` 注释，避免动态类型 stub 噪音。
"""

# 标准库
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

# 第三方
import requests
from dotenv import load_dotenv
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


# 模块导入时即加载 .env，让两个工厂无需再各自调用
load_dotenv()


# MiniMax 默认 base_url，未在环境变量里配置时兜底
_MINIMAX_DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"

# 工厂统一使用的温度，避免每个调用点重复传参
_DEFAULT_TEMPERATURE = 0.3


def _warn(msg: str) -> None:
    """打印统一的 warning 提示，封装 print 以便后续替换日志方案。"""
    print(f"[helpers] WARNING: {msg}")


def _build_minimax_model() -> ChatOpenAI | None:
    """构造 MiniMax-M2.7-highspeed 模型，缺少 KEY 时返回 None。"""
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        _warn("MINIMAX_API_KEY 未设置，无法构造 MiniMax 模型。")
        return None
    # 未设置 base_url 时退回 MiniMax 官方默认地址
    base_url = os.getenv("MINIMAX_BASE_URL") or _MINIMAX_DEFAULT_BASE_URL
    return ChatOpenAI(
        model="MiniMax-M2.7-highspeed",
        api_key=api_key,
        base_url=base_url,
        temperature=_DEFAULT_TEMPERATURE,
    )


# === MiniMax 平台 with_structured_output 适配 ==================================
# MiniMax 平台 OpenAI 兼容 API 在两个层面不兼容 LangChain 的 with_structured_output：
#   1. 新系列 MiniMax-M* 模型强制输出 ``<think>...</think>`` 推理块，污染 JSON 解析；
#   2. 老系列 abab*-chat 模型虽无 think 块，但服务端 schema validation 不支持
#      嵌套结构（Pydantic 嵌套 TypedDict 会生成 ``$ref``，被 MiniMax 拒掉）。
# 解决思路：放弃走原生 schema validation，改用"prompt + 普通 invoke + Markdown
# 包裹剥离 + Pydantic 校验"。这样对调用方无侵入（仍可写
# ``router_llm.with_structured_output(Schema)``，签名/返回值一致）。

# 匹配 ```json ... ``` 或 ``` ... ``` Markdown 代码块
_MD_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_markdown_json(text: str) -> str:
    """剥掉 ```json ... ``` Markdown 代码块包裹，容忍前导/尾随杂质。"""
    if not isinstance(text, str):
        return str(text)
    stripped = text.strip()
    match = _MD_CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    # 兜底：从文本里抓出第一个 {...} 或 [...] JSON 片段
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = stripped.find(start_char)
        end = stripped.rfind(end_char)
        if start >= 0 and end > start:
            return stripped[start : end + 1]
    return stripped


def _inline_schema_refs(schema_dict: dict) -> dict:
    """递归把 Pydantic 嵌套 schema 里的 ``$ref`` 全部内联为 defs 里的真实定义。

    Pydantic 的 :meth:`BaseModel.model_json_schema` 对嵌套模型会输出 ``$defs`` +
    ``$ref`` 引用结构（OpenAI 原生 schema 校验支持），但 MiniMax 老系列 abab*-chat
    会把 ``$defs``/``$ref`` 误当成"待复述的模板"而不是数据描述——输出残缺或被
    截断。内联后 schema 是平铺结构，模型能正确识别为"要输出的数据格式"。

    同时删除 ``title`` / ``description`` 之类不影响结构但会污染输出的元字段。
    """
    import copy

    schema_dict = copy.deepcopy(schema_dict)
    defs = schema_dict.pop("$defs", {})

    # 删除冗余元字段，避免 schema 描述过长导致模型把字段名当模板
    for meta_key in ("title", "description"):
        schema_dict.pop(meta_key, None)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                if ref_name in defs:
                    return _walk(defs[ref_name])
                # 找不到 ref 定义：保留原引用但去掉 $ref（兜底）
                node.pop("$ref")
            cleaned: dict = {}
            for k, v in node.items():
                if k in ("title", "description"):
                    continue
                cleaned[k] = _walk(v)
            return cleaned
        if isinstance(node, list):
            return [_walk(x) for x in node]
        return node

    return _walk(schema_dict)


def _build_schema_description(schema: Any) -> str:
    """把 Pydantic 类 / dict 序列化成 schema 描述文本（嵌套结构已内联）。"""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_dict = schema.model_json_schema()
        schema_dict = _inline_schema_refs(schema_dict)
    elif isinstance(schema, dict):
        schema_dict = _inline_schema_refs(schema)
    elif isinstance(schema, str):
        return schema
    else:
        schema_dict = schema
    return json.dumps(schema_dict, ensure_ascii=False, indent=2)


def _inject_format_instructions(messages: list, format_instructions: str) -> list:
    """把 system / assistant / user 内容合并到单条 user 消息里，并附加 schema 格式说明。

    MiniMax 老系列 abab*-chat 对 ``system`` role 的指令不敏感（system role 容易被
    忽略），但 :class:`langchain_core.messages.SystemMessage` 之类的 system
    消息在 LangChain / OpenAI 协议里是合法的。解决思路：把所有 system 内容
    **合并**到最后一条 user 消息里，作为 ``[System] ...\\n\\n[User] ...`` 的形式
    让模型一次性看到，schema 格式说明追加在末尾。

    对调用方语义透明：LangChain 调用链 / LangGraph graph 里都是按 ``user`` role
    处理；``MiniMax-Text-01`` 在 user role 里遵循指令稳定（含复杂平仄/对仗分析）。
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages + "\n\n" + format_instructions}]

    system_parts: list[str] = []
    other_parts: list[str] = []
    last_user_content: str | None = None

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            other_parts.append(content)
            last_user_content = content
        else:
            other_parts.append(f"[{role}] {content}")

    # 合并：system + 所有对话内容 + schema
    combined_parts: list[str] = []
    if system_parts:
        combined_parts.append("[System]\n" + "\n\n".join(system_parts))
    if other_parts:
        combined_parts.append("[User]\n" + "\n\n".join(other_parts))
    combined_parts.append(format_instructions)
    combined = "\n\n".join(combined_parts)

    return [{"role": "user", "content": combined}]


def _parse_structured_output(content: str, schema: Any) -> Any:
    """把 LLM 返回的字符串解析成 schema 对应的 Pydantic 实例或 Python 对象。"""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_validate_json(content)
    if isinstance(schema, dict):
        return json.loads(content)
    # 其他情况：按 JSON 解析
    return json.loads(content)


class _StructuredRunnable(Runnable):
    """MiniMaxChatOpenAI.with_structured_output 返回的 Runnable。

    对外行为与 LangChain 原生 ``with_structured_output`` 输出一致：
    ``invoke`` / ``ainvoke`` 返回 schema 对应的 Pydantic 实例（或 dict）。
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        schema: Any,
        format_instructions: str,
    ):
        self._llm = llm
        self._schema = schema
        self._format_instructions = format_instructions

    @property
    def InputType(self):
        return Any

    @property
    def OutputType(self):
        return Any

    def invoke(self, input_: Any, config=None, **kwargs) -> Any:
        # 把字符串 / messages 列表标准化后注入格式说明
        if isinstance(input_, str):
            messages = [{"role": "user", "content": input_}]
        elif isinstance(input_, list):
            messages = input_
        else:
            messages = input_
        prepared = _inject_format_instructions(messages, self._format_instructions)
        response = self._llm.invoke(prepared, config=config, **kwargs)
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        return _parse_structured_output(_strip_markdown_json(content), self._schema)

    async def ainvoke(self, input_: Any, config=None, **kwargs) -> Any:
        if isinstance(input_, str):
            messages = [{"role": "user", "content": input_}]
        elif isinstance(input_, list):
            messages = input_
        else:
            messages = input_
        prepared = _inject_format_instructions(messages, self._format_instructions)
        response = await self._llm.ainvoke(prepared, config=config, **kwargs)
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        return _parse_structured_output(_strip_markdown_json(content), self._schema)


class MiniMaxChatOpenAI(ChatOpenAI):
    """MiniMax 平台专用的 ChatOpenAI 子类。

    重写 ``with_structured_output``，绕开 MiniMax OpenAI 兼容 API 的两个限制：

    - 不发 ``response_format=json_schema``（避免嵌套 schema 被服务端 ``$ref`` 校验拒掉）；
    - 改用 prompt 注入 + 普通 invoke，兼容 abab*-chat 无 ``<think>`` 但服务端 schema 校验
      严格的老系列，也兼容 ``<think>`` 会被我们显式 prompt "禁止" 屏蔽的 ``MiniMax-M*`` 系列。

    当前在 ``_build_minimax_chat_model`` 中默认绑定 ``MiniMax-Text-01``（已实测无 think
    块、复杂指令遵循能力强、回答较啰嗦），适合需要平仄/对仗分析等复杂推理的
    聊天场景；如果追求 schema 严格遵循 + 简洁输出，可改 ``abab5.5-chat``。
    """

    def with_structured_output(self, schema: Any, **kwargs) -> Runnable:  # type: ignore[override]
        """返回 :class:`_StructuredRunnable`：invoke/ainvoke 返回 schema 实例。"""
        schema_desc = _build_schema_description(schema)
        format_instructions = (
            "严格按以下 JSON schema 输出（仅输出 JSON 本身，禁止任何 Markdown 代码块、"
            "<think> 块、注释、前后缀解释文字）：\n"
            + schema_desc
        )
        return _StructuredRunnable(self, schema, format_instructions)


# MiniMax embeddings 端点（相对 base_url）
_MINIMAX_EMBEDDINGS_URL = f"{_MINIMAX_DEFAULT_BASE_URL}/embeddings"


class MiniMaxEmbeddings:
    """MiniMax 平台嵌入模型（embo-01），实现 LangChain ``Embeddings`` 接口。

    与官方 ``langchain_community.embeddings.MiniMaxEmbeddings`` 的两点差异：

    1. 不强制要求 ``MINIMAX_GROUP_ID``（MiniMax 平台当前 ``/v1/embeddings``
       端点鉴权只需 ``MINIMAX_API_KEY``，旧库仍传 ``GROUP_ID`` 是历史包袱）；
    2. 不依赖 sunsetting 中的 ``langchain-community``，逻辑直接写在 helpers.py
       里便于排查（与 :class:`MiniMaxChatOpenAI` 风格一致）。

    入库与检索使用**不同的编码**：

    - ``embed_documents``（入库）走 ``type="db"`` 编码；
    - ``embed_query``（在线检索）走 ``type="query"`` 编码。

    同一文本两种 type 会得到不同向量，**必须用对**——调用方不要混用。
    """

    def __init__(
        self,
        model: str = "embo-01",
        batch_size: int = 64,
        api_key: str | None = None,
    ):
        self.model = model
        # 批大小远低于 MiniMax 上限 4096；默认 64 在 1536 维输出下可控且高效
        self.batch_size = batch_size
        self._api_key = api_key or os.getenv("MINIMAX_API_KEY")
        if not self._api_key:
            raise ValueError("MINIMAX_API_KEY 未设置，无法构造 MiniMax embeddings。")

    def _call(self, texts: list[str], type_: str) -> list[list[float]]:
        """调 MiniMax ``/v1/embeddings`` 端点，按 batch_size 分批拉取。"""
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = requests.post(
                _MINIMAX_EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "texts": batch, "type": type_},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            vectors = data.get("vectors")
            if vectors is None:
                base_resp = data.get("base_resp") or {}
                raise RuntimeError(
                    "MiniMax embeddings 调用失败："
                    + base_resp.get("status_msg", "unknown error")
                )
            all_vectors.extend(vectors)
        return all_vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入（用于入库），使用 ``type="query"`` 编码。

        ⚠️ 反直觉设计：MiniMax 官方文档建议入库用 ``type="db"``、检索用
        ``type="query"``，但实测二者处于**不同向量空间**——跨空间余弦相似度
        被压扁到 0.89-0.92 噪声区间，Chroma 检索排序失真（top1 可能比
        top100 还"接近"）。统一用 ``type="query"`` 让入库和检索在同一空间，
        牺牲入库效率换检索精度。
        """
        return self._call(list(texts), type_="query")

    def embed_query(self, text: str) -> list[float]:
        """单条嵌入（用于检索），使用 ``type="query"`` 编码。"""
        return self._call([text], type_="query")[0]


def _build_minimax_chat_model() -> ChatOpenAI | None:
    """构造 MiniMax 平台非推理 chat 模型（默认 ``MiniMax-Text-01``）。

    说明：MiniMax 平台当前所有 ``MiniMax-M*`` 新系列都强制带 ``<think>`` 推理块，
    老系列 ``abab*-chat`` 虽无 think 块但能力一般。``MiniMax-Text-01`` 是
    **唯一同时满足无 think 块 + 复杂指令遵循能力强（适合平仄/对仗分析）
    + 无需 GROUP_ID** 的模型——``MiniMax-M3`` 等推理模型带 think 污染输出，
    ``abab6.5s-chat`` 会把 schema 原文复述、``abab5.5-chat`` 简洁但创意弱。
    所以默认绑 ``MiniMax-Text-01``。

    本工厂返回 :class:`MiniMaxChatOpenAI` 子类实例，其 ``with_structured_output``
    会自动绕开 MiniMax OpenAI 兼容 API 的 schema validation 限制（详见
    :class:`MiniMaxChatOpenAI` 文档）。
    """
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        _warn("MINIMAX_API_KEY 未设置，无法构造 MiniMax chat 模型。")
        return None
    # 未设置 base_url 时退回 MiniMax 官方默认地址
    base_url = os.getenv("MINIMAX_BASE_URL") or _MINIMAX_DEFAULT_BASE_URL
    return MiniMaxChatOpenAI(
        model="MiniMax-Text-01",
        api_key=api_key,
        base_url=base_url,
        temperature=_DEFAULT_TEMPERATURE,
    )


def get_default_model() -> ChatOpenAI | None:
    """获取项目默认模型（MiniMax-M2.7-highspeed）。

    项目里所有模型调用都走 MiniMax-2.7-极速版本（本函数与
    :func:`get_minimax_model` 等价），便于横向对比实验结果。

    Returns:
        已配置好的 :class:`ChatOpenAI` 实例；如果 ``MINIMAX_API_KEY``
        未配置则返回 ``None``，并打印 warning。
    """
    return _build_minimax_model()


def get_minimax_model() -> ChatOpenAI | None:
    """显式获取 MiniMax 模型，与 :func:`get_default_model` 等价。

    保留单独的命名是为了在语义上强调"显式选择 MiniMax"，方便阅读
    调用方代码的意图。同样在缺少 ``MINIMAX_API_KEY`` 时返回 ``None``。
    """
    return _build_minimax_model()


def get_minimax_chat_model() -> ChatOpenAI | None:
    """显式获取 MiniMax 非推理 chat 模型（``MiniMax-Text-01``，已包装 with_structured_output）。

    与 :func:`get_minimax_model` 的关键区别：本函数构造的是**非推理** chat 模型，
    输出纯文本，不会包含 ``<think>...</think>`` 推理块，**适合需要严格结构化
    输出（``with_structured_output`` / 工具调用 + 实时解析）的场景**。

    返回的是 :class:`MiniMaxChatOpenAI` 子类实例：其 ``with_structured_output``
    会自动绕开 MiniMax OpenAI 兼容 API 的 schema validation 限制（详见
    :class:`MiniMaxChatOpenAI` 文档），对调用方完全透明。

    同样在缺少 ``MINIMAX_API_KEY`` 时返回 ``None``。
    """
    return _build_minimax_chat_model()


def get_minimax_embeddings() -> MiniMaxEmbeddings | None:
    """显式获取 MiniMax 平台嵌入模型（embo-01）。

    返回 :class:`MiniMaxEmbeddings` 实例：实现 LangChain ``Embeddings`` 接口
    （``embed_documents`` / ``embed_query``），可直接用于 Chroma / FAISS 等
    向量库的 ``embedding_function`` 参数。与 ``langchain_community`` 内置版本
    相比不依赖 sunsetting 库、不强制要求 ``MINIMAX_GROUP_ID``（详见
    :class:`MiniMaxEmbeddings` 文档）。

    缺少 ``MINIMAX_API_KEY`` 时返回 ``None`` 并打印 warning，调用方可自行降级。
    """
    try:
        return MiniMaxEmbeddings()
    except ValueError as e:
        _warn(str(e))
        return None


__all__ = [
    "MiniMaxChatOpenAI",
    "MiniMaxEmbeddings",
    "get_default_model",
    "get_minimax_model",
    "get_minimax_chat_model",
    "get_minimax_embeddings",
]