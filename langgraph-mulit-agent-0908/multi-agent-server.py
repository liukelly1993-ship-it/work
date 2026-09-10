import asyncio
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from openai.types import embedding_model
from functools import cache
from typing import Any as Any, Any
import uuid

import os
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from typing_extensions import TypedDict, Literal, Annotated

from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.constants import START,END
from langgraph.graph import StateGraph
from operator import add

from langgraph.types import Send
from pydantic import BaseModel, Field

from Director import get_amap_mcp_tools
from 个人.helpers import MiniMaxChatOpenAI, get_minimax_embeddings

config = {'configurable': {'thread_id': str(uuid.uuid4())}}
'''
其他问题，只添加一个简单的响应结果。
娱乐智能体，直接与大模型交互获得一个结果。
对对联智能体，从向量数据库中获取补充的资料，实现一个典型的RAG流程。
路线规划智能体，则需要调度外部的 MCP 服务，获取补充信息
'''
# step1:  初始化llm, 测试llm可用
llm = ChatOpenAI(
    model='MiniMax-M3',
    api_key=os.getenv('MINIMAX_API_KEY'), 
    base_url=os.getenv('MINIMAX_BASE_URL', 'https://api.minimaxi.com/v1'),
    temperature=0.3,
    extra_body={'thinking': {'type': 'disabled'}}
)
result = llm.invoke('hi').content
print(result)

# step2:构建图的参数
class AgentInput(TypedDict):
    input:str

class AgentOutput(TypedDict):
    source: Literal['travel_node', 'joke_node', 'couplet_node', 'other_node']
    result:str


class Classification(TypedDict):
    """单个路由决策：调用哪个agent以及对应的查询"""
    source: Literal["travel_node", "joke_node", "couplet_node", "other_node"]
    query: str

class ClassificationResult(BaseModel):
    """分类器结构化输出（list 语义,支持未来扩展多分类）"""
    classifications: list[Classification] = Field(description="要调用的 agent 列表及其子问题")


class RouterState(TypedDict):
    """主路由状态"""
    query: str #用户输入的提示词
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], add]  # 自动累积结果
    final_answer: str

# step3 旅游智能体
# step3.1 获取高德mcp tools
_mcp_tools = None
_travel_agent = None
async def _get_amap_mcp_tools():
    global _mcp_tools
    if _mcp_tools is None:
        # todo 加双重检验锁
        client = MultiServerMCPClient(
            {
                "amap-maps": {
                    "transport": "streamable-http",
                    "url": "https://mcp.amap.com/mcp?key=204559b07a590786348b11ac11247c17"
                }
            }
        )
        _mcp_tools = await client.get_tools()
        print(f">>>> MCP工具初始化成功，可用工具：{len(_mcp_tools)}个")
    return _mcp_tools

# step3.2 构建旅行规划智能体
async def _get_travel_agent():
    global _travel_agent
    if _travel_agent is None:
        gdmp_tools = await get_amap_mcp_tools()
        _travel_agent = create_agent(
            llm,
            gdmp_tools,
            system_prompt='你是一个旅行规划专家，根据用户的问题，提供详细的旅行路线规划、景点推荐和交通安排。'
        )
    return _travel_agent


#step4 : 四个智能体的具体逻辑
# step4.1 旅行智能体
def travel_node(state: AgentInput) -> AgentOutput:
    # 旅行智能体
    writer = get_stream_writer()
    writer({"node":  ">>> travel_node"})
    writer({"travel_debug": f"正在处理{state['query']}相关的任务..."})
    # 定义一个异步方法invoke agent的方法
    async def async_travel_work():
        try:
            travel_agent = await _get_travel_agent()
            resp = await travel_agent.ainvoke({'messages': state['input']})
            return resp['messages'[-1].content]
        except Exception as e:
            writer(f"旅行节点出错: {e}")
            raise
    #执行异步调用
    try:
        result_content = asyncio.run(async_travel_work)
        writer({'travel_status':'MCP工具调用成功' })
    except Exception as e:
        writer(f"旅行节点出错: {e}")
        response = model.invoke(state["query"])
        result_content: str | list[str | dict[Any, Any]] = response.content
    writer({"travel_result": result_content })
    return {"results": [{"source": "travel_node", "result": result_content}]}

# step4.1 娱乐智能体
def joke_node(state: AgentInput) -> AgentOutput:
    # 娱乐智能体
    writer = get_stream_writer()
    writer({"node": ">>> joke_node"})
    writer({"joke_debug": f"正在处理{state['query']}相关的任务..."})
    joke_response = joke_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    result = joke_response['messages'][-1].content
    return {"results": [{"source": "joke_node", "result": result}]}

# step4.1 对联智能体
# 初始化vector store
@cache
def _get_vector_store():
    return Chroma(collection_name='couplet', embedding_function=get_minimax_embeddings(),persist_directory="./chroma_db")

def couplet_node(state: AgentInput) -> AgentOutput:
    # 对联智能体
    query = state['query']
    writer = get_stream_writer()
    writer({"node": ">>> couplet_node"})
    writer({"couplet_debug": f"正在处理{query}相关的任务..."})
    vector_store = _get_vector_store()
    scored_results = vector_store.similarity_search(query=query, k=10)
    # 推导式简写即可
    samples = [doc.page_content for doc in scored_results]
    #讲向量库获取的数据添加到系统提示词作为少样例参考
    # prompt_template = ChatPromptTemplate.from_messages([
    #     SystemMessagePromptTemplate.from_template(
    #     '你是一个专业的对联大师,你的任务是根据用户给出的上联,设计一个下联,回答是,可以参考下面的参考对联. 参考对联: {samples},请用中文回答问题'),
    #      (HumanMessagePromptTemplate.from_template('{text}'))
    #     ])
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个专业的对联大师,你的任务是根据用户给出的上联,设计一个下联,回答是,可以参考下面的参考对联. 参考对联: {samples},请用中文回答问题"),
        ("user", "{text}")
    ])
    prompt_value = prompt_template.invoke({"samples": samples, "text": query})

    #填充占位符的指,得到完整提示词prompt_value
    prompt_value = prompt_template.invoke({'samples': samples, 'text': query})
    # writer() 这一行执行完就立刻返回了——它做的是入队动作，不是"打印"。graph.astream(..., stream_mode=['values', 'custom']) —— 每个 writer 调用立刻推出来一次
    writer({'couplet_prompt': prompt_value})
    # 此处没有工具直接用model.invoke即可
    resp = model.invoke(prompt_value)
    writer({'couplet_result': resp})
    # return {"results": [{"source": "couplet_node", "result": "对联节点处理的结果！"}]}
    return {"results": [{"source": "couplet_node", "result": resp}]}



def other_node(state: AgentInput) -> AgentOutput:
    # 其他智能体
    writer = get_stream_writer()
    writer({"node": ">>> other_node"})
    writer({"other_debug": f"正在处理{state['query']}相关的任务..."})
    return {"results": [{"source": "other_node", "result": "我暂时无法回答这个问题，但我会尽力提供帮助！"}]}


# 娱乐智能体（joke_node 依赖 joke_agent,必须在所有节点函数定义之前创建）
model = MiniMaxChatOpenAI(
    model='MiniMax-M2.7-highspeed',
    api_key=os.getenv('MINIMAX_API_KEY'),
    base_url=os.getenv('MINIMAX_BASE_URL', 'https://api.minimaxi.com/v1'),
    temperature=0.3,
    extra_body={'thinking': {'type': 'disabled'}}
)
joke_agent = create_agent(
    model,
    [],
    system_prompt='你是一个笑话大师,根据用户的问题,写一个100字以内的笑话,笑点要足并且是中文语境下的笑话,不要讲英文谐音笑话',
)


# 分类方法
def classify_query(state: RouterState) -> AgentOutput:
    print("node: >>> classify_query")

    # json_mode + 嵌套 schema:通过显式示例让 LLM 输出符合预期的嵌套结构
    structured_llm = llm.with_structured_output(ClassificationResult, method="json_mode")
    system_prompt = """你是路由分类器。根据用户查询,选择最匹配的 agent 并生成子问题。

可用来源:
- travel_node: 旅游路线规划、景点推荐、交通安排
- joke_node: 讲笑话、幽默内容
- couplet_node: 对对联、诗词创作
- other_node: 其他问题,通用回答

【严格按以下 JSON 格式输出,只返回 1 个分类】
{
  "classifications": [
    {"source": "xxx_node", "query": "用户原始问题"}
  ]
}
"""
    try:
        # type: ignore[assignment]  # with_structured_output(BaseModel) 静态推断为 dict,实际是 Pydantic 实例
        result = structured_llm.invoke([
            {'role':'system','content':system_prompt},
            {'role':'user','content':state['query']},
        ])
        # type: ignore[return-value]  # 节点实际返回 state 增量,签名沿用 AgentOutput
        return {'classifications': result.classifications}
    except Exception as e:
        # 真正的 API / 解析失败兜底,不再尝试从异常字符串抢救
        print(f"分类失败: {e}")
        # type: ignore[return-value]  # 节点实际返回 state 增量,签名沿用 AgentOutput
        return {'classifications': [{'source': 'other_node', 'query': state['query']}]}

#send并行分发
def route_to_agents(state: RouterState) -> list[Send]:
    """根据分类结果将查询分发到对应agents（支持并行）"""
    result =  [
        # send参数1; 分发的目标节点, 参数2:子任务的状态注入 / 子任务的输入数据载荷
        Send(item['source'] ,{'query':item['query']}) for item in state['classifications']
    ]
    return result

#汇总节点 by Claude
def synthesize_results(state: RouterState) -> dict:
    """合成节点: 先按讲义方式格式化汇总,再流式调用 LLM 综合"""
    writer = get_stream_writer()
    writer({"node": ">>> synthesize_results"})

    if not state["results"]:
        return {"final_answer": "没有从任何知识源找到结果。"}

    # 步骤 1: 按讲义方式格式化(给用户看的「原始汇总」)
    formatted = [f"**[{r['source'].upper()}]** {r['result']}" for r in state["results"]]
    raw_summary = "根据以下信息综合生成一个回答：\n" + "\n".join(formatted)
    writer({"final_summary": raw_summary})

    # 步骤 2: 让 LLM 流式综合(打字机效果)
    final_answer = ""
    for chunk in model.stream(raw_summary):
        if chunk.content:
            final_answer += chunk.content if isinstance(chunk.content, str) else \
                "".join(c.get("text", "") for c in chunk.content if isinstance(c, dict))

    return {"final_answer": final_answer}




# 构建图
builder = StateGraph(RouterState)
builder.add_node('classify_node', classify_query)
builder.add_node('travel_node', travel_node)
builder.add_node('joke_node', joke_node)
builder.add_node('couplet_node', couplet_node)
builder.add_node('other_node', other_node)
builder.add_node('synthesize_results_node', synthesize_results, defer=True)
'''
add_conditional_edges 形参:
源 —起始节点。此条件边将在退出该节点时运行。
路径 -确定下一个或多个节点的可调用对象。如果没有指定path_map，它应该返回一个或多个节点。如果返回‘END’，图形将停止执行。path_map —可选路径到节点名的映射。如果省略，path返回的路径应该是节点名。
'''
builder.add_edge(START,'classify_node')
# 参数1: 源节点, 参数2: 路径函数, 参数3: 目标节点(路径函数返回值与目标节点名的映射关系,名字相同时也可传list或省略)
builder.add_conditional_edges('classify_node', route_to_agents, path_map={'travel_node':'travel_node', 'joke_node':'joke_node', 'couplet_node':'couplet_node', 'other_node':'other_node'})
builder.add_edge('travel_node','synthesize_results_node')
builder.add_edge('joke_node','synthesize_results_node')
builder.add_edge('couplet_node','synthesize_results_node')
builder.add_edge('other_node','synthesize_results_node')
builder.add_edge('synthesize_results_node',END)
graph = builder.compile()
print(graph.get_graph().print_ascii())



# res = graph.invoke({'query': '帮我对个对联，上联是：金榜题名时'}, config, stream_mode='values')
res = graph.invoke({'query': '讲个笑话'}, config, stream_mode='values')
print(res)

#测试对联智能体的可用性
for chunk in graph.stream({'query':'帮我对个对联，上联是：金榜题名时'},config,stream_mode=['custom','messages']):
    print(chunk)


# 旅游规划智能体