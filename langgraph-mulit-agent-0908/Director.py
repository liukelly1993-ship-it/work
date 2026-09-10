from operator import add
from typing_extensions import TypedDict, Annotated,Literal
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from langgraph.types import Send
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
# 嵌入模型改用 helpers 里的 MiniMax 适配，避免依赖 DASHSCOPE_API_KEY
from langchain_chroma import Chroma #%pip install langchain-chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
from 个人.helpers import get_minimax_model, get_minimax_chat_model, get_minimax_embeddings
load_dotenv() # 从 .env 文件加载环境变量
router_llm =get_minimax_chat_model()

# 路由模型
# pip install dashscope
# router_llm = ChatTongyi(model="qwen-plus") # type:ignore
_vector_store = None # 全局变量，存储向量存储

# model = init_chat_model("gpt-4o")
model =get_minimax_chat_model()

# 什么时候使用智能体，什么时候直接使用模型：业务的复杂程度来定，如果涉及多工具的调用可以考虑使用agent，还可以通过中间件来完成harness
joke_agent = create_agent(
    model, 
    tools=[],
    system_prompt="你是一个笑话生成器，请根据用户输入的问题，生成一个笑话。"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
     你是一个专业的对联大师，你的任务是根据用户给出的上联，生成一句下联。
     回答时，可以参考下面的内容：
     {context}
     请用中文回答。
     """),
    ("user", "{text}")
])

class AgentInput(TypedDict):
    """每个子agent的输入状态"""
    query: str                                # 当前任务类型/状态
    
class AgentOutput(TypedDict):
    """每个子agent的输出"""
    source: Literal["travel_agent", "joke_agent", "couplet_agent", "other_agent"]
    result: str

class Classification(TypedDict):
    """单个路由决策：调用哪个agent以及对应的查询"""
    source: Literal["travel", "joke", "couplet", "other"]
    query: str

class RouterState(TypedDict):
    """主路由状态"""
    query: str # 用户输入
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], add]  # 自动累积结果
    final_answer: str
    
# 模型的返回类型
class ClassificationOutput(BaseModel):
    classifications:list[Classification] = Field(..., description="分类结果列表")

# 1、分类节点
def classify(state:RouterState) -> RouterState:
    """根据用户输入的query，分类出对应的任务类型"""
    writer = get_stream_writer()
    writer(f"1-node: classify分类节点")
    system_prompt = """
    你是一个路由分类器，请根据用户输入的问题，进行分类，并返回严格的 JSON 格式。
    类型：
    - travel：旅游路线规划、景点推荐、交通查询等。
    - joke：笑话、段子等。
    - couplet：对对联、诗词等。
    - other：其他类型的问题。
    
    规则：
    - 1. 必须严格按照 JSON 格式返回，格式如下：
    {
        "classifications":[
            {
                "source": "travel",
                "query": "我想去北京旅游，推荐一下路线。"
            },
            {
                "source": "joke",
                "query": "请推荐一个郭德纲的相声。"
            }
        ]
    }
    - 2. source 字段必须为上述四个类型之一。
    - 3. query 字段必须为用户输入的问题，不能进行任何修改。
    - 4. 如果无法分类，请返回空 JSON。

    """
    other = RouterState(
        query=state["query"],
        classifications=[Classification(source="other_agent", query=state["query"])],
        results=[],
        final_answer=""
    )
    
    try:
        structured_llm = router_llm.with_structured_output(ClassificationOutput) #大模型以ClassificationOutput格式输出
        result = structured_llm.invoke([
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":state["query"]}
        ])
        writer(f"1-node: classify result: {result}")
        if result is None:
            return other
        return RouterState(
            query=state["query"],
            classifications=result.classifications,
            results=[],
            final_answer=""
        )
    except Exception as e:
        writer(f"1-node: classify error: {e}")
        return other
# 2、路由分发
def route_to_agents(state:RouterState) ->list[Send]:
    """根据分类结果，将任务分发到对应的节点"""
    return [Send(classification["source"], {"query":classification["query"]}) for classification in state["classifications"]]

def joke_node(state:AgentInput)->RouterState:
    writer = get_stream_writer()
    writer(f"3-node: joke_node笑话节点")
    
    question = state["query"]
    response = joke_agent.invoke(
        {
            "messages":[
                {"role":"user", "content":question}
            ]
        }
    )
    
    content = response["messages"][-1].content
    writer(f"3-node: joke_node result: {content}")
    
    return {"results":[{"source":"joke_agent", "result":content}]} #type:ignore
_mcp_tools = None # 全局变量，存储高德的mcp工具
_travel_agent = None # 全局变量，存储旅游智能体

async def get_amap_mcp_tools():
    global _mcp_tools
    if _mcp_tools is None:
        client = MultiServerMCPClient(
            {
                "amap-maps": {
                    "transport":"streamable-http",
                    "url": "https://mcp.amap.com/mcp?key=204559b07a590786348b11ac11247c17"
                }
            }
        )
        _mcp_tools = await client.get_tools()
        print(f">>>> MCP工具初始化成功，可用工具：{len(_mcp_tools)}个")
        
    return _mcp_tools
    
async def get_travel_agent():
    global _travel_agent
    if _travel_agent is None:
        tools = await get_amap_mcp_tools()
        _travel_agent = create_agent(
            model,
            tools=tools,
            system_prompt="你是一个旅游规划师，请根据用户输入的问题，生成一个旅游路线规划。"
        )
        print(f">>>> 旅行智能体初始化成功！")
    return _travel_agent

def travel_node(state:AgentInput)->RouterState:
    writer = get_stream_writer()
    writer(f"4-node: travel_node旅游节点")
    result = ""
    async def async_travel():
        travel_agent = await get_travel_agent()
        response = await travel_agent.ainvoke({
            "messages":[
                {"role":"user", "content":state["query"]}
            ]
        })
        content = response["messages"][-1].content
        return content
    
    try:
        result = asyncio.run(async_travel()) # 运行异步函数获取结果
        writer(f"4-node: travel_node result: {result}")
    except Exception as e:
        writer(f"4-node: travel_node error: {e}")
        result = "抱歉,旅行规划失败，请稍后再试！"
    
    return {"results":[{"source":"travel_agent", "result":result}]} #type:ignore

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        # 2. 向量模型|嵌入模型
        embedding_model = get_minimax_embeddings() # 使用 MiniMax 平台 embo-01 嵌入

        # 3. 生成向量
        # 4. 向量存储
        _vector_store = Chroma(
            persist_directory="./data/vector_db",#向量存储的位置, 默认是以sqlite文件方式存储
            embedding_function=embedding_model, #指定嵌入模型
            collection_name="couplet" #指定集合名称
        )
    return _vector_store

# 3、定义业务智能体
def couplet_node(state:AgentInput)->RouterState:
    """对对联智能体"""
    writer = get_stream_writer()
    writer(f"5-node: couplet_node对对联节点")
    
    question = state["query"]
    score_results = get_vector_store().similarity_search_with_score(question, k=10)
    
    samples = []
    for doc, _ in score_results:
        samples.append(doc.page_content)
    prompt = prompt_template.invoke({"context":samples, "text":question})
    writer(f"5-node: couplet_node prompt: {prompt}")
    
    result = model.invoke(prompt)
    writer(f"5-node: couplet_node result: {result}")
    
    return {"results":[{"source":"couplet_agent", "result":result.content}]} #type:ignore
def other_node(state:AgentInput)->RouterState:
    writer = get_stream_writer()
    writer(f"6-node: other_node其他节点")
    writer(f"6-node: other_node result: {state['query']}")
    
    return {"results":[{"source":"other_agent", "result":"xxxx"}]} #type:ignore


# 4、汇总结果
def synthesize_results(state:RouterState)->RouterState:
    writer = get_stream_writer()
    writer(f"6-node: synthesize_results汇总结果节点")
    
    # 获取所有结果
    #results = state["results"] # [{"source":"joke_agent", "result":"xxxx"}, ....]
    formatted_results = [f"{r['source']}:{r['result']}" for r in state["results"]]
    synthesize_response = "最终回复的结果如下：" + "\n".join(formatted_results) 
    system_prompt = f"""
    用户提出的问题是：{state["query"]}
    请根据以下内容，生成一个综合回复：
    {synthesize_response}
    请用中文回答。
    """
    result = model.invoke(system_prompt)
    
    writer(f"6-node: synthesize_results results: {result.content}")
    
    return {"final_answer":result.content} #type:ignore


builder = StateGraph(RouterState)
# 构建节点
builder.add_node("classify", classify)
builder.add_node("couplet", couplet_node)
builder.add_node("joke", joke_node)
builder.add_node("travel", travel_node)
builder.add_node("other", other_node)
builder.add_node("synthesize", synthesize_results)

# 构建边
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_to_agents,["couplet", "joke", "travel", "other"])
builder.add_edge("couplet", "synthesize")
builder.add_edge("joke", "synthesize")
builder.add_edge("travel", "synthesize")
builder.add_edge("other", "synthesize")
builder.add_edge("synthesize", END)

# 5、构建流程图
graph = builder.compile()