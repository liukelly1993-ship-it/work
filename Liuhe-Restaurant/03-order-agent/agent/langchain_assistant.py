from dotenv import load_dotenv
from pathlib import Path
from langchain.tools import tool
import pymysql
from pymysql.cursors import DictCursor
from pydantic import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents.middleware import ToolRetryMiddleware, ModelRetryMiddleware
from langchain.messages import ToolMessage, SystemMessage
import os
import sys
import asyncio
import time
import json

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH)) #将项目根目录添加到系统中
from tools.milvus_utils import get_client, get_embedding

load_dotenv()
agent = None
_mcp_tools = None

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

@tool
def search_main_dishes():
    """用来搜索餐厅的特色菜品"""
    #从数据库中获取特色菜品
    with pymysql.connect(
        host=os.getenv("HOST"),
        user=os.getenv("USER"),
        password=os.getenv("PASSWORD"),
        database=os.getenv("DATABASE"),
        port=int(os.getenv("PORT")),
        charset=os.getenv("CHARSET")
    ) as conn: # type:ignore
        with conn.cursor(DictCursor) as cursor:
            cursor.execute("""                                                 
                select 
                    dish_name as 菜品名称, 
                    price as 价格, 
                    description as 描述, 
                    category as 菜品类别, 
                    spice_level as 麻辣程度, 
                    flavor as 口味, 
                    main_ingredients as 主料, 
                    cooking_method as 烹饪方法, 
                    is_vegetarian as 是否素食, 
                    allergens as 过敏源
                    from menu_items
                where 
                    is_featured = 1
            """)
            main_dishes = cursor.fetchall()
    return main_dishes

@tool
def user_favorite_dishes(query:str):
    """根据用户的口味，推荐菜品"""
    client = get_client()
    collection_name = os.getenv("COLLECTION_NAME")
    
    embeddings = get_embedding()
    vector_query = embeddings.embed_query(query)
    
    #向量检索
    search_res = client.search(
        collection_name=collection_name,
        data=[vector_query],
        anns_field="vector",
        output_fields=["text"],
        limit=2
    )
    
    #print(f"搜索结果: {search_res}")
    # 4、解析搜索结果
    if search_res:
        all_results = search_res[0]
        # all_results: 列表
        final_result = []

        for item in all_results:
            dis = item['distance']
            #if dis > 0.5: #阈值判断，根据实际效果调整
            item_str = item["entity"]['text']
            final_result.append(item_str)
        
        return final_result
    else:
        return "在当前库里面没有找到和用户喜好相关的菜品"


class ReservationToolArgsInfo(BaseModel):
    num_people:int = Field(description="预订人数")
    num_children:int = Field(description="预订儿童人数")                                                       
    arrival_time:str = Field(description="到店时间，格式：YYYY-MM-DD HH")
    seat_preference:str= Field(description="座位偏好")        
    main_dish_preference:str=Field(description="主菜偏好")
    other_comments:str= Field(description="其他备注")

@tool(args_schema=ReservationToolArgsInfo)
def make_reservation(num_people,num_children,arrival_time,seat_preference,main_dish_preference,other_comments):
    """进行餐厅预订"""
    try:
        with pymysql.connect(
                host=os.getenv("HOST"),
                user=os.getenv("USER"),
                password=os.getenv("PASSWORD"),
                database=os.getenv("DATABASE"),
                port=int(os.getenv("PORT")),
                charset=os.getenv("CHARSET")
            ) as conn: # type:ignore
                with conn.cursor(DictCursor) as cursor:
                    cursor.execute("""                                                 
                        insert into reservation_order
                        (num_people, num_children, arrival_time, seat_preference, main_dish_preference, other_comments)
                        values (%s, %s, %s, %s, %s, %s)
                    """, (num_people, num_children, arrival_time, seat_preference, main_dish_preference, other_comments))
                    conn.commit()
                    return "预订成功"
    except Exception as e:
        return f"预订失败: {str(e)}"

async def get_agent():
    global agent
    if agent is None:
        from langchain.agents import create_agent
        from langchain.chat_models import init_chat_model
        from langgraph.checkpoint.memory import InMemorySaver
        
        checkpointer = InMemorySaver()
        llm = init_chat_model("gpt-4o")
        with open(str(ROOT_PATH / "agent/prompt/system_prompt.txt"), "r", encoding="utf-8") as f:
            system_prompt = f.read()
        mcp_amap_tools = await get_amap_mcp_tools()
        
        all_tools = [search_main_dishes, user_favorite_dishes,make_reservation]+mcp_amap_tools
        agent = create_agent(
            model=llm,
            tools=all_tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            middleware=[
                ToolRetryMiddleware(
                    max_retries=3, #最大重试次数
                    backoff_factor=2, #退避因子
                    initial_delay=1 #初始延迟
                ),
                ModelRetryMiddleware(
                    max_retries=3, #最大重试次数
                    backoff_factor=2, #退避因子
                    initial_delay=1 #初始延迟
                )
            ]
        )
        
    return agent
        

async def test_agent():
    agent = await get_agent()
    config = {"configurable":{"thread_id":"001"}}
    # res = agent.invoke({
    #     "messages":[
    #         #{"role": "user", "content": "你们餐厅有什么特色菜品吗？"}
    #         #{"role": "user", "content": "我喜欢吃辣的，帮我推荐川菜！"}
    #         {"role": "user", "content": "帮我约定以下明天下午的餐桌，4个人，2个小孩，到店时间下午4点，喜欢靠窗的座位，主菜偏好是红烧肉，备注是不要辣的。"}
    #     ]
    # }, config=config)
    # print(res["messages"][-1].content)
    
    # res = agent.invoke({
    #         "messages":[
    #             {"role": "user", "content": "我确认上面的预订，请帮我确认一下。"}
    #         ]
    #     }, config=config)
    # print(res["messages"][-1].content)
        
        
    # res = search_main_dishes.invoke({})
    # print(res)
    
    
    # res = user_favorite_dishes.invoke({"query":"川菜"})
    # print(res)
    
    res = await agent.ainvoke({
        "messages":[
            {"role": "user", "content": "我现在在中山公园，帮我规划路线如何去你们餐厅？"}
        ]
    }, config=config)
    print(res["messages"][-1].content)        
    
    
#调用Agent使用SSE方式将每个块返回给前端    
async def assistant_query(query:str):
    agent = await get_agent()
    config = {"configurable":{"thread_id":"001"}}
    current_data = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    async for chunk in agent.astream({
        "messages":[
            {"role": "system", "content": f"当前时间：{current_data}"},
            {"role": "user", "content": query}
        ]
    }, config=config, stream_mode="messages"): # type:ignore
        message = chunk[0]
        if type(message) == ToolMessage:
            continue
        #{"content":"内容", "type"："token"}
        payload_str = json.dumps({"content": message.content, "type":"token"}, ensure_ascii=False)
        yield f"data: {payload_str}\n\n"  #"data: {token}\n\n"
    
if __name__ == "__main__":
    asyncio.run(test_agent())